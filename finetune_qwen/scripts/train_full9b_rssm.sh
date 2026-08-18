#!/bin/bash
#SBATCH --job-name=q9b_full
#SBATCH --partition=cuda13pr.120h
#SBATCH --gpus=nvidia_rtx_pro_6000:4
#SBATCH --constraint=EPYC_9654
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=120:00:00
#SBATCH --output=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/logs/qfull9b_%j.out
#SBATCH --error=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/logs/qfull9b_%j.err
# Qwen3.5-9B 全量FT (08-06 用户拍板): 同参最强底(4B榜单赢VL-30B), 4×PRO6000, 配方=full8s全同
#   变量表 vs 第一发(full4): +RxR数据源 / state v885k→ncev5混合 / +READNCE(答案锚)
#   不做stop过采样(FutureNav没做, 我们也不做 — 用户令)。
set -eo pipefail
unset CONSUL_HTTP_ADDR
module load cuda/13.0.2   # deepspeed JIT编译FusedAdam需要CUDA_HOME(full4的wrap带了, 首发脚本漏了)
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '^/var/run' | paste -sd ':')
cd /cluster/project/rsl/ruiqzhang/codes/qwen35-vln
PY=/cluster/project/rsl/ruiqzhang/miniconda3/envs/vllm/bin
export PATH="$PY:$PATH"
export PYTHONPATH="/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/bw_overlay:${PYTHONPATH}"   # Blackwell修复: fla 0.5.2覆盖0.5.1(gated_delta_rule sm_120 misaligned病)
export HF_HOME=/cluster/scratch/ruiqzhang/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHELASTIC_ERROR_FILE=/cluster/scratch/ruiqzhang/q9b_full_err.json
# 08-16 机型自适应(抄pro8): 本线可落A100(sm_80)或PRO6000(sm_120), 按实卡设编译目标
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPUNAME" in
  *PRO*6000*|*RTX*PRO*)
    export TORCH_EXTENSIONS_DIR=/cluster/scratch/ruiqzhang/torch_ext_sm120
    export TORCH_CUDA_ARCH_LIST="12.0"
    export TRITON_MAX_WARPS=4 ;;   # Blackwell sm_120 num_warps=8病
  *)
    export TORCH_EXTENSIONS_DIR=/cluster/scratch/ruiqzhang/torch_ext_sm80
    export TORCH_CUDA_ARCH_LIST="8.0" ;;
esac
echo "实测显卡: ${GPUNAME:-未知} → ARCH=$TORCH_CUDA_ARCH_LIST"

# 08-15 9B-RSSM净血发(用户拍板): 公版塔RSSM state, 从公开基座全新起跑, 冠军配方全同
export QWEN_DIR=$(ls -d /cluster/scratch/ruiqzhang/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
export STAGE1_PROJ=none
export N_MEM=16
export STATE_CACHE=/cluster/scratch/ruiqzhang/tssm_states_rssmpub_full.pt
export ANN=/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json
export ANN_RXR=/cluster/scratch/ruiqzhang/streamvln_data/annotations_stream_v5.json
export ANN_RXR2=
export SQLITE_RXR2=/cluster/scratch/ruiqzhang/rxr_expand/expand.sqlite
export SQLITE=/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/train.sqlite
export SQLITE_RXR=/cluster/scratch/ruiqzhang/streamvln_data/RxR/stream_rxr.sqlite
export READNCE=1
export READNCE_QANCHOR=all   # 冠军口径(08-01): 全答案位+逐位置查询, 对齐rncenp
export READNCE_DYN_LAGS=1,3,5   # 08-06: dyn未来检索延到5步(retro已3/5/10, TWISTER长视距证据)
export READNCE_ANCHOR_LAYER=11   # retro锚: 34%深度全注意力(REPA甜点)
export READNCE_DYN_ANCHOR_LAYER=23  # dyn锚(08-07用户拍板): 72%深度全注意力, 预想考靠近决策端
export READNCE_KEYS=/cluster/scratch/ruiqzhang/tssm_readnce_keys_rssmpub.pt
export READNCE_ACTS=/cluster/scratch/ruiqzhang/tssm_action_seqs_full8.pt
export DS_CONFIG=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/configs/zero2.json
export OUT=/cluster/scratch/ruiqzhang/qwen9b_rssmpub
# 满1epoch(用户拍板): (353894+417578)/64 = 12054步; LR按NaVILA官方线性缩放:
# 1e-4 @ 全局批160 → 4e-5 @ 64 (full4的1e-5比官方等效低4倍, 疑似欠拟合元凶之一)
export NPROC=${NPROC:-8}
export ACCUM_UNUSED=8   # 2卡暗马: 2×4×8=64全局批不变, ~16.6s/步 98.7h贴墙
export MAX_STEPS=${MAX_STEPS:-21404}   # (353894+1015940)/64 full8满epoch
export LR=${LR:-2e-5}   # 08-03四家对表定案: 批64共识2e-5(旧4e-5过热)
export SAVE_STEPS=${SAVE_STEPS:-1000}
# 08-09 提速手术(用户拍板): bs4×8卡×accum2=64全局批不变, 吞吐≈1.7×(显存53G→预估~68G/80G)
# OOM保险: 若上一棒bs4曾OOM(哨兵文件在), 本棒自动回落bs2×accum4
if [ -f "/cluster/scratch/ruiqzhang/qwen9b_rssmpub/.bs4_oom" ]; then
  echo "⚠ bs4曾OOM, 本棒回落 bs2 (ACCUM按卡数配平保全局批64)"
  export PER_DEVICE_BS=${PER_DEVICE_BS:-2}
  export ACCUM=${ACCUM:-$((32 / ${NPROC:-8}))}
else
  export PER_DEVICE_BS=${PER_DEVICE_BS:-4}
  export ACCUM=${ACCUM:-2}
fi

echo "=== Job $SLURM_JOB_ID Node $SLURMD_NODENAME ==="
# 08-09 双分区赛跑互斥锁: A100棒与PRO6000棒抢同一ds存档, 只许一棒在跑。
# 陈旧锁(持有者已不在RUNNING)可被接管 → 最坏退化成今天的行为, 不会锁死链条。
mkdir -p "$OUT"   # 08-16教训: 新线目录必须自建, 不能赌历史遗留
LOCK=/cluster/scratch/ruiqzhang/qwen9b_rssmpub/.train.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  OTHER=$(cat "$LOCK/jobid" 2>/dev/null)
  if [ -n "$OTHER" ] && squeue -h -j "$OTHER" -o "%T" 2>/dev/null | grep -q RUNNING; then
    # 让位: 优先 requeue 把名额退回队列(保住优先级, 下次再抢), requeue 不成才退出
    echo "另一棒 $OTHER 正在训练, 本棒让位(不动存档)"
    # 名额不能白烧: 补一张挂依赖的替补票(afterany:在跑棒)——在跑棒不结束它不上车。
    # 08-09实弹教训: 不挂依赖时, 空卡即上车→让位→再补投, 一小时烧光40次上限。
    CNT=/cluster/scratch/ruiqzhang/qwen9b_rssmpub/.resubmit_count
    N=$(cat "$CNT" 2>/dev/null || echo 0)
    if [ "$N" -lt 12 ]; then
      echo $((N + 1)) > "$CNT"
      bash /cluster/project/rsl/ruiqzhang/codes/qwen35-vln/scripts/submit_hop.sh \
        "${HOP_FLAVOR:-a100}" --dependency=afterany:$OTHER \
        && echo "已补投挂依赖替补票(afterany:$OTHER, 累计 $((N + 1)) 次)"
    else
      echo "补投已达上限12次, 不再补投(防失控)"
    fi
    exit 0
  fi
  echo "接管陈旧锁(前持有者 ${OTHER:-未知} 已不在跑)"
fi
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
trap 'rm -rf "$LOCK"' EXIT
RC=0
$PY/torchrun --nnodes=1 --nproc_per_node=${NPROC:-4} --master_port=${MASTER_PORT:-29621} \
  train_stage2.py || RC=$?
LOGF=$(ls -t /cluster/project/rsl/ruiqzhang/codes/qwen35-vln/logs/qfull9b_*_${SLURM_JOB_ID}.out 2>/dev/null | head -1)
# 08-09 就地降档重试: bs4 撞显存不必把整棒赔进去(重排一次要等十几小时),
# 趁本作业还握着这8张卡, 立刻用 bs2/accum4 从最近ds存档续跑, 只损失几分钟。
if [ "$RC" != "0" ] && [ "${PER_DEVICE_BS}" = "4" ] \
   && grep -q "OutOfMemoryError" "$LOGF" 2>/dev/null; then
  touch /cluster/scratch/ruiqzhang/qwen9b_rssmpub/.bs4_oom
  echo "⚠ bs4 撞显存 → 就地降档 bs2/accum4 重试(同一分配, 不重排队)"
  export PER_DEVICE_BS=2 ACCUM=$((32 / ${NPROC:-8}))
  RC=0
  $PY/torchrun --nnodes=1 --nproc_per_node=${NPROC:-4} --master_port=$((${MASTER_PORT:-29621} + 1)) \
    train_stage2.py || RC=$?
fi
if [ "$RC" != "0" ] && grep -qiE "No module named .causal_conv1d|ImportError.*causal_conv1d" "$LOGF" 2>/dev/null; then
  /cluster/project/rsl/ruiqzhang/miniconda3/envs/vllm/bin/pip uninstall -y causal-conv1d
  echo "坏内核自愈: causal-conv1d已卸载, 下一棒回落torch慢速路径"
fi
if [ "$RC" != "0" ] && grep -q "OutOfMemoryError" "$LOGF" 2>/dev/null; then
  touch /cluster/scratch/ruiqzhang/qwen9b_rssmpub/.bs4_oom
  echo "OOM哨兵已立, 下一棒自动回落bs2"
fi
echo "End: $(date)"
