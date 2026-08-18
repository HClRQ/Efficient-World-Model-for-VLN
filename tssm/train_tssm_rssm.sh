#!/bin/bash
#SBATCH --job-name=tssm_rssm
#SBATCH --partition=gpuhe.24h
#SBATCH --gpus=nvidia_geforce_rtx_4090:2
#SBATCH --exclude=eu-g3-[062-077,083-084],eu-lo-g3-[049-061]   # 08-07: 放宽机型(4090/3090皆可), 只排除Turing(无bf16)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=14G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/project/rsl/ruiqzhang/codes/NaVILA/logs/tssm_rssm_%j.out
#SBATCH --error=/cluster/project/rsl/ruiqzhang/codes/NaVILA/logs/tssm_rssm_%j.out
set -eo pipefail
# 真·RSSM档 (08-08 用户拍板): GRU核(h_t=GRU(h_{t-1},[z_{t-1},a_{t-1}]), Dreamer正统),
# 其余与ncev5逐位相同。trained-in-regime回答"为何不需要recurrent": 若W1的NCE检索
# 训出来仍崩, 则递归瓶颈是训练层面的, 不是手术伪影。原头注释: ncev5配方(10万步 NCE=8@3/5/10)单变量
# +未来InfoNCE: 给GT动作串(未来窗嵌入均值), state_t在全集帧认出t+k真身(动作条件CPC)。
# 判决: ①retro命中不掉(vs ncev5同口径lag_sweep) ②fut_nce_acc显著高于随机 → 判别式想象无害/有益。
# 原头注释:
# TransDreamer 原文 §3.1 点名但因复杂度放弃的设计: q(z_t | f_transformer(x_1:t, a_1:t))。
# 观测塔(2层d512, 5.5M)吃给定观测序列 -> 无递归、全并行; z 从"孤帧快照"变"信念码";
# KL 靶子含历史后, prior(h)→z 首次真正逼 h 编码历史(直击 state≈当前帧预测器病根)。
# 配方 = v10R (RECON_W=1.0 RETRO_W=8 无latent) + posterior_ctx=transformer, 单变量。
# 判据: ck35000 对 v10R(Δmem+0.0455/RETR_IN 0.9204/对照+0.0400) 同步数过闸。
# 冒烟已验: 梯度流通; 平行训练路径 vs 逐步推理路径 ctx 逐位一致(1e-5)。
source /cluster/project/rsl/ruiqzhang/miniconda3/etc/profile.d/conda.sh
conda activate navila
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cluster/project/rsl/ruiqzhang/codes/NaVILA
export HF_HOME=/cluster/scratch/ruiqzhang/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/cluster/project/rsl/ruiqzhang/codes/NaVILA:/cluster/project/rsl/ruiqzhang/codes/NaVILA/RSSM_v4
unset CONSUL_HTTP_ADDR
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '^/var/run' | paste -sd ':')

SAVE_DIR=${SAVE_DIR:-/cluster/scratch/ruiqzhang/tssm_rssm_ckpt}
OUT_PT=${OUT_PT:-/cluster/scratch/ruiqzhang/tssm_rssm.pt}
ANNOT_R2R=/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json
ANNOT_RXR=/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/annotations.json
FEAT_R2R=/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip8f_p9_lm2
FEAT_RXR=/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/feat/siglip8f_p9_lm2
# 关键旋钮(可 env 覆盖)
FUTURE_W=${FUTURE_W:-0.0}       # 多步未来预测权重(新主信号)
FUTURE_K=${FUTURE_K:-10}         # 开环步数(记忆压力旋钮; 与rollout闸h<=6对齐)
FUTURE_NS=${FUTURE_NS:-1}       # 起点数(batch=64已给多样性; 8->2 提速~3x)
RECON_W=${RECON_W:-1.0}         # 当前重建降权(从主导→辅助)
RETRO_W=${RETRO_W:-0.0}         # 与v8实跑一致
RETRO_LAGS=${RETRO_LAGS:-1,2,5,10,20}   # 多滞后: 1步太弱, 20步才逼长程保留
FWD_W=${FWD_W:-0.0}               # v9L 关掉观测空间前向
LATENT_W=${LATENT_W:-0.0}         # 隐空间前向; 配平到与retro等**贡献**(fable④)
LATENT_LAGS=${LATENT_LAGS:-5,10,20}
ACTX=${ACTX:-gru}
FWD_LAGS=${FWD_LAGS:-1,2,5,10,20}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_STEPS=${MAX_STEPS:-100000}
LR=${LR:-3e-4}

mkdir -p "$SAVE_DIR" logs
for f in "$FEAT_R2R/features.pt" "$FEAT_RXR/features.pt" "$ANNOT_R2R" "$ANNOT_RXR"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
LATEST=$(ls -t "$SAVE_DIR"/checkpoint-*.pt 2>/dev/null | grep -v final | head -1 || true)
[ -n "$LATEST" ] && { RESUME=(--resume "$LATEST"); echo "Resume $LATEST"; } || { RESUME=(); echo "Fresh futnce"; }
echo "=== v11(观测塔posterior) LATENT_W=$LATENT_W lags=$LATENT_LAGS actx=$ACTX RETRO_W=$RETRO_W RECON_W=$RECON_W ==="

torchrun --nproc_per_node=2 --master_port=$((30000 + SLURM_JOB_ID % 5000)) RSSM_v4/train_tssm_feat.py \
    --annotation "$ANNOT_R2R,$ANNOT_RXR" \
    --cache_path "$FEAT_R2R,$FEAT_RXR" \
    --save_dir "$SAVE_DIR" --full_episode \
    --batch_size "$BATCH_SIZE" --num_workers 3 \
    --max_steps "$MAX_STEPS" --lr "$LR" --kl_warmup 10000 \
    --log_every 50 --ckpt_every 5000 \
    --free_nats 1.0 --kl_balance 0.9 --kl_scale 0.4 \
    --imagination_loss_weight 0.0 \
    --future_weight "$FUTURE_W" --future_horizon "$FUTURE_K" --future_n_starts "$FUTURE_NS" ${FUTURE_SGDEC:+--future_stopgrad_decoder} \
    --recon_weight "$RECON_W" \
    --retro_weight "$RETRO_W" --retro_lags "$RETRO_LAGS" --retro_n_starts 4 \
    --fwd_weight "$FWD_W" --fwd_lags "$FWD_LAGS" --fwd_n_starts 4 \
    --latent_weight "$LATENT_W" --latent_lags "$LATENT_LAGS" --latent_n_starts 4 \
    --aux_actx "$ACTX" --fast_aux \
    --nce_weight 8.0 --nce_lags "3,5,10" --core_type gru --nce_tau ${NCE_TAU:-0.1} ${NCE_FOURIER:+--nce_lag_fourier} ${NCE_SAMPLE:+--nce_lag_sample $NCE_SAMPLE} ${NCE_GTAU:+--nce_grade_tau $NCE_GTAU} ${SLOW_MODS:+--slow_modules "$SLOW_MODS"} ${NCE_HH:+--nce_head_hidden $NCE_HH} ${NCE_HL:+--nce_head_layers $NCE_HL} ${NCE_DIM:+--nce_dim $NCE_DIM} ${NCE_EVENT:+--nce_event_mode} ${NCE_EQ:+--nce_event_quantile $NCE_EQ} \
    --action_inv_weight 0.1 --pose_delta_weight 0.1 \
    --d_model 512 --n_heads 8 --n_layers 4 --ffn_dim 2048 --transformer_dropout 0.1 \
    --deter_type concat_o --recon_type normal --rec_sigma 2.0 --recon_groups 9 \
    "${RESUME[@]}"

[ -f "$SAVE_DIR/checkpoint-final.pt" ] && cp -f "$SAVE_DIR/checkpoint-final.pt" "$OUT_PT" && echo "-> $OUT_PT"
echo "End: $(date)"
