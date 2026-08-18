#!/bin/bash
# RNCEANS 臂 (07-30 用户拍板): 检索锚点从记忆位置搬到答案位置: 冠军配方 + ncev5 state + 读出检索损失。
# RSSMDS臂 (08-11 用户拍板): GRU真递归state + 9B全量深监督配方(双中间层锚+dyn三档)。
#   单变量链: vs rnceans差三处(state源rssm/锚层11+23/dyn加5档)——主判据是"GRU state当LLM记忆是否更强"。
#   动机: 机制链证明 ncev5 state 里历史"在且可读"(90%检索命中), 但 LoRA 只有
#   动作CE, 无损失逼LLM去读 → nceL ck1000 与对照打平。本臂加两个查询头:
#     retro头: LLM在记忆token位置的隐状态 → 冻结ncev5钥匙空间里检索t-k帧;
#     dyn头:   隐状态+动作窗嵌入 → 检索t+k帧(FutureNav式动力学grounding)。
#   单变量对照: 除 TSSM_READNCE 家族外与 ncelora(-a4) 逐项一致, 全新LoRA。
#SBATCH --job-name=rssmds
#SBATCH --gpus=nvidia_geforce_rtx_4090:2
#SBATCH --constraint=EPYC_9554
#SBATCH --partition=gpuhe.24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rssmds_%j.out
#SBATCH --error=logs/rssmds_%j.err
set -eo pipefail
source /cluster/project/rsl/ruiqzhang/miniconda3/etc/profile.d/conda.sh
conda activate navila
cd /cluster/project/rsl/ruiqzhang/codes/NaVILA
export PYTHONPATH="/cluster/project/rsl/ruiqzhang/codes/NaVILA:${PYTHONPATH}"
unset CONSUL_HTTP_ADDR
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '^/var/run' | paste -sd ':')
export NCCL_P2P_DISABLE=${NCCL_P2P:-1}
export NCCL_IB_DISABLE=1
echo "=== Job $SLURM_JOB_ID Node $SLURMD_NODENAME ==="

BASE_MODEL=/cluster/project/rsl/ruiqzhang/codes/NaVILA/navila_8f
OUTPUT=${OUTPUT:-/cluster/scratch/ruiqzhang/checkpoints/navila-stage2-rssmds-lora}
MIX_CACHE=/cluster/scratch/ruiqzhang/tssm_states_rssm_r2r_rxr.pt
export NAVILA_R2R_IMAGE_PATH=/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/train.sqlite
export NAVILA_RXR_IMAGE_PATH=/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/train.sqlite

export TSSM_PROJ_MODE=nav3l
export TSSM_FREEZE_PROJECTOR=0
export TSSM_NAV3L_WARMUP=300
export TSSM_ACTION_SEQS=/cluster/scratch/ruiqzhang/tssm_action_seqs.pt
unset TSSM_STAGE1_PROJECTOR_PATH
export TSSM_PROJ_SLICE=0
export TSSM_PROJ_XATTN=0
export TSSM_HISTORY_STATES=1
export TSSM_HISTORY_SPAN=full
export TSSM_PSEUDO_FRAMES=0
export TSSM_KL_ANCHOR=0
export TSSM_IMAGE_DROPOUT_PROB=0.0
export TSSM_BLACK_AUX=1
export TSSM_NARR_R2R_ANNOTATIONS=/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/tssm_semantic_narration_fgr2r.json
export TSSM_NARR_RXR_ANNOTATIONS=/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/tssm_semantic_narration_official.json

# ---- 读出检索 (本臂唯一新变量) ----
export TSSM_READNCE=1
export TSSM_READNCE_ANCHOR=answer   # 07-30 答案位锚定: 出动作的隐状态里必须查得出历史(焊死最后一跳)
export TSSM_READNCE_KEYS=/cluster/scratch/ruiqzhang/tssm_readnce_keys_rssm.pt
export TSSM_READNCE_RETRO_LAGS=${RETRO_LAGS:-3,5,10}
export TSSM_READNCE_DYN_LAGS=${DYN_LAGS:-1,3,5}
export TSSM_READNCE_W=${READNCE_W:-0.2}
export TSSM_READNCE_DYN_W=${DYNNCE_W:-0.2}
export TSSM_READNCE_TAU=0.1
export TSSM_READNCE_ANCHOR_LAYER=${RN_LAYER:-11}   # 08-11 9B配方: retro锚34%深度
export TSSM_READNCE_DYN_ANCHOR_LAYER=${DYN_LAYER:-23} # dyn锚72%深度

# ---- val 刹车 ----
export TSSM_VAL_HOLDOUT_FILE=/cluster/project/rsl/ruiqzhang/codes/NaVILA/RSSM_v4/val_holdout_scenes_v1.json
export TSSM_VAL_MAX=1000

[ -f "$MIX_CACHE" ] || { echo "ERROR: ncev5 cache missing"; exit 1; }
[ -f "$TSSM_READNCE_KEYS" ] || { echo "ERROR: readnce keys missing"; exit 1; }
[ -f "$TSSM_ACTION_SEQS" ] || { echo "ERROR: action seqs missing"; exit 1; }
[ -f "$TSSM_VAL_HOLDOUT_FILE" ] || { echo "ERROR: holdout list missing"; exit 1; }
echo "READNCE arm: retro=${TSSM_READNCE_RETRO_LAGS} dyn=${TSSM_READNCE_DYN_LAGS} w=${TSSM_READNCE_W}/${TSSM_READNCE_DYN_W}"

RESUME_FLAG=""
if [ -d "${OUTPUT}" ]; then
    L=$(find "${OUTPUT}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
    [ -n "$L" ] && { echo "Resuming $L"; RESUME_FLAG="--resume_from_checkpoint $L"; }
fi

torchrun --nnodes=1 --nproc_per_node=${NPROC:-2} --master_port=${MASTER_PORT:-29594} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$BASE_MODEL" --version llama_3 --seed 42 \
    --data_mixture "r2r+rxr+nav_stops_r2r*3+nav_stops_rxr*3" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch --mm_projector mlp_downsample \
    --mm_use_im_patch_token False --num_video_frames 1 \
    --tune_vision_tower False --tune_mm_projector False --tune_language_model False \
    --use_rssm_v4 True --rssm_v4_cache_path "$MIX_CACHE" \
    --rssm_v4_state_dim 3072 --rssm_v4_proj_hidden ${PHID:-2048} --rssm_num_state_tokens ${NTOK:-16} \
    --lora_enable True --lora_llm True --lora_r 64 --lora_alpha 128 \
    --lora_dropout 0.05 --bf16 True --model_max_length 4096 --gradient_checkpointing True \
    --output_dir "$OUTPUT" --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BS:-1} --gradient_accumulation_steps ${ACCUM:-32} \
    --learning_rate 5e-5 --weight_decay 0.01 --warmup_ratio 0.05 \
    --lr_scheduler_type cosine --logging_steps 10 --save_steps 500 \
    --evaluation_strategy steps --eval_steps 500 --per_device_eval_batch_size 1 \
    --save_total_limit 8 --dataloader_num_workers ${WORKERS:-1} --ddp_timeout 3600 \
    --report_to tensorboard ${RESUME_FLAG} ${MAX_STEPS:+--max_steps $MAX_STEPS}
echo "End: $(date)"
