#!/bin/bash
#SBATCH --job-name=q4bsmoke
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --constraint=EPYC_9554
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/logs/smoke_%j.out
#SBATCH --error=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/logs/smoke_%j.out
# Stage2 全量 ckpt 冒烟: 同卡双进程 = qwen_server(vllm env) + habitat评测桥(navila-eval env)。
# 用法: CKPT_STEP=250 SCENES=... PER_SCENE=3 sbatch scripts/smoke_stage2.sbatch
set -eo pipefail
unset CONSUL_HTTP_ADDR
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '^/var/run' | paste -sd ':')
Q=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln
NAVILA=/cluster/project/rsl/ruiqzhang/codes/NaVILA
CKPT_STEP=${CKPT_STEP:?need CKPT_STEP}
H=${QWEN_OUT_DIR:-/cluster/scratch/ruiqzhang/qwen4b_full4}/hf_step${CKPT_STEP}   # full5起可用QWEN_OUT_DIR切目录
[ -f "$H/pytorch_model.bin" ] || { echo "ERROR: $H 不存在"; exit 1; }
PORT=$((30000 + SLURM_JOB_ID % 5000))
export HF_HOME=/cluster/scratch/ruiqzhang/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# --- 进程1: qwen server (后台) ---
VLLM=/cluster/project/rsl/ruiqzhang/miniconda3/envs/vllm
cd $Q
MODEL=$H PROJ=$H/projector.pt PORT=$PORT $VLLM/bin/python qwen_server.py > logs/server_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!
for i in $(seq 1 300); do
  sleep 5
  curl -sf --noproxy "*" http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "server 早死:"; tail -20 logs/server_${SLURM_JOB_ID}.log; exit 1; }
done
curl -sf --noproxy "*" http://127.0.0.1:$PORT/health >/dev/null || { echo "server 25分钟未就绪"; tail -20 logs/server_${SLURM_JOB_ID}.log; exit 1; }
echo "[launcher] server ready on :$PORT"

# --- 进程2: habitat 评测桥 ---
EVALENV=/cluster/project/rsl/ruiqzhang/miniconda3/envs/navila-eval
HABITAT_LAB=/cluster/project/rsl/ruiqzhang/codes/habitat/habitat-lab-old/habitat-lab
export PYTHONPATH=$HABITAT_LAB:$NAVILA:$NAVILA/RSSM_v4:$NAVILA/evaluation
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TSSM_ONLINE_CKPT=${TSSM_ONLINE_CKPT:-/cluster/scratch/ruiqzhang/tssm_ncev5_ckpt/checkpoint-final.pt}
export TSSM_ONLINE_SIGLIP=${TSSM_ONLINE_SIGLIP:-/cluster/project/rsl/ruiqzhang/codes/NaVILA/navila_8f/vision_tower}
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export SERVER=http://127.0.0.1:$PORT
export TAG=${TAG:-q4b_ck${CKPT_STEP}}
export SCENES=${SCENES:-}
export PER_SCENE=${PER_SCENE:-0}
$EVALENV/bin/python $Q/eval_qwen_vln.py
kill $SRV 2>/dev/null || true
echo "End: $(date)"
