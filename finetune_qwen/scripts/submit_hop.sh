#!/bin/bash
# 9B全量统一发射器 (08-09): submit_hop.sh {a100|long|pro8} [额外sbatch参数...]
#   三个池子同配方(全局批64/同ds存档), 靠 train_full9b_*.sh 里的互斥锁保证只有一棒在跑。
#   让位的票会调用本脚本补投替补 → 名额不烧掉(集群 JobRequeue=0, 不能自动重排)。
S=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/scripts
L=/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/logs
F=${1:-a100}; shift 2>/dev/null
COMMON="--nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=24G --requeue"  # 08-10: 192G被bs4的17h慢涨OOM杀过, 节点1.5T随便给
case "$F" in
  a100) sbatch $COMMON --job-name=q9b_a100 --partition=gpupr.24h --time=24:00:00 \
          --exclude=eu-a65-02,eu-a65-03,eu-a65-04 \
          --gpus=nvidia_a100_80gb_pcie:8 --output=$L/qfull9b_a100_%j.out \
          --wrap="HOP_FLAVOR=a100 bash $S/train_full9b_a100.sh" "$@" ;;
  long) sbatch $COMMON --job-name=q9b_long --partition=gpupr.120h --time=72:00:00 \
          --gpus=nvidia_a100_80gb_pcie:8 --output=$L/qfull9b_a100_%j.out \
          --wrap="HOP_FLAVOR=long bash $S/train_full9b_a100.sh" "$@" ;;
  rssm9b) sbatch --nodes=1 --ntasks=1 --cpus-per-task=40 --mem-per-cpu=16G --requeue \
          --job-name=q9bR_a100 --partition=gpupr.24h --time=24:00:00 \
          --exclude=eu-a65-02,eu-a65-03,eu-a65-04 \
          --gpus=nvidia_a100_80gb_pcie:8 --output=$L/qfull9b_rssm_%j.out \
          --wrap="HOP_FLAVOR=rssm9b bash $S/train_full9b_rssm.sh" "$@" ;;
  rssm9b_pro) sbatch --nodes=1 --ntasks=1 --cpus-per-task=40 --mem-per-cpu=16G --requeue \
          --job-name=q9bR_pro --partition=cuda13pr.120h --time=110:00:00 \
          --gpus=nvidia_rtx_pro_6000:8 --constraint=EPYC_9654 --output=$L/qfull9b_rssm_%j.out \
          --wrap="HOP_FLAVOR=rssm9b_pro bash $S/train_full9b_rssm.sh" "$@" ;;
  nomem2) sbatch --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=12G --requeue \
          --job-name=q9bNM_2pro --partition=cuda13pr.120h --time=110:00:00 \
          --gpus=nvidia_rtx_pro_6000:2 --constraint=EPYC_9654 --output=$L/qfull9b_nomem_%j.out \
          --wrap="NPROC=2 ACCUM=8 HOP_FLAVOR=nomem2 bash $S/train_full9b_nomem.sh" "$@" ;;
  nomem4a) sbatch --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=12G --requeue \
          --job-name=q9bNM_4a --partition=gpupr.120h --time=110:00:00 \
          --exclude=eu-a65-02,eu-a65-03,eu-a65-04 \
          --gpus=nvidia_a100_80gb_pcie:4 --output=$L/qfull9b_nomem_%j.out \
          --wrap="NPROC=4 ACCUM=4 HOP_FLAVOR=nomem4a bash $S/train_full9b_nomem.sh" "$@" ;;
  tssm9b_pro) sbatch --nodes=1 --ntasks=1 --cpus-per-task=40 --mem-per-cpu=16G --requeue \
          --job-name=q9bT_pro --partition=cuda13pr.120h --time=110:00:00 \
          --gpus=nvidia_rtx_pro_6000:8 --constraint=EPYC_9654 --output=$L/qfull9b_tssm_%j.out \
          --wrap="HOP_FLAVOR=tssm9b_pro bash $S/train_full9b_tssm.sh" "$@" ;;
  tssm9b_pro4) sbatch --nodes=1 --ntasks=1 --cpus-per-task=20 --mem-per-cpu=12G --requeue \
          --job-name=q9bT_pro4 --partition=cuda13pr.120h --time=110:00:00 \
          --gpus=nvidia_rtx_pro_6000:4 --constraint=EPYC_9654 --output=$L/qfull9b_tssm_%j.out \
          --wrap="NPROC=4 ACCUM=4 HOP_FLAVOR=tssm9b_pro4 bash $S/train_full9b_tssm.sh" "$@" ;;
  pro8) sbatch $COMMON --job-name=q9b_pro8 --partition=cuda13pr.24h --time=24:00:00 \
          --gpus=nvidia_rtx_pro_6000:8 --constraint=EPYC_9654 --output=$L/qfull9b_pro_%j.out \
          --wrap="HOP_FLAVOR=pro8 bash $S/train_full9b_pro8.sh" "$@" ;;
  *) echo "用法: submit_hop.sh {a100|long|pro8}"; exit 1 ;;
esac
