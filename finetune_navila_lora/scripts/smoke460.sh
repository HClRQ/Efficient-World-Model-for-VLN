#!/bin/bash
# 460集裁决级冒烟(每场景随机抽25%, 固定种子, sigma±2.3), 按场景拆4组并行4090。
# 抽样与全量场景构成成比例, 每场景独立种子 → 4组并集 == 单作业460集, 逐臂可比。
# 用法:
#   STOK_ENV="TSSM_PSEUDO_FRAMES=0 TSSM_HISTORY_STATES=1 TSSM_PROJ_SLICE=1 TSSM_ONLINE_CKPT=..." \
#     bash scripts/eval/smoke460.sh <CKPT目录> <TAG>
# 聚合: bash scripts/eval/smoke460.sh --agg <TAG>
set -eo pipefail
ROOT=/cluster/project/rsl/ruiqzhang/codes/NaVILA
cd $ROOT
# eval jobs run from a different cwd: a relative CKPT passes the [ -d ] check
# here but dies with FileNotFoundError inside the job (07-18 incident: five
# 460 decks silently dead). Absolutize before anything else.
[ "$1" != "--agg" ] && [ -n "$1" ] && set -- "$(readlink -f "$1")" "${@:2}"

# 组间按比例集数配平: 115/114/109/122
G1="['zsNo4HB9uLZ','Z6MFQCViBuw']"
G2="['TbHJrupSAjP','oLBMNvg9in8','pLe4wQe7qrG']"
G3="['QUCTc6BB5sX','X7HyMhZNoso','8194nk5LbLH']"
G4="['2azQ1b91cZZ','EU6Fwq7SyZv','x8F5xyUWy9e']"

if [ "$1" = "--agg" ]; then
  TAG=$2
  /cluster/project/rsl/ruiqzhang/miniconda3/envs/navila-eval/bin/python - "$TAG" <<'PY'
import json, glob, sys
import numpy as np
tag = sys.argv[1]
eps = {}
files = glob.glob(f"evaluation/eval_out/tssm_3072_smoke/{tag}_g*/r2r/**/val_unseen_*.json", recursive=True)
for f in files:
    eps.update(json.load(open(f)))
_frac_implied = len(eps) / 460 * 0.25 if eps else 0
print(f"{tag}: {len(files)}组文件, 共{len(eps)}集 (FRAC=0.25基准460集; 折算本次FRAC≈{_frac_implied:.2f})")
if eps:
    for k in ["success","oracle_success","spl","ndtw","steps_taken"]:
        print(f"  {k}: {np.mean([e[k] for e in eps.values()]):.4f}")
PY
  exit 0
fi

CKPT=$1; TAG=$2
[ -d "$CKPT" ] || { echo "CKPT不存在: $CKPT"; exit 1; }
i=0
for SCENES in "$G1" "$G2" "$G3" "$G4"; do
  i=$((i+1))
  env $STOK_ENV SMOKE_FRAC=${FRAC:-0.25} EPISODE_COUNT=-1 \
    CKPT=$CKPT TAG=${TAG}_g${i} \
    EXTRA_OPTS="TASK_CONFIG.DATASET.CONTENT_SCENES $SCENES" \
    sbatch --time=${EVAL_TIME:-04:00:00} ${EVAL_SBATCH_OPTS:-} evaluation/scripts/eval/tssm_3072_r2r_SMOKE_4090.sbatch
done
echo "4组已发 TAG=${TAG}_g1..g4 (共460集); 聚合: bash scripts/eval/smoke460.sh --agg $TAG"
