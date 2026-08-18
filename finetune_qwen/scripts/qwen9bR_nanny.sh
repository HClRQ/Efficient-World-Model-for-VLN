#!/bin/bash
# 9B-RSSM净血线冒烟保姆 (08-16 用户令): 每2000步一考+最终步, 184卷。
# ⚠在线塔=净血套装: TSSM_ONLINE_CKPT=rssmpub blob + TSSM_ONLINE_SIGLIP=Google公版(与state缓存同源家法)
cd /cluster/project/rsl/ruiqzhang/codes/qwen35-vln
LOG=logs/qwen9bR_nanny.log
SCENE_GROUPS=("zsNo4HB9uLZ,Z6MFQCViBuw" "TbHJrupSAjP,oLBMNvg9in8,pLe4wQe7qrG" \
              "QUCTc6BB5sX,X7HyMhZNoso,8194nk5LbLH" "2azQ1b91cZZ,EU6Fwq7SyZv,x8F5xyUWy9e")
while true; do
  for h in $(ls -d /cluster/scratch/ruiqzhang/qwen9b_rssmpub/hf_step* 2>/dev/null | sort -V); do
    n=$(basename $h | sed 's/hf_step//')
    [ -f "$h/.smoked" ] && continue
    if [ $((n % 2000)) -ne 0 ] && [ "$n" -lt 21404 ]; then
      touch "$h/.smoked"; continue
    fi
    [ -f "$h/pytorch_model.bin" ] || continue
    [ -f "$h/projector.pt" ] || continue
    sleep 60
    SNAP=$(ls -d /cluster/scratch/ruiqzhang/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
    for f in config.json preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
      [ -f "$h/$f" ] || cp "$SNAP/$f" "$h/" 2>>$LOG
    done
    i=0
    for S in "${SCENE_GROUPS[@]}"; do
      i=$((i+1))
      SCENES="$S" SMOKE_FRAC=0.1 EPISODE_COUNT=-1 TAG=q9bR_${n}_g$i CKPT_STEP=$n \
        QWEN_OUT_DIR=/cluster/scratch/ruiqzhang/qwen9b_rssmpub \
        TSSM_ONLINE_CKPT=/cluster/scratch/ruiqzhang/tssm_rssmpub.pt \
        TSSM_ONLINE_SIGLIP=/cluster/scratch/ruiqzhang/hf_cache/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/ \
        sbatch --export=ALL --time=04:00:00 scripts/smoke_stage2.sbatch >>$LOG 2>&1
      sleep 8
    done
    touch "$h/.smoked"
    echo "$(date '+%m-%d %H:%M') 已发4组184卷 ck$n (9B-RSSM净血)" >>$LOG
  done
  sleep 300
done
