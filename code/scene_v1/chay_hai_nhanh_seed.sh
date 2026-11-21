#!/bin/bash
# 21/11: cận trên hai_nhanh seed 1, 2 (s0 = 59,27 / 10 lớp, +2,30 vs max(sage, tb) -> luật chay_tran_v2.sh: xác nhận bằng seed 1, 2).
# Máy nhà, batch 8 (s0 chạy batch 8 không OOM). Chờ chay_a4.sh in XONG (dùng chung GPU) rồi mới chạy.
# Chạy: bash scene_v1/chay_hai_nhanh_seed.sh > scene_v1/chay_hai_nhanh_seed.out 2>&1   (~3 h)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
GOP=$TL/scene_v1/runs_b0_gop
until tail -n 1 $TL/scene_v1/chay_a4.out | grep -q '^XONG'; do sleep 30; done
cd $TL/b3
for s in ${SEEDS:-1 2}; do
  run=$SAGE_RUNS/hai_nhanh_b0_s$s
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 hai_nhanh seed $s"
    $PY train_seg.py --bien_the hai_nhanh --model b0 --seed $s --tiep >> $SAGE_RUNS/log_hai_nhanh_b0_s$s.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_hai_nhanh_b0_s$s.txt 2>&1
  fi
  echo "[$(date +%H:%M)] hai_nhanh s$s: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1)"
  [ -d $GOP/hai_nhanh_b0_s$s ] || cp -r $run $GOP/
done
echo XONG
