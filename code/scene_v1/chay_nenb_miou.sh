#!/bin/bash
# Bước sau chay_nenb_ce.sh: nếu ảnh mới ĐẠT (|ảnh - tbsua_w150| >= 2 mức xám, MS-SSIM >= 99 và Q_ABF >= 98 % SAGE)
# thì train B0 MỚI từ đầu trên ảnh đó (841 ảnh, 100 ep, batch 8, 14 lớp – như Kaggle sage-b0-kham-pha), chấm val 151 last.pt.
# 1 seed = sàng lọc; so với tb/sage/tbsua_w150 seed 0 của Kaggle.
# Chạy: bash scene_v1/chay_nenb_miou.sh   (từ tai_lap; ~1,2 h)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
ten=${TEN:-m11b_l5_b01}
s=${SEED:-0}

goc=${GOC:-tbsua_w150}   # ảnh nền để đo lệch (nền A: nosam_kp841)
lech=$(grep "^|$goc - $ten|" $TL/scene_v1/$ten.log | tail -1 | sed 's/.*= \([0-9.]*\).*/\1/')
lech=${lech:-0}
read ms qabf < <($PY -c "
import pandas as pd
d = pd.read_csv('$TL/cai_tien/chi_so_ghep/dev_kp_${ten}_tong_hop.csv').set_index('bien_the').loc['$ten']
print(d['MS_SSIM_%sage'], d['Q_ABF_%sage'])")
echo "[$(date +%H:%M)] $ten: lệch so $goc = $lech mức xám; MS-SSIM $ms %; Q_ABF $qabf % SAGE"
dat=$($PY -c "print(int($lech >= 2 and $ms >= 99 and $qabf >= 98))")
if [ "$dat" != "1" ] && [ -z "$BO_LOC" ]; then   # BO_LOC=1: bỏ lọc (19/11: lọc theo TB cả ảnh che mất sửa cục bộ ở người)
  echo "KHÔNG ĐẠT điều kiện lọc -> không train B0."; echo XONG; exit 0
fi

mkdir -p $SAGE_RUNS
run=$SAGE_RUNS/${ten}_b0_s$s
cd $TL/b3
echo "[$(date +%H:%M)] ĐẠT -> train B0 $ten seed $s"
$PY train_seg.py --bien_the $ten --model b0 --seed $s --tiep > $SAGE_RUNS/log_${ten}_b0_s$s.txt 2>&1
echo "[$(date +%H:%M)] chấm val 151"
$PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${ten}_b0_s$s.txt 2>&1
tail -n 15 $SAGE_RUNS/log_${ten}_b0_s$s.txt
echo XONG
