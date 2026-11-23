#!/bin/bash
# du2_mat_ft QUA sàng lọc chep_delta (dev: person chep 0,54, nhieu 6,9; đèn nhieu 6,3; road/vỉa hè 3,3). Lưu ý: đèn chep −0,32 (ngược thầy).
# Bước tiếp (luật đặt trước): xuất ảnh + B0 s0 máy nhà; ĐI TIẾP Kaggle 3 seed nếu du2_mat_ft − nosam_kp841 (56,99) > 0 ở 10 lớp.
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
TEN=du2_mat_ft
cd $TL/b3
[ $(ls $SAGE_ANH/$TEN/train 2>/dev/null | wc -l) -ge 992 ] || $PY xuat_anh.py --bien_the $TEN
run=$SAGE_RUNS/${TEN}_b0_s0
if [ ! -f $run/dev_ket_qua_last.json ]; then
  echo "[$(date +%H:%M)] train B0 $TEN seed 0"
  $PY train_seg.py --bien_the $TEN --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${TEN}_b0_s0.txt 2>&1
  $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${TEN}_b0_s0.txt 2>&1
fi
cp $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv $SAGE_RUNS/_bak.csv 2>/dev/null || true
$PY so_sanh_lop_du.py --bien_the $TEN cc_vhn_thuan nosam_kp841 --model b0 --tap dev --seeds 0 --min_anh 20 1
mv $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv $SAGE_RUNS/so_sanh_lop_du_dev_b0_du2.csv
[ -f $SAGE_RUNS/_bak.csv ] && mv $SAGE_RUNS/_bak.csv $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv
$PY $TL/scene_v1/iou_lop_bang.py nosam=$SAGE_RUNS/nosam_kp841_b0_s0 du2=$run thuan=$SAGE_RUNS/cc_vhn_thuan_b0_s0 vhnKG0=$TL/scene_v1/runs_b0_gop/vhn_b0_s0
echo "[$(date +%H:%M)] XONG"
