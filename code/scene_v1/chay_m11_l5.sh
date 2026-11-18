#!/bin/bash
# Sửa M11 theo chẩn đoán 18/11: lam 5 (cân gradient CE ~ rào), lr_st 5e-4; 10 epoch ngắn + đo gradient.
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
ten=m11_kp_l5
LOG=$TL/scene_v1/$ten.log
cd $TL/cai_tien
$PY m11_hoc_chung.py --ten $ten --init nosam_kp841 --beta 1 --lam 5 --lr_st 5e-4 --epochs 10 --do_grad 10 > $LOG 2>&1
cd $TL/b3
$PY xuat_anh.py --bien_the $ten >> $LOG 2>&1
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage tb nosam_kp841 m11_kp $ten --tap dev --hau kp_m11l5 >> $LOG 2>&1
$PY lech_y.py nosam_kp841 m11_kp $ten sage >> $LOG 2>&1
grep "^epoch" $LOG | tail -6; tail -n 12 $LOG
