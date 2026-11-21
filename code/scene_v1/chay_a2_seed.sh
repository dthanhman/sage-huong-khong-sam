#!/bin/bash
# 21/11 – A2 λ10 (chốt theo luật chọn λ trong chay_a2_vung.sh): train Student seed 1, 2 + xuất ảnh.
# Checkpoint ra tai_lap/b3/sage_student/nosam_kp841_vung10_s{1,2}.pt -> đẩy lên Kaggle chấm B0 (cùng máy với mốc sage/tb).
# Chạy (Git Bash): bash scene_v1/chay_a2_seed.sh > scene_v1/chay_a2_seed.out 2>&1   (~35 phút)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
LOG=$TL/scene_v1
SEEDS=${SEEDS:-"1 2"}

for s in $SEEDS; do
  ten=nosam_kp841_vung10_s$s
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train $ten (lam_vung 10, seed $s)"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --ten $ten --seed $s --lam_vung 10 > $LOG/$ten.log 2>&1
  fi
  if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten/train | wc -l) -lt 992 ]; then
    echo "[$(date +%H:%M)] xuất ảnh $ten"
    cd $TL/b3
    $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
  fi
done

echo "[$(date +%H:%M)] chỉ số ghép val 151 (rào chắn)"
BT=""; for s in $SEEDS; do BT="$BT nosam_kp841_vung10_s$s"; done
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage nosam_kp841_vung10 $BT --tap dev --hau a2_seed 2>&1 | tail -n 5
echo XONG
