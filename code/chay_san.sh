#!/bin/bash
# Chốt SÀN (độ dao động do seed) trên tập khám phá scene_v1: nosam seed 1–4 (seed 0 = nosam_kp841 đã có).
# Cùng công thức 17/11 trên split cũ: ngưỡng = SAGE − 2 × SD(5 seed nosam), làm tròn 1%.
# Bước: train nosam_kp841_s{1..4} -> xuất ảnh -> chỉ số ghép val 151 -> CI + SD seed.
# Chạy lại được: checkpoint / ảnh đã có thì bỏ qua.
# Chạy: bash scene_v1/chay_san.sh   (từ thư mục tai_lap; phiên phải mở suốt, ~1 h)
set -e
export SAGE_MAN=D:/CS406/DoAn/Sage/tai_lap/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=D:/CS406/DoAn/Sage/luu_tru/tim_gap_b0_0811/seg/anh
PY=D:/Miniconda3/envs/DoAn/python.exe
TL=D:/CS406/DoAn/Sage/tai_lap
LOG=$TL/scene_v1

for s in 1 2 3 4; do
  ten=nosam_kp841_s$s
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train $ten"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --ten $ten --seed $s > $LOG/$ten.log 2>&1
  fi
  echo "[$(date +%H:%M)] xuất ảnh $ten"
  cd $TL/b3
  $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
done

echo "[$(date +%H:%M)] chỉ số ghép val 151"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage nosam_kp841 nosam_kp841_s1 nosam_kp841_s2 nosam_kp841_s3 nosam_kp841_s4 \
  --tap dev --hau kp_san > $LOG/san_chi_so.log 2>&1
$PY cong2_ci.py chi_so_ghep/dev_kp_san_tung_anh.csv > $LOG/san_ci.txt 2>&1
cat $LOG/san_ci.txt
echo XONG
