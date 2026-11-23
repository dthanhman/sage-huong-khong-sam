#!/bin/bash
# S-MẮT (23/11) – CHẨN ĐOÁN, chạy song song với Kaggle sage-b0-vhn. Không train B0.
# CÂU HỎI: Student có "mắt ngữ nghĩa" (ResNet18 ImageNet đóng băng tới layer3, b3/sage_student/model_mat.py)
#   có LÀM THEO được luật theo lớp mà Student gốc 0,136M không làm được không?
# VÌ SAO: lint (L_int theo lớp, w_lop.json) rớt vì Student không biết vùng nào là lớp nào: hệ số kéo về IR a ở
#   person 0,41 (đích 0,7), vỉa hè 0,38 / biển 0,20 / truck 0,23 / bus 0,28 (đích 0,1). Chưng cất từ hai_nhanh (--ref vhn)
#   sẽ gặp đúng vấn đề này -> kiểm trước bằng chính loss lint, đo trên val151 (Student KHÔNG học ảnh val).
# CHẠY: 2 Student seed 0 trên 841 ảnh train (mỗi cái ~25 phút), rồi đo a theo lớp (do_he_so_ir.py, ~3 phút):
#   lint_nha = kiến trúc gốc + w_lop  (lặp lại lint của Kaggle trên máy nhà, làm mốc cùng máy)
#   lint_mat = kiến trúc mắt + w_lop
# ĐẶT TRƯỚC (val151):
#   GO (gắn mắt vào bước chưng cất): lint_mat có a_person >= 0,6 VÀ TB a(vỉa hè, đèn, biển, truck, bus) <= 0,2
#        VÀ độ lệch D = TB |a − w| trên 6 lớp đó <= ½ D(lint_nha).
#   Không đạt: mắt ResNet18 không đủ -> ghi nhận; chưng cất dùng Student gốc (hoặc thêm L_cs kiểu SAGE).
# Chạy (Git Bash): bash scene_v1/chay_mat.sh > scene_v1/chay_mat.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
ST=$TL/b3/sage_student

cd $TL/e17
for cfg in "lint_nha goc" "lint_mat mat"; do
  set -- $cfg
  if [ ! -f $ST/$1.pt ]; then
    echo "[$(date +%H:%M)] train Student $1 (kiến trúc $2)"
    $PY train_student_nosam.py --chi_train --ten $1 --seed 0 --kien_truc $2 --w_lop $TL/scene_v1/w_lop.json \
      > $TL/scene_v1/log_student_$1.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $1 xong: $(tail -1 $TL/scene_v1/log_student_$1.txt)"
done

echo "[$(date +%H:%M)] đo hệ số kéo về IR theo lớp (val151)"
cd $TL/scene_v1
RA_CSV=$TL/scene_v1/he_so_ir_mat.csv $PY do_he_so_ir.py nosam_kp841 lint lint_nha lint_mat sage
$PY ket_luan_mat.py
echo "[$(date +%H:%M)] XONG"
