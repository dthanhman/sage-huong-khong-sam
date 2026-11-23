#!/bin/bash
# CHƯNG CẤT PHẦN DƯ (23/11). Chẩn đoán vì sao cc_vhn/cc_vhn_thuan rớt (scene_v1/chep_delta.py, dev val151):
#  - thầy vhn chỉnh person RMS 15 mức xám, Student chỉ chép 9% (train 5%);
#  - train lại CHÍNH nosam (nha2) đã lệch person RMS 33 > 15 -> tín hiệu thầy chìm trong nhiễu train lại;
#  - grad chiếm ~95% loss, thầy chỉnh mượt nên grad không thấy;
#  - Student vùng nhìn ~46 px, không có ngữ nghĩa.
# Sửa: khởi tạo từ nosam_kp841 (nền của thầy) + hạ grad + (tùy) nhánh mắt ResNet18 đóng băng.
# LUẬT SÀNG LỌC (đặt trước, dev val151, chep_delta): person chep >= 0,5 VÀ person nhieu <= 10 VÀ đèn nhieu <= 10
#   VÀ TB nhieu(road, sidewalk) <= 5. Đạt -> xuất ảnh + B0 s0 máy nhà (so nosam 56,99, 10 lớp) -> Kaggle 3 seed.
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
CHUNG="--chi_train --ref vhn --chi_ref --w_khac 10 --khoi_tao nosam_kp841 --epochs 30"
cd $TL/e17
tr() { ten=$1; shift; [ -f $TL/b3/sage_student/$ten.pt ] || { echo "[$(date +%H:%M)] train $ten"; $PY train_student_nosam.py $CHUNG --ten $ten "$@" > $TL/scene_v1/$ten.log 2>&1; tail -2 $TL/scene_v1/$ten.log; }; }
tr du_ft          --lr 2e-4 --w_grad_chi_ref 1
tr du_mat_bang_g1 --lr 1e-3 --w_grad_chi_ref 1 --kien_truc mat --dong_bang_goc
tr du_mat_bang_g9 --lr 1e-3 --w_grad_chi_ref 9 --kien_truc mat --dong_bang_goc
tr du_mat_ft      --lr 2e-4 --w_grad_chi_ref 1 --kien_truc mat
cd $TL/scene_v1
echo "[$(date +%H:%M)] chep_delta dev"
$PY chep_delta.py dev vhn nosam_kp841 ck:du_ft ck:du_mat_bang_g1 ck:du_mat_bang_g9 ck:du_mat_ft
echo "[$(date +%H:%M)] chep_delta train"
$PY chep_delta.py train vhn nosam_kp841 ck:du_ft ck:du_mat_bang_g1 ck:du_mat_bang_g9 ck:du_mat_ft
echo "[$(date +%H:%M)] XONG"
