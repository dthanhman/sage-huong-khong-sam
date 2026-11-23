#!/bin/bash
# Vòng 2 (23/11): chay_du cho thấy khởi tạo từ nosam đã DẸP nhiễu (person 33 -> 3-6) nhưng person chep vẫn thấp
# (dev 0,09-0,31; TRAIN 0,14-0,38) -> học chưa tới ngay cả trên train. Thử: vùng thầy chỉnh nặng hơn (w_khac 100) + học lâu/mạnh hơn.
# Nếu TRAIN person chep vẫn < 0,5 -> giới hạn là sức mạng/cách gắn mắt, không phải loss. Luật như chay_du.sh.
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
CHUNG="--chi_train --ref vhn --chi_ref --w_khac 100 --khoi_tao nosam_kp841 --epochs 60 --w_grad_chi_ref 1 --kien_truc mat --lr_du"
cd $TL/e17
tr() { ten=$1; shift; [ -f $TL/b3/sage_student/$ten.pt ] || { echo "[$(date +%H:%M)] train $ten"; $PY train_student_nosam.py $CHUNG --ten $ten "$@" > $TL/scene_v1/$ten.log 2>&1; tail -2 $TL/scene_v1/$ten.log; }; }
tr du2_mat_bang --lr 3e-3 --dong_bang_goc
tr du2_mat_ft   --lr 5e-4
cd $TL/scene_v1
$PY chep_delta.py dev vhn nosam_kp841 ck:du2_mat_bang ck:du2_mat_ft
$PY chep_delta.py train vhn nosam_kp841 ck:du2_mat_bang ck:du2_mat_ft
echo "[$(date +%H:%M)] XONG"
