#!/bin/bash
# A4 (21/11): A2 λ10 + loss giữ độ TÁCH 2 vùng kề nhau (mọi cặp lớp trong nhãn 841 Train, không SAM, chỉ lúc train).
# Bằng chứng (scene_v1/kiem_a3/do_tuong_phan_lop.py, ảnh TRAIN): tách đường–vỉa hè VIS 0,45 / IR 1,06 / SAGE 0,83 /
#   TB 0,69 / A2 λ10 0,66 / nosam 0,49 -> ảnh ghép không SAM làm mất thông tin VÙNG của IR (cạnh biên thì vẫn giữ ~90%).
# λ_tach = 1 (ĐẶT 1 LẦN: --thu 60 cho lam*tach ~0,48 / tổng ~5,6 ≈ 8%, cỡ phần loss vùng của A2). Không dò thêm.
# ĐẶT TRƯỚC (A4 s0 so với A2 λ10 s0, cả hai chấm B0 trên Kaggle):
#   (1) tách đường–vỉa hè trên ảnh ghép tăng so với A2 (0,66), tiến về SAGE 0,83;
#   (2) IoU sidewalk tăng; (3) 10 lớp >= A2 + 0,5; (4) qua rào ảnh (MS-SSIM >= 99%, Q_ABF >= 98% SAGE).
#   Đạt cả 4 -> seed 1, 2.
# Chạy (Git Bash): bash scene_v1/chay_a4.sh > scene_v1/chay_a4.out 2>&1   (~20 phút)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
LOG=$TL/scene_v1
ten=nosam_kp841_vung10_tach1
if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
  echo "[$(date +%H:%M)] train $ten (lam_vung 10, lam_tach 1, seed 0)"
  cd $TL/e17
  $PY train_student_nosam.py --chi_train --ten $ten --seed 0 --lam_vung 10 --lam_tach 1 > $LOG/$ten.log 2>&1
fi
if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten/train | wc -l) -lt 992 ]; then
  echo "[$(date +%H:%M)] xuất ảnh $ten"
  cd $TL/b3
  $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
fi
echo "[$(date +%H:%M)] chỉ số ghép val 151 (rào chắn)"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage nosam_kp841 nosam_kp841_vung10 $ten --tap dev --hau a4 2>&1 | tail -n 5
echo XONG
