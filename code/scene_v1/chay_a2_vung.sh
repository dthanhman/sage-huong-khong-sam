#!/bin/bash
# A2 (20/11): nosam + loss tương phản VẬT–VÀNH theo vùng, mask lấy từ NHÃN 841 ảnh train (không SAM, chỉ lúc train).
# Bằng chứng (phan_tich_nosam.py): SAGE làm nổi người CÓ CHỌN LỌC (tương phản người/vành 44,5; tỉ lệ so std cả ảnh, trung vị từng ảnh, 1,32),
#   nosam 23,5 / 0,97; A1 (tăng w_ir) chỉ tăng tương phản đều cả ảnh -> rớt. nosam thua nặng nhất ở người NHỎ (−8,4 IoU).
# Loss: lam * TB_vật relu(max(c(Y), c(IR)) − c(ghép)), c = |TB vật − TB vành 7 px|; vật = person, car, truck, bus, moto, bicycle.
# Bước: train Student lam 10 và 30 (841 ảnh, seed 0, ~14 phút/cái) -> xuất ảnh -> tương phản + chỉ số ghép val 151
#       -> B0 seed 0 từng cái (~1h15/cái, MÁY NHÀ) -> so với nosam_kp841 s0 (máy nhà).
# ĐẶT TRƯỚC: số CHÍNH = mIoU 10 lớp >= 20 ảnh (so_sanh_lop_du.py). Chọn lam có 10 lớp cao nhất.
#   ĐẠT nếu >= nosam_kp841 (56,99) + 1,0 VÀ person IoU tăng -> seed 1 để xác nhận.
#   Chẩn đoán (không dùng để chọn): tỉ lệ tương phản người (trung vị từng ảnh) tiến về SAGE 1,32 (nosam 0,97, tb 1,05); rào MS-SSIM >= 99%, Q_ABF >= 98% SAGE.
# Chạy lại được: bước nào có kết quả thì bỏ qua; B0 dở thì train tiếp (--tiep).
# LAM nhận đuôi "mr" (mở rộng lớp vật, 20/11), vd LAM="30 10mr".
# Chạy (Git Bash): bash scene_v1/chay_a2_vung.sh > scene_v1/chay_a2_vung.out 2>&1   (~3 h)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
LOG=$TL/scene_v1
LAM=${LAM:-"10 30"}
BT=""

for l in $LAM; do
  # đuôi "mr" = MỞ RỘNG lớp vật: thêm đèn (4), biển báo (5), cột (14) vào người + xe (8–13)
  ten=nosam_kp841_vung$l; BT="$BT $ten"; lam=${l%mr}; them=""
  [ "$l" != "$lam" ] && them="--lop_vat 4 5 8 9 10 11 12 13 14"
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train $ten (lam_vung $lam $them)"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --ten $ten --seed 0 --lam_vung $lam $them > $LOG/$ten.log 2>&1
  fi
  if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten/train | wc -l) -lt 992 ]; then
    echo "[$(date +%H:%M)] xuất ảnh $ten"
    cd $TL/b3
    $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
  fi
done

echo "[$(date +%H:%M)] tương phản người (val 151)"
cd $TL/b3
$PY ../scene_v1/do_tuong_phan.py --bien_the sage tb nosam_kp841 $BT
echo "[$(date +%H:%M)] chỉ số ghép val 151"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage nosam_kp841 $BT --tap dev --hau kp_vung 2>&1 | tail -n 6

cd $TL/b3
for bt in $BT; do
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1); $(grep person $run/dev_iou_lop_last.csv)"
done

echo "[$(date +%H:%M)] SỐ CHÍNH: mIoU lớp đủ ảnh (>= 20 ảnh), so với nosam_kp841 (máy nhà, s0)"
CAP=""; for bt in $BT; do CAP="$CAP $bt-nosam_kp841"; done
$PY so_sanh_lop_du.py --bien_the nosam_kp841 $BT --model b0 --tap dev --seeds 0 --min_anh 20 --cap $CAP 2>&1
echo "nosam_kp841 person: $(grep person $SAGE_RUNS/nosam_kp841_b0_s0/dev_iou_lop_last.csv)"
echo XONG
