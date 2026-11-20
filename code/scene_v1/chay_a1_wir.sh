#!/bin/bash
# A1 (20/11): quét tỉ lệ IR trong loss cường độ của nosam (gốc w_ir 0,4 = 3*Y + 2*IR).
# Giả thuyết: nosam dùng IR quá ít -> thua ở person / ảnh tối. Bằng chứng cũ (13/11, trộn tuyến tính, B0):
#   vis 51,78 / ir30 53,65 / tb 54,26 / ir70 54,81 -> nhiều IR hơn thì mIoU chung tăng, nhưng ảnh sáng giảm.
# Bước: train nosam w 0,5/0,6/0,7 (841 ảnh, seed 0, ~11 phút/cái) -> xuất ảnh -> chỉ số ghép val 151
#       -> B0 seed 0 từng cái (~1h20/cái, MÁY NHÀ) -> so cặp với nosam_kp841 s0 (máy nhà, từ B1) -> mIoU theo độ sáng.
# ĐẶT TRƯỚC (sửa 20/11, sau khi thấy lợi thế cũ của nosam đến từ motorcycle/bicycle 1–2 ảnh):
#   Số CHÍNH = mIoU trên 10 lớp có >= 20 ảnh trong val 151 (so_sanh_lop_du.py, khối "Lớp có >= 20 ảnh").
#   Chọn w có mIoU 10 lớp s0 cao nhất. ĐẠT nếu hơn nosam (w 0,4, cùng máy) >= +1,0 trên 10 lớp
#   VÀ IoU person tăng (mục tiêu: kéo person lên ngang sage) -> chạy seed 1 để xác nhận.
#   Mốc tham chiếu (B0 2 seed, Kaggle): 10 lớp nosam 57,32 / sage 57,15 / tb 56,91; person 44,7 / 47,2 / 46,7.
#   mIoU đủ 14 lớp chỉ là số phụ. Chỉ số ghép: báo kèm (rào MS-SSIM >= 99%, Q_ABF >= 98% SAGE), không dùng chọn w.
# Chạy lại được: bước nào có kết quả thì bỏ qua; B0 dở thì train tiếp (--tiep).
# Chạy (Git Bash, SAU KHI B1 XONG): bash scene_v1/chay_a1_wir.sh > scene_v1/chay_a1_wir.out 2>&1   (~4,5 h)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
LOG=$TL/scene_v1
W="05 06 07"

for w in $W; do
  ten=nosam_kp841_w$w
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train $ten (w_ir 0.${w#0})"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --ten $ten --seed 0 --w_ir 0.${w#0} > $LOG/$ten.log 2>&1
  fi
  if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten | wc -l) -lt 1500 ]; then
    echo "[$(date +%H:%M)] xuất ảnh $ten"
    cd $TL/b3
    $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
  fi
done

echo "[$(date +%H:%M)] chỉ số ghép val 151"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage nosam_kp841 nosam_kp841_w05 nosam_kp841_w06 nosam_kp841_w07 \
  --tap dev --hau kp_wir 2>&1 | tail -n 8

cd $TL/b3
for w in $W; do
  bt=nosam_kp841_w$w; run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep > $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1)"
done

echo "[$(date +%H:%M)] so cặp (cùng máy nhà, seed 0)"
$PY so_sanh_cap.py --bien_the nosam_kp841 nosam_kp841_w05 nosam_kp841_w06 nosam_kp841_w07 \
  --model b0 --tap dev --seeds 0 2>&1 | sed -n 3,40p
echo "[$(date +%H:%M)] SỐ CHÍNH: mIoU lớp đủ ảnh (>= 20 ảnh)"
$PY so_sanh_lop_du.py --bien_the nosam_kp841 nosam_kp841_w05 nosam_kp841_w06 nosam_kp841_w07 \
  --model b0 --tap dev --seeds 0 --min_anh 20 \
  --cap nosam_kp841_w05-nosam_kp841 nosam_kp841_w06-nosam_kp841 nosam_kp841_w07-nosam_kp841 2>&1
echo "[$(date +%H:%M)] mIoU theo độ sáng (phụ; nhóm sáng/tối ~ đoạn cảnh trên val 151)"
$PY ../scene_v1/miou_theo_sang.py --bien_the nosam_kp841 nosam_kp841_w05 nosam_kp841_w06 nosam_kp841_w07 \
  --model b0 --seeds 0 --goc nosam_kp841 2>&1 | tail -n 25
echo XONG
