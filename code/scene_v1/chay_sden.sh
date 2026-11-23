#!/bin/bash
# S-đèn (21/11): S-xa + A2 λ10 + GIỮ ĐÈN THEO VIS (e17/train_student_nosam.py --lam_den, loss_den).
# NGUYÊN NHÂN cần sửa (phân tích sxa10mr 21/11 ~22h, ma trận nhầm lẫn val151 3 seed):
#   A2 làm IoU biển báo giảm KHÔNG phải do hỏng biển trên ảnh, mà do B0 BÁO NHẦM đèn->biển (2,0k -> 5,1k px/seed).
#   A2 tô đậm cả đèn (đèn không có trong loss): tương phản đèn–vành (val151, 13 ảnh, trung vị mức xám)
#   SAGE 7,3 / nosam 9,0 / A2 14,9 / sxa10 14,2 / sxa10mr 12,9  -> đèn trông giống biển.
#   sxa10mr (thêm đèn/biển/cột vào loss A2) là cách sửa SAI: cột chiếm 53% vật, pha loãng loss người.
# SỬA: phạt relu(c_ghép(đèn) − c_VIS(đèn)) – đèn không được đậm hơn VIS; người/xe vẫn theo loss A2.
# BIẾN THỂ (seed 0, máy nhà): sden10 = λ_den 10, sden30 = λ_den 30. Đối chứng cùng máy: sxa10 (B0 máy nhà).
# ĐẶT TRƯỚC (val151 B0 s0, 10 lớp >= 20 ảnh, so với sxa10 CÙNG MÁY):
#   (1) tương phản đèn giảm về gần nosam (<= ~10) mà tương phản người giữ (>= 90% sxa10);
#   (2) nhầm đèn->biển giảm VÀ IoU biển >= sxa10 + 1,0; (3) 10 lớp >= sxa10; (4) qua rào (MS-SSIM >= 99%, Q_ABF >= 98% SAGE).
#   Đạt -> seed 1, 2 rồi so 3 seed với SAGE/TB (đích: hơn nhất quán cả hai).
#   Không đạt (1) -> loss chưa đủ mạnh/ít đèn trong crop (đếm đèn/batch); đạt (1) mà không đạt (2) -> nhầm đèn->biển
#   không do tương phản -> xem lại nguyên nhân, không bỏ hướng.
# Thứ tự: 2 Student (~20 ph/cái) -> xuất ảnh -> tương phản người + đèn -> rào -> B0 s0 sden10, sden30, sxa10 (~1,4 h/cái).
# Tổng ~5 h. Chạy lại được (bỏ qua bước đã xong, B0 có --tiep). Mặc định CHỈ phần ghép (~45 ph); B0=1 mới chấm B0.
# Chạy (Git Bash): bash scene_v1/chay_sden.sh > scene_v1/chay_sden.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
LOG=$TL/scene_v1

hoc() {  # hoc <tên> <seed> <lam_den>
  local ten=$1 s=$2 lam=$3
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train Student $ten (xa, seed $s, lam_vung 10, lam_den $lam)"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --kien_truc xa --ten $ten --seed $s --lam_vung 10 --lam_den $lam > $LOG/$ten.log 2>&1
  fi
}
xuat() {
  local ten=$1
  if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten/train | wc -l) -lt 992 ]; then
    echo "[$(date +%H:%M)] xuất ảnh $ten"
    cd $TL/b3; $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
  fi
}

hoc sden10 0 10; hoc sden30 0 30
for t in sden10 sden30; do xuat $t; done
cd $TL/b3
echo "[$(date +%H:%M)] tương phản NGƯỜI (val 151)"
$PY ../scene_v1/do_tuong_phan.py --bien_the sage nosam_kp841 sxa10 sden10 sden30
echo "[$(date +%H:%M)] tương phản ĐÈN (val 151)"
$PY ../scene_v1/do_tuong_phan.py --lop 4 --px_min 30 --bien_the sage nosam_kp841 sxa10 sden10 sden30
echo "[$(date +%H:%M)] rào ảnh val 151"
cd $TL/cai_tien; $PY chi_so_ghep.py --bien_the sage nosam_kp841 sxa10 sden10 sden30 --tap dev --hau kp_sden 2>&1 | tail -n 8

# 21/11 22h30 (Mân chốt): B0 CHỈ chạy khi B0 sxa10 (Kaggle) xác nhận còn nhầm đèn->biển. Mặc định dừng ở đây.
# 21/11 23h40 (Mân chốt): mIoU là tiêu chí CHÍNH; tương phản người/đèn chỉ CHẨN ĐOÁN (bỏ cổng "người >= 90% sxa10" – đổi SAU khi thấy số: sden10 77% sxa10, tỉ lệ 1,76 ≈ A2 1,79). Chấm: B0=1 BT_B0=sden10.
if [ "${B0:-0}" != 1 ]; then echo "[$(date +%H:%M)] XONG phần ghép (B0: chờ kết quả sxa10 Kaggle; chạy B0=1 bash scene_v1/chay_sden.sh)"; exit 0; fi

cd $TL/b3
for bt in ${BT_B0:-sden10 sden30 sxa10}; do
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0 xong"
  have=""; for b in sden10 sden30 sxa10; do [ -f $SAGE_RUNS/${b}_b0_s0/dev_ket_qua_last.json ] && have="$have $b"; done
  $PY so_sanh_lop_du.py --bien_the $have nosam_kp841 nosam_kp841_vung10 --model b0 --tap dev --seeds 0 --min_anh 20 2>&1 | sed -n '/>= 20 ảnh/,/^$/p'
  for b in $have nosam_kp841 nosam_kp841_vung10; do
    echo "  $b: $(grep -E '^(person|traffic_sign|traffic_light),' $SAGE_RUNS/${b}_b0_s0/dev_iou_lop_last.csv | tr '\n' ' ')"
  done
done
echo XONG
