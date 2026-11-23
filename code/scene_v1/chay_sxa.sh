#!/bin/bash
# S-xa (21/11): Student + nhánh ngữ cảnh rộng (b3/sage_student/model_xa.py, 0,252M tham số; gốc 0,136M).
# NGUYÊN NHÂN cần sửa (phan_tich_nut_that.py): A2 làm person tăng nhưng biển báo/car giảm ở CẢ 3 seed (A2−nosam person
#   +1,3/+4,6/+1,3; biển −4,0/−3,7/−3,2; car −0,7/−0,9/−1,2) -> Student nhìn hẹp, không phân biệt vật -> luật chung cả ảnh.
#   λ10mr (thêm biển/đèn/cột vào loss) với Student gốc làm person TỤT 47,4 -> 43,8 (ép nhiều lớp lên mạng nhỏ).
# BIẾN THỂ (seed 0): xa0 = S-xa loss nosam (tách tác dụng kiến trúc) | xa10 = S-xa + A2 λ10 | xa10mr = S-xa + A2 λ10 lớp mở rộng.
# ĐẶT TRƯỚC (val151 B0 s0 máy nhà, 10 lớp >=20 ảnh; mốc máy nhà: nosam 56,99, A2 λ10 57,44, λ10mr 57,02):
#   Ứng viên ĐI TIẾP (seed 1, 2) nếu: 10 lớp >= 57,99 (nosam + 1,0) VÀ person > nosam VÀ biển báo >= nosam − 1,0
#   VÀ qua rào ảnh (MS-SSIM >= 99%, Q_ABF >= 98% SAGE). Nhiều ứng viên đạt -> lấy 10 lớp cao nhất (cách < 0,3 -> lấy bản đơn giản hơn: xa0 < xa10 < xa10mr).
#   Không ai đạt: so xa0 − nosam để biết kiến trúc có giúp không, rồi sửa tiếp (không bỏ hướng khi chưa rõ nguyên nhân).
# Thứ tự: 3 Student s0 (~20 ph/cái) -> xuất ảnh -> rào ảnh + tương phản -> Student s1 của xa10, xa10mr (để Kaggle)
#         -> B0 s0 xa10mr, xa10, xa0 (~1,4 h/cái). Tổng ~5,5 h. Chạy lại được (bỏ qua bước đã xong, B0 có --tiep).
# 21/11: mặc định CHỈ chạy phần ghép (Student + ảnh + rào); B0 chấm trên Kaggle (xem dưới).
# Chạy (Git Bash): bash scene_v1/chay_sxa.sh > scene_v1/chay_sxa.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
LOG=$TL/scene_v1
MR="--lop_vat 4 5 8 9 10 11 12 13 14"

hoc() {  # hoc <tên> <seed> <lam> [mr]
  local ten=$1 s=$2 lam=$3 them=""; [ "$4" = mr ] && them=$MR
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train Student $ten (xa, seed $s, lam_vung $lam $them)"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --kien_truc xa --ten $ten --seed $s --lam_vung $lam $them > $LOG/$ten.log 2>&1
  fi
}
xuat() {
  local ten=$1
  if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten/train | wc -l) -lt 992 ]; then
    echo "[$(date +%H:%M)] xuất ảnh $ten"
    cd $TL/b3; $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
  fi
}

hoc sxa0 0 0; hoc sxa10 0 10; hoc sxa10mr 0 10 mr
for t in sxa0 sxa10 sxa10mr; do xuat $t; done
echo "[$(date +%H:%M)] tương phản người (val 151)"
cd $TL/b3; $PY ../scene_v1/do_tuong_phan.py --bien_the sage nosam_kp841 nosam_kp841_vung10 sxa0 sxa10 sxa10mr
echo "[$(date +%H:%M)] rào ảnh val 151"
cd $TL/cai_tien; $PY chi_so_ghep.py --bien_the sage nosam_kp841 nosam_kp841_vung10 sxa0 sxa10 sxa10mr --tap dev --hau kp_sxa 2>&1 | tail -n 8

hoc sxa10_s1 1 10; hoc sxa10mr_s1 1 10 mr   # sẵn cho Kaggle (chấm B0 seed 1)

# 21/11 (Mân chốt): CHẤM B0 TRÊN KAGGLE (so thẳng với mốc 3 seed sage/tb/nosam/A2 trên Kaggle). Máy nhà chỉ ghép.
# Muốn chấm B0 ở máy nhà như cũ: B0=1 bash scene_v1/chay_sxa.sh
if [ "${B0:-0}" != 1 ]; then echo "[$(date +%H:%M)] XONG phần ghép (B0 chấm trên Kaggle)"; echo XONG; exit 0; fi

cd $TL/b3
for bt in sxa10mr sxa10 sxa0; do
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0 xong"
  # so ngay sau mỗi run (đọc sớm được)
  have=""; for b in sxa10mr sxa10 sxa0; do [ -f $SAGE_RUNS/${b}_b0_s0/dev_ket_qua_last.json ] && have="$have $b"; done
  $PY so_sanh_lop_du.py --bien_the $have nosam_kp841 nosam_kp841_vung10 --model b0 --tap dev --seeds 0 --min_anh 20 2>&1 | sed -n '/>= 20 ảnh/,/^$/p'
  for b in $have nosam_kp841 nosam_kp841_vung10; do
    echo "  $b: $(grep -E '^(person|traffic_sign),' $SAGE_RUNS/${b}_b0_s0/dev_iou_lop_last.csv | tr '\n' ' ')"
  done
done
echo XONG
