#!/bin/bash
# L_int THEO LỚP (21/11 ~0h): MSE cường độ với trọng số IR riêng từng vùng lớp (mask từ NHÃN train, chỉ lúc train).
# BẰNG CHỨNG (val151 B0): mọi ảnh ghép (SAGE/TB/nosam/A2/sxa) được person nhờ IR (VIS 28,0 -> ~46) nhưng MẤT lớp VIS mạnh
#   (truck VIS 13,0 vs ghép 3,7; đèn 54,8 vs 40,0; biển 47,5 vs 44,7; vỉa hè 24,1 vs 20,8). hai_nhanh lấy CẢ HAI phía
#   (person 53,3, truck 11,2, vỉa hè 26,4) -> +2,2. Họ "làm nổi vật thể" (A1/A2/A4/sxa) chỉ đẩy về IR = mới một nửa.
# TRỌNG SỐ: tự động theo luật đặt trước (scene_v1/tinh_w_lop.py -> w_lop.json): VIS hơn ghép >= 2 -> w_IR 0,1;
#   kém >= 5 -> 0,7; còn lại 0,4 (gốc). Kết quả: vỉa hè/đèn/biển/truck/bus 0,1; person 0,7; còn lại 0,4.
# BIẾN THỂ (Student gốc 0,136M, seed 0): lint = nosam + L_int lớp | lint_a2 = lint + A2 λ10 (A2 lấy person, lint giữ lớp VIS).
# TIÊU CHÍ (mIoU TRƯỚC; val151 B0 s0 máy nhà, 10 lớp >= 20 ảnh; mốc máy nhà nosam 56,99, A2 λ10 57,44):
#   ĐI TIẾP (seed 1, 2 + so 3 seed với SAGE/TB) nếu 10 lớp >= 57,99 (nosam + 1,0) VÀ qua rào (MS-SSIM >= 99%, Q_ABF >= 98% SAGE).
#   Chẩn đoán (không chặn): IoU truck/biển/vỉa hè tăng so nosam, person không giảm.
# Thứ tự: chờ visvis xong -> 2 Student (~20 ph) -> ảnh -> rào -> B0 s0 lint, lint_a2, rồi sden10 (~1,4 h/cái).
# Chạy (Git Bash): bash scene_v1/chay_lint.sh > scene_v1/chay_lint.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
LOG=$TL/scene_v1
W=$TL/scene_v1/w_lop.json

until grep -q '^XONG' $LOG/chay_visvis.out 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] visvis xong"

hoc() {  # hoc <tên> <lam_vung>
  local ten=$1 lam=$2
  if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
    echo "[$(date +%H:%M)] train Student $ten (goc, seed 0, w_lop, lam_vung $lam)"
    cd $TL/e17
    $PY train_student_nosam.py --chi_train --ten $ten --seed 0 --w_lop $W --lam_vung $lam > $LOG/$ten.log 2>&1
  fi
}
xuat() {
  local ten=$1
  if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten/train | wc -l) -lt 992 ]; then
    echo "[$(date +%H:%M)] xuất ảnh $ten"
    cd $TL/b3; $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
  fi
}

hoc lint 0; hoc lint_a2 10
for t in lint lint_a2; do xuat $t; done
echo "[$(date +%H:%M)] rào ảnh val 151"
cd $TL/cai_tien; $PY chi_so_ghep.py --bien_the sage nosam_kp841 nosam_kp841_vung10 lint lint_a2 --tap dev --hau kp_lint 2>&1 | tail -n 8

cd $TL/b3
for bt in lint lint_a2 sden10; do
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0 xong"
  have=""; for b in lint lint_a2 sden10; do [ -f $SAGE_RUNS/${b}_b0_s0/dev_ket_qua_last.json ] && have="$have $b"; done
  $PY so_sanh_lop_du.py --bien_the $have nosam_kp841 nosam_kp841_vung10 --model b0 --tap dev --seeds 0 --min_anh 20 2>&1 | sed -n '/>= 20 ảnh/,/^$/p'
  for b in $have nosam_kp841 nosam_kp841_vung10; do
    echo "  $b: $(grep -E '^(person|truck|traffic_sign|sidewalk),' $SAGE_RUNS/${b}_b0_s0/dev_iou_lop_last.csv | tr '\n' ' ')"
  done
done
echo XONG
