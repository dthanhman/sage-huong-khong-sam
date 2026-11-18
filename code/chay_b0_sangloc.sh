#!/bin/bash
# Sàng lọc M11 bằng SegFormer-B0 trên tập khám phá scene_v1 (train 841, chấm val 151 bằng last.pt).
# 4 biến thể x 3 seed = 12 lượt (~1,2 h/lượt máy nhà). Vòng seed ở NGOÀI: xong seed 0 của cả 4 biến thể rồi mới sang seed 1,
# để dừng giữa chừng vẫn có cặp đủ để so. Chạy lại được (run đã có last.pt thì bỏ qua train; train dở thì --tiep).
# Quy tắc kết luận: tai_lap/QUY_TAC_CHAM.md (thang 3 mức, so với CẢ tb và sage).
# Chạy: bash scene_v1/chay_b0_sangloc.sh   (từ tai_lap; phiên phải mở suốt)
set -e
export SAGE_MAN=D:/CS406/DoAn/Sage/tai_lap/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=D:/CS406/DoAn/Sage/luu_tru/tim_gap_b0_0811/seg/anh
export SAGE_RUNS=D:/CS406/DoAn/Sage/tai_lap/scene_v1/runs_b0
PY=D:/Miniconda3/envs/DoAn/python.exe
BT="tb sage nosam_kp841 m11_kp"
SEEDS="${SEEDS:-0 1 2}"
cd D:/CS406/DoAn/Sage/tai_lap/b3
mkdir -p $SAGE_RUNS

for s in $SEEDS; do
  for b in $BT; do
    run=$SAGE_RUNS/${b}_b0_s$s
    if [ ! -f $run/last.pt ]; then
      echo "[$(date +%H:%M)] train $b seed $s"
      $PY train_seg.py --bien_the $b --model b0 --seed $s --tiep > $SAGE_RUNS/log_${b}_b0_s$s.txt 2>&1
    fi
    if [ ! -f $run/dev_nham_last.npz ]; then
      echo "[$(date +%H:%M)] chấm val 151: $b seed $s"
      $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${b}_b0_s$s.txt 2>&1
    fi
  done
done

echo "[$(date +%H:%M)] so cặp"
$PY so_sanh_cap.py --bien_the $BT --model b0 --tap dev \
  --cap m11_kp-tb m11_kp-sage m11_kp-nosam_kp841 tb-sage > $SAGE_RUNS/so_sanh_cap.txt 2>&1
cat $SAGE_RUNS/so_sanh_cap.txt
echo XONG
