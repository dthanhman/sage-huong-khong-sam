#!/bin/bash
# TRẦN 4 KÊNH (20/11; ý cũ "E18 cận trên", chưa từng chạy): SegFormer B0 nhận THẲNG VIS (RGB) + IR, không ghép ảnh.
# Câu hỏi: nếu không mất thông tin nào, B0 đạt tới đâu? -> còn bao nhiêu chỗ cho mạng ghép cải thiện.
# Kèm mốc VIS (chỉ ảnh thường) trên cùng giao thức 841/151 (chưa có) -> 4 kênh − VIS = giá trị thật của IR.
# Giao thức: như sàng lọc (841 ảnh, 100 epoch, batch 8, 14 lớp, chấm val 151 last.pt), seed 0, MÁY NHÀ (~1h15/lượt).
# Tầng đầu: kênh IR khởi tạo = TB trọng số RGB pretrained; IR chuẩn hóa bằng TB MEAN/STD ImageNet (train_seg.py, BT_4KENH).
# ĐẶT TRƯỚC (đọc kết quả, số chính mIoU 10 lớp >= 20 ảnh), mốc nosam s0 máy nhà 56,99, đích 5% ≈ 60,0:
#   4 kênh < ~59 (dưới đích)  -> ghép ảnh khó đạt +5% trên giao thức này -> bàn lại đích/cách với nhóm.
#   4 kênh >= ~60             -> thông tin IR có thật, mạng ghép đang làm rơi -> tiếp tục cải thiện nosam (A2...).
#   1 seed, so khác máy với sage/tb (Kaggle) -> chỉ đọc định hướng.
# Chạy (Git Bash, SAU A2): bash scene_v1/chay_tran_4kenh.sh > scene_v1/chay_tran_4kenh.out 2>&1
#   Chỉ 4 kênh: BT=vis_ir4 bash scene_v1/chay_tran_4kenh.sh ...
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
GOP=$TL/scene_v1/runs_b0_gop
BT=${BT:-"vis_ir4 vis"}

cd $TL/b3
for bt in $BT; do
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1); $(grep person $run/dev_iou_lop_last.csv)"
  [ -d $GOP/${bt}_b0_s0 ] || cp -r $run $GOP/
done
[ -d $GOP/nosam_kp841_nha_b0_s0 ] || cp -r $SAGE_RUNS/nosam_kp841_b0_s0 $GOP/nosam_kp841_nha_b0_s0

echo "[$(date +%H:%M)] mIoU lớp đủ ảnh, seed 0 (vis_ir4/vis/nosam_kp841_nha: máy nhà; sage/tb/nosam_kp841: Kaggle)"
cp $GOP/so_sanh_lop_du_dev_b0.csv $GOP/_bak.csv  # giữ bảng 3 seed, không ghi đè
SAGE_RUNS=$GOP $PY so_sanh_lop_du.py --bien_the $BT nosam_kp841_nha nosam_kp841 sage tb --model b0 --tap dev --seeds 0 \
  --min_anh 20 10 --cap $(for b in $BT; do echo "$b-nosam_kp841_nha $b-sage $b-tb"; done) 2>&1
mv $GOP/so_sanh_lop_du_dev_b0.csv $GOP/so_sanh_lop_du_dev_b0_tran4kenh.csv; mv $GOP/_bak.csv $GOP/so_sanh_lop_du_dev_b0.csv
echo XONG
