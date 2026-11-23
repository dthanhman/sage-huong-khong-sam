#!/bin/bash
# 21/11 – ĐỐI CHỨNG hai_nhanh_visvis: y hệt hai_nhanh (2 encoder MiT-B0, trộn conv1x1, cùng lr/batch/epoch)
# nhưng nhánh 2 nhận ẢNH XÁM CỦA VIS thay cho IR. Câu hỏi: +2,2 của hai_nhanh là nhờ THÔNG TIN IR hay nhờ 2 ENCODER?
# Mốc (val151 B0, 10 lớp >=20 ảnh, TB 3 seed): hai_nhanh 59,42 | nosam 57,50 | sage 57,25 | tb 57,11.
# TIÊU CHÍ ĐẶT TRƯỚC (TB 3 seed, 10 lớp):
#   visvis <= 57,9  -> lợi thế hai_nhanh là do IR  -> còn thông tin IR mà ảnh ghép 1 kênh làm mất (đi tiếp S-xa / kênh phụ).
#   visvis >= 58,9  -> lợi thế chủ yếu do mô hình chấm 2 encoder -> hạ kỳ vọng "tận dụng hai_nhanh".
#   ở giữa          -> chưa kết luận; xem thêm so cặp hai_nhanh − visvis và IoU person.
# Máy nhà (cùng máy với hai_nhanh s0–s2), ~1,5 h/seed, ~4,5 h tổng. Có --tiep: chạy lại đúng lệnh là train tiếp.
# Chạy: bash scene_v1/chay_visvis.sh > scene_v1/chay_visvis.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
GOP=$TL/scene_v1/runs_b0_gop
BT=hai_nhanh_visvis
cd $TL/b3
for s in ${SEEDS:-0 1 2}; do
  run=$SAGE_RUNS/${BT}_b0_s$s
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $BT seed $s"
    $PY train_seg.py --bien_the $BT --model b0 --seed $s --tiep >> $SAGE_RUNS/log_${BT}_b0_s$s.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${BT}_b0_s$s.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $BT s$s xong"
  rm -rf $GOP/${BT}_b0_s$s; cp -r $run $GOP/
done
echo "[$(date +%H:%M)] SO SÁNH (10 lớp >=20 ảnh, 3 seed)"
cp $GOP/so_sanh_lop_du_dev_b0.csv $GOP/_bak.csv
SAGE_RUNS=$GOP $PY so_sanh_lop_du.py --bien_the hai_nhanh $BT nosam_kp841 sage tb --model b0 --tap dev --seeds 0 1 2 --min_anh 20
mv $GOP/so_sanh_lop_du_dev_b0.csv $GOP/so_sanh_lop_du_dev_b0_visvis.csv; mv $GOP/_bak.csv $GOP/so_sanh_lop_du_dev_b0.csv
echo XONG
