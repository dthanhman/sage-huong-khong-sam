#!/bin/bash
# ORACLE THEO LỚP (22/11) – CHẨN ĐOÁN, không phải phương pháp. Không train Student, chỉ train B0.
# CÂU HỎI: nếu ảnh ghép chọn nguồn theo lớp HOÀN HẢO (dùng nhãn thật), giao thức "B0 chỉ nhìn ảnh ghép" có vượt ~57,5 không?
# VÌ SAO HỎI: đo hệ số kéo về IR a theo lớp (val151): lint KHÔNG làm theo luật (person a 0,41 < nosam 0,48, mục tiêu 0,7;
#   vỉa hè/biển/truck/bus ~0,2–0,4, mục tiêu 0,1) -> Student 0,136M không biết vùng nào là lớp nào. Muốn thêm "nhánh tạm"
#   (đầu phân đoạn phụ lúc train, bỏ khi suy luận) dạy Student ngữ nghĩa thì phải biết luật theo lớp có đáng không đã.
# BIẾN THỂ (xuat_anh.py, w theo lớp từ w_lop.json, mờ Gauss 4 px):
#   orc_dung  = luật lint đúng (person 0,7; vỉa hè/đèn/biển/truck/bus 0,1; còn lại 0,4)
#   orc_nguoc = luật ngược w' = 0,8 − w (cùng đường nối theo nhãn, sai luật) -> đối chứng phần lợi do LỘ RANH GIỚI NHÃN.
# ĐẶT TRƯỚC (val151, B0 s0, 10 lớp >= 20 ảnh, cùng máy nhà):
#   GO  : orc_dung − orc_nguoc >= +1,0 VÀ orc_dung >= nosam (máy nhà 56,99) + 1,5
#         -> làm Student + đầu phân đoạn phụ (chỉ lúc train) để tự học luật theo lớp.
#   DỪNG: orc_dung − orc_nguoc < +1,0 -> luật theo lớp không có giá trị kể cả khi làm hoàn hảo
#         (lợi nếu có chỉ do lộ nhãn) -> bỏ cả họ "chọn nguồn theo lớp".
#   orc_dung cao nhưng orc_dung < nosam + 1,5 -> có giá trị nhưng trần thấp, không đáng làm Student mới.
# ~1,2–1,4 h mỗi B0, tổng ~2,5–3 h. Chạy lại được (B0 có --tiep).
# Chạy (Git Bash): bash scene_v1/chay_oracle.sh > scene_v1/chay_oracle.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
cd $TL/b3
for bt in orc_dung orc_nguoc; do
  [ $(ls $SAGE_ANH/$bt/train 2>/dev/null | wc -l) -ge 992 ] || $PY xuat_anh.py --bien_the $bt
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0 xong"
done
cp $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv $SAGE_RUNS/_bak.csv
$PY so_sanh_lop_du.py --bien_the orc_dung orc_nguoc nosam_kp841 vis hai_nhanh --model b0 --tap dev --seeds 0 --min_anh 20 2>&1 | sed -n '/>= 20 ảnh/,/^$/p'
mv $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv $SAGE_RUNS/so_sanh_lop_du_dev_b0_oracle.csv; mv $SAGE_RUNS/_bak.csv $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv
echo XONG
