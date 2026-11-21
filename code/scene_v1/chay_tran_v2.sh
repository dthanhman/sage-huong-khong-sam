#!/bin/bash
# CẬN TRÊN v2 (21/11). Lần vis_ir4 (20/11, 56,80 / 10 lớp) KHÔNG phải trần: IR ghép thô ở conv đầu, kênh IR mới học
# bằng lr encoder rất nhỏ, model pretrained chỉ quen RGB. Ở đây sửa đúng 2 điểm yếu đó (train_seg.py):
#   vis_ir4_lr10 : như vis_ir4, conv đầu vào lr x10.
#   hai_nhanh    : 2 encoder MiT-B0 pretrained (VIS | IR nhân 3 kênh), trộn mỗi tầng v + i + conv1x1([v,i]) (khởi tạo 0),
#                  chung decoder. Kiểu CMX đơn giản, 7,2M tham số (gấp đôi). CHỈ để đo cận trên, KHÔNG phải pp nộp.
# Giao thức như sàng lọc: 841 Train / val 151, 100 epoch, batch 8, 14 lớp, last.pt, seed 0, máy nhà.
#
# ĐẶT TRƯỚC (đích nhóm 21/11: pp nhóm "hơn nhất quán" so với CẢ SAGE và TB + rào chắn; số chính 10 lớp >= 20 ảnh).
# Cận trên dùng để biết CÒN BAO NHIÊU CHỖ và Ở LỚP NÀO, không phải để đậu/rớt.
# Mốc s0 (10 lớp): sage 56,90 / tb 56,98 (Kaggle), nosam 56,99 / vis_ir4 56,80 / vis 56,27 (máy nhà).
#   best = max(vis_ir4_lr10, hai_nhanh)
#   best − max(sage, tb) >= +1,0  -> có thông tin IR mà ảnh ghép đang làm rơi: chạy seed 1, 2 cho best để xác nhận,
#                                    rồi soi IoU từng lớp: lớp nào best > sage/tb >= 2 điểm = chỗ loss của pp nhóm nhắm vào.
#   best − max(sage, tb) <  +0,5  -> ở B0, dùng IR kiểu mạnh cũng không hơn ảnh ghép: chỗ hơn SAGE/TB phải tìm ở lớp
#                                    cụ thể (person, A2) và giảm dao động seed, KHÔNG trông vào thêm thông tin đầu vào.
#   ở giữa                        -> vùng xám, báo đúng như vậy; quyết chạy thêm seed cùng nhóm.
#   1 seed, khác máy với sage/tb -> chỉ đọc định hướng.
#
# Chạy (Git Bash, khi máy rảnh – script tự CHỜ chay_a2_vung (λ30, λ10mr) in XONG):
#   bash scene_v1/chay_tran_v2.sh > scene_v1/chay_tran_v2.out 2>&1
#   Chỉ 1 biến thể: BT=hai_nhanh bash scene_v1/chay_tran_v2.sh ...   | Bỏ chờ: CHO=0 ...
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
GOP=$TL/scene_v1/runs_b0_gop
BT=${BT:-"vis_ir4_lr10 hai_nhanh"}
A2OUT=$TL/scene_v1/chay_a2_vung_30_10mr.out

if [ "${CHO:-1}" = 1 ] && [ -f $A2OUT ]; then
  echo "[$(date +%H:%M)] chờ A2 (λ30, λ10mr) xong..."
  until tail -n 1 $A2OUT | grep -q '^XONG'; do sleep 120; done
fi

cd $TL/b3
for bt in $BT; do
  run=$SAGE_RUNS/${bt}_b0_s0
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed 0"
    if ! $PY train_seg.py --bien_the $bt --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1; then
      # 2 nhánh có thể thiếu VRAM (3050 6 GB) ở batch 8 -> batch 4 x tích lũy 2 (chỉ khi chưa có resume.pt)
      grep -q "OutOfMemory\|out of memory" $SAGE_RUNS/log_${bt}_b0_s0.txt && [ ! -f $run/resume.pt ] || exit 1
      echo "[$(date +%H:%M)]   $bt thiếu VRAM ở batch 8 -> chạy batch 4 x tích lũy 2 (GHI CHÚ khi đọc kết quả)"
      $PY train_seg.py --bien_the $bt --model b0 --seed 0 --batch 4 --tich_luy 2 >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
    fi
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s0.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s0: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1); $(grep person $run/dev_iou_lop_last.csv)"
  [ -d $GOP/${bt}_b0_s0 ] || cp -r $run $GOP/
done

echo "[$(date +%H:%M)] mIoU lớp đủ ảnh, seed 0 (cận trên + vis_ir4/vis/nosam_nha: máy nhà; sage/tb: Kaggle)"
cp $GOP/so_sanh_lop_du_dev_b0.csv $GOP/_bak.csv  # giữ bảng 3 seed, không ghi đè
SAGE_RUNS=$GOP $PY so_sanh_lop_du.py --bien_the $BT vis_ir4 vis nosam_kp841_nha sage tb --model b0 --tap dev --seeds 0 \
  --min_anh 20 10 --cap $(for b in $BT; do echo "$b-sage $b-tb $b-vis_ir4"; done) 2>&1
mv $GOP/so_sanh_lop_du_dev_b0.csv $GOP/so_sanh_lop_du_dev_b0_tran_v2.csv; mv $GOP/_bak.csv $GOP/so_sanh_lop_du_dev_b0.csv

echo "[$(date +%H:%M)] IoU từng lớp (val 151, s0)"
$PY - <<EOF
import pandas as pd
g = "$GOP"
bts = "$BT".split() + ["vis_ir4", "vis", "nosam_kp841_nha", "sage", "tb"]
d = {b: pd.read_csv(f"{g}/{b}_b0_s0/dev_iou_lop_last.csv", index_col=0)["IoU_280"] for b in bts}
print(pd.DataFrame(d).dropna(how="all").round(1).to_string())
EOF
echo XONG
