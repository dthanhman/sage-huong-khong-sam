#!/bin/bash
# B1 (19/11): ảnh TDFusion (CVPR 2025, checkpoint FMB công khai) có hơn nosam theo giao thức của nhóm không?
# Bước: xuất ảnh tdfusion -> chỉ số ghép val 151 -> B0 (841 ảnh, 100 ep, batch 8, 14 lớp, last.pt, chấm val 151):
#   tdfusion s0 -> nosam_kp841 s0 (chạy lại TRÊN MÁY NHÀ để so cùng máy) -> tdfusion s1 -> so cặp.
# Đặt trước: tdfusion đáng làm B2 nếu hơn nosam (cùng máy) >= +1,0 ở seed 0 và vẫn dương ở seed 1.
# Lưu ý: checkpoint TDFusion đã học toàn bộ FMB train có nhãn (đã thấy val 151) -> nó được lợi.
# Chạy lại được: bước nào có kết quả thì bỏ qua; B0 dở thì train tiếp (--tiep).
# Chạy (Git Bash): bash scene_v1/chay_b1_tdfusion.sh > scene_v1/chay_b1_tdfusion.out 2>&1   (~4 h)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}

echo "[$(date +%H:%M)] xuất ảnh tdfusion"
cd $TL/cai_tien
$PY xuat_tdfusion.py --repo ${TDFUSION:?đặt TDFUSION=<thư mục clone HaowenBai/TDFusion>}
echo "[$(date +%H:%M)] chỉ số ghép val 151"
$PY chi_so_ghep.py --bien_the sage tb nosam_kp841 tdfusion --tap dev --hau kp_td 2>&1 | tail -n 7

cd $TL/b3
for cap in tdfusion:0 nosam_kp841:0 tdfusion:1; do
  bt=${cap%:*}; s=${cap#*:}; run=$SAGE_RUNS/${bt}_b0_s$s
  if [ ! -f $run/dev_ket_qua_last.json ]; then
    echo "[$(date +%H:%M)] train B0 $bt seed $s"
    $PY train_seg.py --bien_the $bt --model b0 --seed $s --tiep > $SAGE_RUNS/log_${bt}_b0_s$s.txt 2>&1
    $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${bt}_b0_s$s.txt 2>&1
  fi
  echo "[$(date +%H:%M)] $bt s$s: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1)"
done

echo "[$(date +%H:%M)] so cặp (cùng máy nhà, seed 0)"
$PY so_sanh_cap.py --bien_the nosam_kp841 m11a_l0_e30_bn m11a_l5_e30_bn tdfusion --model b0 --tap dev --seeds 0 2>&1 | sed -n 3,40p
echo XONG
