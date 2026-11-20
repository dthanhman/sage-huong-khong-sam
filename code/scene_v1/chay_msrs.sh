#!/bin/bash
# MSRS (20/11): train mạng ghép nosam trên 841 ảnh FMB + 1.083 cặp MSRS train (không nhãn, chỉ loss ghép).
# Giả thuyết: thêm cảnh đường phố (nửa đêm) giúp nosam ghép tốt hơn ở ảnh tối -> mIoU FMB tăng.
# Chỉ đổi DỮ LIỆU: cùng w_ir với mốc so sánh, cùng 100 epoch (thêm ảnh nên số bước nhiều hơn ~2,3 lần – ghi chú khi báo cáo).
# Segmenter B0 vẫn train/chấm trên FMB như cũ (841 / val 151). KHÔNG dùng MSRS test (giữ làm dataset thứ 2).
# Chọn w_ir: W=04 (mặc định, gốc 3Y+2IR, mốc nosam_kp841) hoặc w tốt nhất của A1, vd W=06 (mốc nosam_kp841_w06).
# ĐẶT TRƯỚC: ĐẠT nếu hơn mốc cùng w (cùng máy, seed 0) >= +1,0 -> chạy seed 1 để xác nhận.
#            Chỉ số ghép: báo kèm (rào chắn MS-SSIM >= 99%, Q_ABF >= 98% SAGE).
# Chạy lại được: bước nào có kết quả thì bỏ qua; B0 dở thì train tiếp (--tiep).
# Chạy (Git Bash, SAU B1 và A1): W=06 bash scene_v1/chay_msrs.sh > scene_v1/chay_msrs.out 2>&1   (~2 h)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
LOG=$TL/scene_v1
MSRS=${MSRS:?đặt MSRS=<thư mục MSRS train có vi/ ir/>}
W=${W:-04}

if [ "$W" = "04" ]; then MOC=nosam_kp841; THAM_W=""; else MOC=nosam_kp841_w$W; THAM_W="--w_ir 0.${W#0}"; fi
ten=nosam_kp841_msrs_w$W
echo "[$(date +%H:%M)] w_ir 0.${W#0} | mốc $MOC | biến thể $ten"
[ -f $SAGE_RUNS/${MOC}_b0_s0/dev_ket_qua_last.json ] || { echo "Chưa có B0 của mốc $MOC (chạy B1/A1 trước)"; exit 1; }

if [ ! -f $TL/b3/sage_student/$ten.pt ]; then
  echo "[$(date +%H:%M)] train $ten (FMB 841 + MSRS train)"
  cd $TL/e17
  $PY train_student_nosam.py --chi_train --ten $ten --seed 0 $THAM_W --them $MSRS > $LOG/$ten.log 2>&1
  grep -E "cặp|Tổng" $LOG/$ten.log
fi
if [ ! -d $SAGE_ANH/$ten ] || [ $(ls $SAGE_ANH/$ten | wc -l) -lt 1500 ]; then
  echo "[$(date +%H:%M)] xuất ảnh FMB bằng $ten"
  cd $TL/b3
  $PY xuat_anh.py --bien_the $ten >> $LOG/$ten.log 2>&1
fi

echo "[$(date +%H:%M)] chỉ số ghép val 151"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage $MOC $ten --tap dev --hau kp_msrs_w$W 2>&1 | tail -n 6

cd $TL/b3
run=$SAGE_RUNS/${ten}_b0_s0
if [ ! -f $run/dev_ket_qua_last.json ]; then
  echo "[$(date +%H:%M)] train B0 $ten seed 0"
  $PY train_seg.py --bien_the $ten --model b0 --seed 0 --tiep > $SAGE_RUNS/log_${ten}_b0_s0.txt 2>&1
  $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${ten}_b0_s0.txt 2>&1
fi
echo "[$(date +%H:%M)] $ten s0: $(grep -o '"mIoU_co_mat": [0-9.]*' $run/dev_ket_qua_last.json | head -1)"

echo "[$(date +%H:%M)] so cặp với mốc (cùng máy nhà, seed 0)"
$PY so_sanh_cap.py --bien_the $MOC $ten --model b0 --tap dev --seeds 0 2>&1 | sed -n 3,40p
echo "[$(date +%H:%M)] mIoU theo độ sáng (phụ; nhóm sáng/tối ~ đoạn cảnh trên val 151)"
$PY ../scene_v1/miou_theo_sang.py --bien_the $MOC $ten --model b0 --seeds 0 --goc $MOC 2>&1 | tail -n 20
echo XONG
