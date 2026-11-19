#!/bin/bash
# Nền B + CE, thử ngắn (19/11): ảnh = TB + s*Student, init tbsua_w150; lam 5, lr_st 5e-4;
# B0 khởi động 15 epoch (Student đứng yên) rồi B0 học chậm 10 lần (b0_st 0.1) để Student phải tự đổi ảnh.
# 20 epoch = 5 epoch Student học. Đạt nếu |ảnh - tbsua_w150| >= 2 mức xám và vẫn qua sàn MS-SSIM>=99, Q_ABF>=98 %SAGE.
# Chạy: bash scene_v1/chay_nenb_ce.sh   (từ thư mục tai_lap)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
ten=${TEN:-m11b_l5_b01}
LOG=$TL/scene_v1/$ten.log
echo "[$(date +%H:%M)] train $ten"
cd $TL/cai_tien
$PY m11_hoc_chung.py --ten $ten --nen tb --init tbsua_w150 --beta 1 --lam ${LAM:-5} --lr_st 5e-4 \
    --b0_st 0.1 --warm_seg 15 --epochs ${EPOCHS:-20} --do_grad 10 ${BN:+--bn_dong} > $LOG 2>&1
echo "[$(date +%H:%M)] xuất ảnh"
cd $TL/b3
$PY xuat_anh.py --bien_the $ten >> $LOG 2>&1
echo "[$(date +%H:%M)] chỉ số ghép + lệch ảnh"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage tb tbsua_w150 $ten --tap dev --hau kp_$ten >> $LOG 2>&1
$PY $TL/scene_v1/do_lech_anh.py tbsua_w150 $ten tb >> $LOG 2>&1
grep "^epoch" $LOG | tail -6; tail -n 14 $LOG
echo XONG
