#!/bin/bash
# Nền A + CE, bản sửa BatchNorm (19/11): Student không SAM (init nosam_kp841) học cùng B0;
# lam 5, lr_st 5e-4, B0 khởi động 15 epoch rồi học chậm 10 lần (b0_st 0.1), --bn_dong, 30 epoch.
# Sau đó xuất ảnh, chỉ số ghép val 151, đo lệch so với nosam_kp841.
# Chạy: bash scene_v1/chay_nena_ce.sh   (từ tai_lap; ~50 phút)
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
PY=${PY:-python}
ten=${TEN:-m11a_l5_e30_bn}
LOG=$TL/scene_v1/$ten.log
if [ ! -f $TL/b3/sage_student/${ten}_cfg.json ]; then
  echo "[$(date +%H:%M)] train $ten"
  cd $TL/cai_tien
  $PY m11_hoc_chung.py --ten $ten --init nosam_kp841 --beta 1 --lam ${LAM:-5} --lr_st 5e-4 \
      --b0_st 0.1 --warm_seg 15 --epochs ${EPOCHS:-30} --do_grad 10 --bn_dong > $LOG 2>&1
fi
echo "[$(date +%H:%M)] xuất ảnh"
cd $TL/b3
$PY xuat_anh.py --bien_the $ten >> $LOG 2>&1
echo "[$(date +%H:%M)] chỉ số ghép + lệch ảnh"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage tb nosam_kp841 m11_kp $ten --tap dev --hau kp_$ten >> $LOG 2>&1
$PY $TL/scene_v1/do_lech_anh.py nosam_kp841 $ten >> $LOG 2>&1
grep "^epoch" $LOG | tail -4; tail -n 9 $LOG
echo XONG
