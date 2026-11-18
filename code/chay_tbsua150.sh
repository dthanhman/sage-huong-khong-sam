#!/bin/bash
# Nền TB bước 2, bản sửa: tăng w_tb 15 -> 150 để phần sửa nhỏ lại (lần trước |phần sửa| 11,3 mức xám, MS-SSIM rớt 98,95%).
# Chạy: bash scene_v1/chay_tbsua150.sh   (từ thư mục tai_lap; ~15 phút)
set -e
export SAGE_MAN=D:/CS406/DoAn/Sage/tai_lap/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
export SAGE_ANH=D:/CS406/DoAn/Sage/luu_tru/tim_gap_b0_0811/seg/anh
PY=D:/Miniconda3/envs/DoAn/python.exe
TL=D:/CS406/DoAn/Sage/tai_lap
LOG=$TL/scene_v1/tbsua_w150.log
ten=tbsua_w150
echo "[$(date +%H:%M)] train $ten"
cd $TL/cai_tien
$PY tb_sua.py --chi_train --ten $ten --w_tb 150 > $LOG 2>&1
echo "[$(date +%H:%M)] xuất ảnh"
cd $TL/b3
$PY xuat_anh.py --bien_the $ten >> $LOG 2>&1
echo "[$(date +%H:%M)] chỉ số ghép val 151"
cd $TL/cai_tien
$PY chi_so_ghep.py --bien_the sage tb tbcd nosam_kp841 tbsua_kp841 $ten --tap dev --hau kp_tbsua150 >> $LOG 2>&1
tail -n 12 $LOG
echo XONG
