#!/bin/bash
# Chuỗi đêm 19/11 (~3,3 h), chạy trong Git Bash riêng: bash scene_v1/chay_dem_1911.sh  (từ tai_lap)
# 1) mIoU B0 val 151 cho nền B v3 (m11b_l15_e30_bn)            ~1,2 h
# 2) Nền A bản sửa BatchNorm (m11a_l5_e30_bn): train + ảnh + chỉ số ~0,9 h
# 3) mIoU B0 val 151 cho nền A                                   ~1,2 h
# Bỏ lọc (BO_LOC=1): lọc theo lệch TB cả ảnh che mất sửa cục bộ. Bước nào lỗi thì bước sau vẫn chạy.
# Chạy lại được: B0 có --tiep, train nền A bỏ qua nếu đã có checkpoint.
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M)] (1) mIoU nền B v3"
TEN=m11b_l15_e30_bn BO_LOC=1 bash scene_v1/chay_nenb_miou.sh > scene_v1/chay_nenb_miou_v3.out 2>&1
tail -n 4 scene_v1/chay_nenb_miou_v3.out
echo "[$(date +%H:%M)] (2) nền A sửa BN"
TEN=m11a_l5_e30_bn bash scene_v1/chay_nena_ce.sh > scene_v1/chay_nena_ce.out 2>&1
tail -n 6 scene_v1/chay_nena_ce.out
echo "[$(date +%H:%M)] (3) mIoU nền A"
TEN=m11a_l5_e30_bn GOC=nosam_kp841 BO_LOC=1 bash scene_v1/chay_nenb_miou.sh > scene_v1/chay_nena_miou.out 2>&1
tail -n 4 scene_v1/chay_nena_miou.out
echo "[$(date +%H:%M)] XONG CẢ CHUỖI"
