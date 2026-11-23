#!/bin/bash
# CHƯNG CẤT vhn VÀO STUDENT (22/11) – việc 3. Chạy song song với Kaggle sage-b0-vhn-s12 (vhn seed 1, 2).
# CÂU HỎI: Student 0,136M (không SAM, không hai_nhanh lúc chạy) học theo ảnh thầy vhn có giữ được lợi của vhn không?
#   vhn (Kaggle s0, 10 lớp val151): 57,77 vs nosam 57,19 / sage 56,90 / tb 56,98 – hơn chắc chắn cả SAGE và TB,
#   lợi chủ yếu ở person (+3,4). Nhưng vhn cần hai_nhanh lúc chạy -> không phải pp hợp lệ. Student cc_vhn thì hợp lệ.
# CÁCH: ảnh thầy vhn dựng từ ĐÚNG 2 checkpoint hai_nhanh nửa A/B của Kaggle đợt 9 (nhãn đoán CHÉO: ảnh train nửa A
#   dùng mô hình B và ngược lại; không mô hình nào thấy val151 hay Dev228 – đã kiểm id 22/11).
#   Student = cấu hình nosam_kp841 + --ref vhn (w_ref 4, w_ref_grad 3 như loss gốc SAGE), kiến trúc gốc, seed 0.
# ĐẶT TRƯỚC (sàng lọc, máy nhà, B0 s0, val151, 10 lớp >= 20 ảnh; mốc cùng máy nosam_kp841 56,99):
#   ĐI TIẾP (-> Kaggle B0 3 seed, so với sage/tb): cc_vhn − nosam_kp841 > 0.
#   Không qua: tìm nguyên nhân (do_he_so_ir bên dưới cho biết Student có kéo person về IR như vhn không).
#   Đủ 14 lớp chỉ báo kèm. do_he_so_ir chỉ để giải thích, không chặn.
# ~25 ph dựng ảnh + 25 ph Student + 1,3 h B0. Chạy lại được (bỏ bước đã xong).
# Chạy (Git Bash): bash scene_v1/chay_chung_cat.sh > scene_v1/chay_chung_cat.out 2>&1
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SAGE_ANH=${SAGE_ANH:-$TL/b3/anh}
export SAGE_RUNS=${SAGE_RUNS:-$TL/scene_v1/runs_b0}
PY=${PY:-python}
VHN=$TL/scene_v1/_vhn
TEN=cc_vhn

# 1) chia 2 nửa (tất định) + kiểm trùng bản Kaggle + chép checkpoint
cd $TL/b3
if [ $(ls $SAGE_ANH/vhn/train 2>/dev/null | wc -l) -lt 992 ]; then
  echo "[$(date +%H:%M)] dựng ảnh thầy vhn"
  $PY vung_hai_nhanh.py chia --ra $VHN
  $PY -c "import pandas as p,sys; a,b=(p.read_csv(f,dtype=str) for f in sys.argv[1:]); assert a.equals(b), 'nua.csv KHÁC Kaggle'; print('nua.csv khớp Kaggle')" \
    $VHN/nua.csv $TL/_kaggle_out/vhn/vhn/nua.csv
  for x in A B; do
    mkdir -p $VHN/runs_$x/hai_nhanh_b0_s0
    cp $TL/_kaggle_kp_code/vhn_ckpt/$x/{last.pt,config.json} $VHN/runs_$x/hai_nhanh_b0_s0/
  done
  md5sum $VHN/runs_*/hai_nhanh_b0_s0/last.pt
  $PY vung_hai_nhanh.py dung --ra $VHN
fi
echo "[$(date +%H:%M)] ảnh vhn: $(ls $SAGE_ANH/vhn/train | wc -l)"

# 2) Student chưng cất
cd $TL/e17
if [ ! -f $TL/b3/sage_student/$TEN.pt ]; then
  echo "[$(date +%H:%M)] train Student $TEN"
  $PY train_student_nosam.py --chi_train --ten $TEN --seed 0 --ref vhn > $TL/scene_v1/log_student_$TEN.txt 2>&1
fi
echo "[$(date +%H:%M)] Student xong: $(tail -1 $TL/scene_v1/log_student_$TEN.txt)"

# 3) xuất ảnh ghép + đo hệ số kéo về IR (giải thích)
cd $TL/b3
[ $(ls $SAGE_ANH/$TEN/train 2>/dev/null | wc -l) -ge 992 ] || $PY xuat_anh.py --bien_the $TEN
cd $TL/scene_v1
RA_CSV=$TL/scene_v1/he_so_ir_cc.csv $PY do_he_so_ir.py nosam_kp841 vhn $TEN sage || echo "do_he_so_ir lỗi (không chặn)"

# 4) B0 chấm s0
cd $TL/b3
run=$SAGE_RUNS/${TEN}_b0_s0
if [ ! -f $run/dev_ket_qua_last.json ]; then
  echo "[$(date +%H:%M)] train B0 $TEN seed 0"
  $PY train_seg.py --bien_the $TEN --model b0 --seed 0 --tiep >> $SAGE_RUNS/log_${TEN}_b0_s0.txt 2>&1
  $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${TEN}_b0_s0.txt 2>&1
fi
cp $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv $SAGE_RUNS/_bak.csv 2>/dev/null || true
$PY so_sanh_lop_du.py --bien_the $TEN nosam_kp841 --model b0 --tap dev --seeds 0 --min_anh 20 1
mv $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv $SAGE_RUNS/so_sanh_lop_du_dev_b0_cc.csv
[ -f $SAGE_RUNS/_bak.csv ] && mv $SAGE_RUNS/_bak.csv $SAGE_RUNS/so_sanh_lop_du_dev_b0.csv
echo "[$(date +%H:%M)] XONG"
