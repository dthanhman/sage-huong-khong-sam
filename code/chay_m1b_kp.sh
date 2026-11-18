#!/bin/bash
# M1b trên tập khám phá scene_v1 (train 841 / val 151). Bản cũ (split 1.045/175): cai_tien/chay_m1b.sh.
# M1b = fine-tune Student SAGE (--init sage) bằng loss nội dung + lam·CE, sửa 3 lỗi của M1:
#   (1) thầy thuộc lòng -> thầy chéo: B0 học nửa A chấm ảnh nửa B và ngược lại (nua_kp/, khối 50 ảnh)
#   (2) proxy thiên vị  -> bỏ proxy (--proxy none; B0 proxy cũ đã học ảnh val 151), chấm bằng B0 MỚI ở bước 3
#   (4) KHÔNG đổi loss nội dung nữa: giữ 60/40 gốc SAGE (đổi sang 50/50 chỉ kéo ảnh về TB -> lặp vòng),
#       thêm bản đối chứng lam=0 để tách riêng tác dụng của CE.
#   Chưa sửa (3) thầy đóng băng (M11 học chung mới sửa).
# Lưu ý: KHỞI TẠO TỪ SAGE -> kế thừa SAM (không "SAM-free"); ghi rõ khi báo.
#
# Chạy TỪNG BƯỚC từ tai_lap (phiên phải mở suốt; chạy lại được, run đã có thì bỏ qua/train tiếp):
#   bash scene_v1/chay_m1b_kp.sh thay              # 1. 2 thầy chéo B0 trên ảnh sage (~1,2 h) + kiểm thầy
#   bash scene_v1/chay_m1b_kp.sh m1b 1             # 2a. M1b lam=1, loss nội dung gốc 60/40 (~30 phút) -> m1b_kp_l1.pt
#   bash scene_v1/chay_m1b_kp.sh m1b 0             # 2b. ĐỐI CHỨNG lam=0 (không CE, không cần thầy)  -> m1b_kp_l0.pt
#   bash scene_v1/chay_m1b_kp.sh danh_gia m1b_kp_l1 0   # 3. xuất ảnh + B0 mới seed 0, chấm val 151 (~1,2 h); làm cả m1b_kp_l0
# So sánh cần mốc tb/sage B0 cùng cách chấm trong scene_v1/runs_b0 (từ chay_b0_sangloc.sh).
# Tiêu chí đặt trước: (a) CE có ích: m1b_kp_l1 − m1b_kp_l0 > 0 ở cả 2 seed;
#                     (b) đạt mục tiêu: m1b_kp_l1 − max(tb, sage) ≥ +1,0 ở cả 2 seed.
set -e
TL=D:/CS406/DoAn/Sage/tai_lap
SV=$TL/scene_v1
PY=D:/Miniconda3/envs/DoAn/python.exe
export PYTHONUTF8=1 SAGE_ANH=D:/CS406/DoAn/Sage/luu_tru/tim_gap_b0_0811/seg/anh
export SAGE_NUA=$SV/nua_kp SAGE_THAY_RUNS=$SV

case "$1" in
  thay)
    [ -f $SV/nua_kp/chia_nua.csv ] || $PY $SV/tao_nua_kp.py
    cd $TL/b3
    for N in A B; do
      run=$SV/runs_b0_$N/sage_b0_s0
      if [ ! -f $run/last.pt ]; then
        echo "[$(date +%H:%M)] thầy nửa $N"
        SAGE_MAN=$SV/nua_kp/$N SAGE_RUNS=$SV/runs_b0_$N \
          $PY train_seg.py --bien_the sage --model b0 --seed 0 --tiep > $SV/log_thay_$N.txt 2>&1
      fi
    done
    # Kiểm thầy chéo có tín hiệu: CE trên nửa CHƯA thấy phải cao hơn rõ nửa đã học.
    # train_student_seg ghi kiem_thay.json vào cai_tien/ -> giữ bản cũ (thầy TB, split cũ), chuyển bản mới sang scene_v1.
    cd $TL/cai_tien
    [ -f kiem_thay.json ] && cp kiem_thay.json kiem_thay_cu_bak.json
    SAGE_MAN=$SV/kham_pha SO_TRAIN=841 $PY train_student_seg.py --kiem_thay --thay_bt sage --proxy none
    mv kiem_thay.json $SV/kiem_thay_sage_kp.json
    [ -f kiem_thay_cu_bak.json ] && mv kiem_thay_cu_bak.json kiem_thay.json
    cat $SV/kiem_thay_sage_kp.json
    ;;
  m1b)
    # Mặc định giữ loss nội dung GỐC SAGE (w_ir 0.4 = 60/40) để chỉ CE là khác -> đo đúng tác dụng của CE.
    # Đối chứng: bash scene_v1/chay_m1b_kp.sh m1b 0   (cùng mọi thứ, không CE -> m1b_kp_l0)
    L=${2:-1}
    W=${3:-0.4}
    TEN=m1b_kp_l$L; [ "$W" = "0.4" ] || TEN=${TEN}_w$W
    THAY="--hai_nua --thay_bt sage"; [ "$L" = "0" ] && THAY=""
    cd $TL/cai_tien
    SAGE_MAN=$SV/kham_pha SO_TRAIN=841 $PY train_student_seg.py --ten $TEN --dau sage --init sage \
        --w_ir $W --lam $L $THAY --proxy none --epochs 20 2>&1 \
      | grep --line-buffered -E "epoch|Lưu|Thầy|Error|error|Traceback" | tee $SV/$TEN.log
    ;;
  danh_gia)
    TEN=${2:?cần tên checkpoint, vd m1b_kp_l1}
    S=${3:-0}
    export SAGE_MAN=$SV/kham_pha SO_TRAIN=841 SAGE_RUNS=$SV/runs_b0
    cd $TL/b3
    [ -d $SAGE_ANH/$TEN ] || $PY xuat_anh.py --bien_the $TEN
    run=$SAGE_RUNS/${TEN}_b0_s$S
    [ -f $run/last.pt ] || $PY train_seg.py --bien_the $TEN --model b0 --seed $S --tiep > $SAGE_RUNS/log_${TEN}_b0_s$S.txt 2>&1
    [ -f $run/dev_nham_last.npz ] || $PY eval_seg.py --run $run --ckpt last --tap dev >> $SAGE_RUNS/log_${TEN}_b0_s$S.txt 2>&1
    if [ -d $SAGE_RUNS/tb_b0_s$S ] && [ -d $SAGE_RUNS/sage_b0_s$S ]; then
      $PY so_sanh_cap.py --bien_the tb sage $TEN --model b0 --tap dev --cap $TEN-tb $TEN-sage tb-sage
    else
      echo "Chưa có mốc tb/sage seed $S trong $SAGE_RUNS -> chạy chay_b0_sangloc.sh trước để so."
    fi
    if [ "$TEN" != "m1b_kp_l0" ] && [ -f $SAGE_RUNS/m1b_kp_l0_b0_s$S/dev_nham_last.npz ]; then
      $PY so_sanh_cap.py --bien_the m1b_kp_l0 $TEN --model b0 --tap dev --cap $TEN-m1b_kp_l0   # tác dụng riêng của CE
    fi
    ;;
  *)
    sed -n '2,16p' "$0"
    ;;
esac
