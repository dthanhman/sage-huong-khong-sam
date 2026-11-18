#!/bin/bash
# Phương án 1 (M11) trên tập khám phá scene_v1: bước 1 nosam_kp841 -> bước 2 m11_kp
set -e
TL=$(cd "$(dirname "$0")/.." && pwd)
export SAGE_MAN=$TL/scene_v1/kham_pha SO_TRAIN=841 PYTHONUTF8=1
PY=${PY:-python}
cd $TL/e17
$PY train_student_nosam.py --chi_train --ten nosam_kp841 > ../scene_v1/nosam_kp841.log 2>&1
cd ../cai_tien
$PY m11_hoc_chung.py --ten m11_kp --init nosam_kp841 --beta 1 --lam 1 > ../scene_v1/m11_kp.log 2>&1
echo XONG
