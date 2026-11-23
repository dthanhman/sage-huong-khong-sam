"""Đo tương phản vật–vành trên ảnh ghép (val 151) – chẩn đoán cho A2 (20/11).

Với mỗi người (nhãn 8, cả lớp trong ảnh, >= 200 px): c = |TB_Y(người) - TB_Y(vành 15 px)|, và tỉ lệ c / std(Y cả ảnh)
(> 1 = làm nổi người CÓ CHỌN LỌC). Báo trung vị trên các ảnh có người.
Nhãn val 151 chỉ dùng để ĐO (phân tích), không đưa vào mô hình.

Chạy: cd b3 && SAGE_MAN=../scene_v1/kham_pha SAGE_ANH=.../seg/anh python ../scene_v1/do_tuong_phan.py --bien_the sage tb nosam_kp841 ...
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, duong  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", nargs="+", required=True)
    ap.add_argument("--lop", type=int, default=8, help="nhãn cần đo (8 người; 4 đèn – S-đèn 21/11)")
    ap.add_argument("--px_min", type=int, default=200, help="số px tối thiểu của lớp trong ảnh (đèn nhỏ: dùng 30)")
    a = ap.parse_args()
    man = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str}).set_index("image_id")
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    rows = []
    for i in man.index:
        lab = cv2.imread(str(duong(man.loc[i, "label_path"])), cv2.IMREAD_GRAYSCALE)
        mn = (lab == a.lop).astype(np.uint8)
        if mn.sum() < a.px_min:
            continue
        vanh = (cv2.dilate(mn, ker) > 0) & (mn == 0) & (lab != 0)
        if vanh.sum() < min(100, a.px_min):
            continue
        for b in a.bien_the:
            y = cv2.cvtColor(cv2.imread(str(ANH / b / "train" / f"{i}.png")), cv2.COLOR_BGR2YCrCb)[..., 0].astype(float)
            c = abs(y[mn > 0].mean() - y[vanh].mean())
            rows.append({"id": i, "bien_the": b, "tuong_phan": c, "ti_le": c / y.std()})
    d = pd.DataFrame(rows).groupby("bien_the")[["tuong_phan", "ti_le"]].median().loc[a.bien_the]
    print(f"{len(rows) // len(a.bien_the)} ảnh có lớp {a.lop} – trung vị:")
    print(d.round(2).to_string())


if __name__ == "__main__":
    main()
