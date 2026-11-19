"""Đo |Y(a) - Y(b)| trung bình (mức xám 0..255) trên val 151 của kham_pha.
Dùng để biết CE có làm ảnh đổi thật không (so với ảnh nền khởi tạo).
Ví dụ: python do_lech_anh.py tbsua_w150 m11b_kp_l5_b01 tb
       -> in lệch của mọi cặp (cột đầu là ảnh gốc so sánh).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ANH = Path(os.environ.get("SAGE_ANH", Path(__file__).resolve().parents[1] / "b3" / "anh"))
DEV = Path(__file__).resolve().parent / "kham_pha" / "dev.csv"


def doc_y(bt, id_):
    bgr = cv2.imread(str(ANH / bt / "train" / f"{id_}.png"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)


def main():
    ten = sys.argv[1:]
    ids = pd.read_csv(DEV, dtype={"image_id": str}).image_id.tolist()
    for i in range(len(ten)):
        for j in range(i + 1, len(ten)):
            d = [np.abs(doc_y(ten[i], k) - doc_y(ten[j], k)).mean() for k in ids]
            print(f"|{ten[i]} - {ten[j]}| = {np.mean(d):.2f} mức xám (val {len(ids)} ảnh)")


if __name__ == "__main__":
    main()
