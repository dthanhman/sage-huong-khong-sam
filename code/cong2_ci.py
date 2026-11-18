"""Cổng 2 có khoảng tin cậy: tỉ số (TB biến thể / TB SAGE) cho SCD, MS-SSIM, Q_ABF trên Dev,
CI95 bootstrap theo cặp ảnh (cùng bộ ảnh cho mọi biến thể), bốc lại theo KHỐI 10 ảnh id liền nhau
(Dev chỉ có 5 khối liền lớn – quá ít để bootstrap; khung FMB liền số là cùng cảnh nên khối nhỏ giữ được tương quan gần).

Ví dụ: python cong2_ci.py chi_so_ghep/dev_seed_tung_anh.csv [file khác ...]
In: tỉ số %SAGE, CI95, và (nếu có >= 2 seed nosam) độ lệch do seed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CS = ["SCD", "MS_SSIM", "Q_ABF"]
B, SEED, CO_KHOI = 2000, 0, 10

kq = pd.concat([pd.read_csv(p, dtype={"image_id": str}) for p in sys.argv[1:]]).drop_duplicates(["bien_the", "image_id"])
ids = sorted(kq.image_id.unique())
bang = {c: kq.pivot(index="image_id", columns="bien_the", values=c).loc[ids] for c in CS}
so = np.array([int(i) for i in ids])
lien = np.r_[0, np.cumsum(np.diff(so) != 1)]             # khối liền lớn
khoi = np.zeros(len(ids), int)
k = 0
for g in np.unique(lien):                                 # cắt mỗi khối liền thành khối con 10 ảnh
    vt = np.flatnonzero(lien == g)
    for j in range(0, len(vt), CO_KHOI):
        khoi[vt[j:j + CO_KHOI]] = k
        k += 1
nhom = [np.flatnonzero(khoi == i) for i in range(k)]
rng = np.random.default_rng(SEED)
mau = [np.concatenate([nhom[i] for i in rng.integers(k, size=k)]) for _ in range(B)]
print(f"{len(ids)} ảnh, {k} khối {CO_KHOI} ảnh, B = {B}")

bt = [b for b in bang["SCD"].columns if b != "sage"]
for c in CS:
    X = bang[c]
    print(f"\n== {c}  (SAGE = {X['sage'].mean():.4f})")
    for b in bt:
        r = 100 * X[b].mean() / X["sage"].mean()
        boot = [100 * X[b].values[i].mean() / X["sage"].values[i].mean() for i in mau]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {b:10s} {r:6.1f}%  CI95 [{lo:5.1f}; {hi:5.1f}]")
    ns = [b for b in bt if b.startswith("nosam")]
    if len(ns) >= 2:
        v = np.array([X[b].mean() for b in ns])
        print(f"  -> {len(ns)} seed nosam: lệch max−min = {100 * (v.max() - v.min()) / X['sage'].mean():.2f}% SAGE, "
              f"SD = {100 * v.std(ddof=1) / X['sage'].mean():.2f}% SAGE")
