"""Chia 841 ảnh train của tập khám phá scene_v1 thành 2 nửa A/B cho thầy chéo (M1b).

Cách chia giống nua/ cũ: xếp theo id, cắt khối 50 ảnh liền nhau, khối chẵn -> A, khối lẻ -> B
(ảnh liền số là cùng cảnh, nên mỗi cảnh nằm trọn ở một nửa, trừ chỗ cắt khối).
dev.csv của mỗi nửa = val 151 của kham_pha (chỉ để train_seg ghi log, không dùng chọn gì).

Ra: scene_v1/nua_kp/chia_nua.csv, nua_kp/{A,B}/{train,dev}.csv
Chạy: python scene_v1/tao_nua_kp.py   (từ tai_lap)
"""
from pathlib import Path

import pandas as pd

DAY = Path(__file__).resolve().parent
KP = DAY / "kham_pha"
RA = DAY / "nua_kp"
KHOI = 50

tr = pd.read_csv(KP / "train.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
dv = pd.read_csv(KP / "dev.csv", dtype={"image_id": str})
assert len(tr) == 841 and len(dv) == 151 and not set(tr.image_id) & set(dv.image_id)

tr["nua"] = ["A" if (i // KHOI) % 2 == 0 else "B" for i in range(len(tr))]
for n in "AB":
    (RA / n).mkdir(parents=True, exist_ok=True)
    tr[tr.nua == n].drop(columns="nua").to_csv(RA / n / "train.csv", index=False)
    dv.to_csv(RA / n / "dev.csv", index=False)
tr[["image_id", "nua"]].to_csv(RA / "chia_nua.csv", index=False)

a, b = (tr.nua == "A").sum(), (tr.nua == "B").sum()
assert a + b == 841 and set(tr[tr.nua == "A"].image_id).isdisjoint(tr[tr.nua == "B"].image_id)
print(f"A {a} ảnh, B {b} ảnh, {-(-len(tr) // KHOI)} khối; ra {RA}")
