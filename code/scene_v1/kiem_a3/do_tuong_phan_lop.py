"""21/11: tương phản VẬT–VÀNH (7 px) theo từng lớp, và độ tách vùng đường–vỉa hè, trên ảnh TRAIN (không đụng val 151).
c = |TB vật − TB vành| trên kênh Y (VIS: Y của RGB; IR: xám; ghép: Y). Vật = thành phần liên thông >= 30 px.
Tách đường–vỉa hè: |TB road − TB sidewalk| / sqrt((var_r + var_s)/2), chỉ trong dải 20 px quanh biên.
"""
import sys
import cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from chung import MAN, duong, thu_muc, ANH

BT = ["sage", "tb", "nosam_kp841", "nosam_kp841_vung10", "nosam_kp841_vung10_tach1"]
LOP = {4: "traffic_light", 5: "traffic_sign", 8: "person", 9: "car", 10: "truck", 11: "bus", 14: "pole"}
VANH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
DAI = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
Yf = lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)

df = pd.read_csv(MAN / "train.csv", dtype={"image_id": str}).iloc[::2]
vat, tach = [], []
for r in df.itertuples():
    lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE)
    anh = {"vis": Yf(cv2.imread(str(duong(r.vis_path)))),
           "ir": cv2.imread(str(duong(r.ir_path)), cv2.IMREAD_GRAYSCALE).astype(np.float32)}
    for b in BT:
        anh[b] = Yf(cv2.imread(str(ANH / b / thu_muc(r) / f"{r.image_id}.png")))
    for k, ten in LOP.items():
        n, cc = cv2.connectedComponents((lab == k).astype(np.uint8))
        for j in range(1, n):
            o = (cc == j).astype(np.uint8)
            if o.sum() < 30:
                continue
            v = (cv2.dilate(o, VANH) > 0) & (lab != k)
            if v.sum() < 30:
                continue
            o = o > 0
            vat.append({"lop": ten, **{b: abs(a[o].mean() - a[v].mean()) for b, a in anh.items()}})
    R, S = (lab == 1).astype(np.uint8), (lab == 2).astype(np.uint8)
    dai = (cv2.dilate(R, DAI) > 0) & (cv2.dilate(S, DAI) > 0)
    rr, ss = dai & (R > 0), dai & (S > 0)
    if rr.sum() > 200 and ss.sum() > 200:
        tach.append({b: abs(a[rr].mean() - a[ss].mean()) / np.sqrt((a[rr].var() + a[ss].var()) / 2 + 1e-6)
                     for b, a in anh.items()})

d = pd.DataFrame(vat)
cot = ["vis", "ir"] + BT
t = d.groupby("lop")[cot].median().round(1)
t.insert(0, "so_vat", d.groupby("lop").size())
t["nguon_manh"] = np.where(t.vis > t.ir, "VIS", "IR")
pd.set_option("display.width", 250)
print("Tương phản vật–vành (trung vị):"); print(t.to_string())
s = pd.DataFrame(tach)
print(f"\nTách đường–vỉa hè ({len(s)} ảnh, trung vị):"); print(s.median().round(3).to_string())
