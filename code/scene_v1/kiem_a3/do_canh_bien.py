"""Kiểm nhanh trước A3 (21/11): ở biên giữa 2 lớp trong nhãn, cạnh (Sobel) của ảnh ghép có thấp hơn
cạnh mạnh hơn của 2 nguồn không? Chỉ dùng ảnh TRAIN 841 (không đụng val 151).
Ra: trung vị từng ảnh của |Sobel Y| trên pixel biên, theo cặp lớp, cho vis/ir/max(vis,ir) và các bản ghép.
Chạy: cd tai_lap/b3 && SAGE_MAN=.../scene_v1/kham_pha python ../scene_v1/kiem_a3/do_canh_bien.py
"""
import sys
import cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from chung import MAN, duong, thu_muc, ANH

BT = ["sage", "tb", "nosam_kp841", "nosam_kp841_vung10"]
CAP = {"road-sidewalk": (1, 2), "sidewalk-building": (2, 3), "sign-sky": (5, 7), "sign-building": (5, 3),
       "person-road": (8, 1), "person-sidewalk": (8, 2), "car-road": (9, 1), "pole-sky": (14, 7)}
K = np.ones((3, 3), np.uint8)


def sob(y):
    y = y.astype(np.float32)
    return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))


def Y(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]


df = pd.read_csv(MAN / "train.csv", dtype={"image_id": str})
df = df.iloc[::4]   # ~210 ảnh rải đều
rows = []
for r in df.itertuples():
    lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE)
    g = {"vis": sob(Y(cv2.imread(str(duong(r.vis_path))))),
         "ir": sob(cv2.imread(str(duong(r.ir_path)), cv2.IMREAD_GRAYSCALE))}
    g["max_nguon"] = np.maximum(g["vis"], g["ir"])
    for b in BT:
        g[b] = sob(Y(cv2.imread(str(ANH / b / thu_muc(r) / f"{r.image_id}.png"))))
    for ten, (a, c) in CAP.items():
        bien = cv2.dilate((lab == a).astype(np.uint8), K) & cv2.dilate((lab == c).astype(np.uint8), K)
        if bien.sum() < 30:
            continue
        m = bien.astype(bool)
        rows.append({"cap": ten, "id": r.image_id, "n_px": int(m.sum()), **{k: float(np.median(v[m])) for k, v in g.items()}})

d = pd.DataFrame(rows)
cot = ["vis", "ir", "max_nguon"] + BT
bang = d.groupby("cap")[cot].median().round(1)
bang.insert(0, "so_anh", d.groupby("cap").size())
for b in BT:
    bang[f"%{b}/max"] = (100 * d.assign(t=d[b] / d.max_nguon).groupby("cap").t.median()).round(0)
pd.set_option("display.width", 250)
print(bang.to_string())
d.to_csv(Path(__file__).with_name("canh_bien_train.csv"), index=False)
