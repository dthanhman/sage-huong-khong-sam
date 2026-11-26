"""26/11: kiểm nhanh a04_cvis thua a04 có do cổng VIS? Mức cổng mỗi ảnh = so ảnh dựng a04_cvis với a04 (kênh Y).
(từ code/) SAGE_MAN=scene_v1/goc SAGE_ANH=<thư mục ảnh ghép> python scene_v1/kiem_cong_cvis.py"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, duong, thu_muc
G = Path(__file__).resolve().parents[2] / "ket_qua/dev228_b3/runs"
T = ["0","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
S = (0, 1)
def nap(b, s):
    z = np.load(G / f"{b}_b3_s{s}_15lop/dev_nham_last.npz"); a = z["image_id"].astype(int); o = np.argsort(a); return a[o], z["conf"][o].astype(float)
def iou(c):
    c = c.copy(); c[0] = 0; u = c.sum(0) + c.sum(1) - np.diag(c)
    return np.where(c.sum(1) > 0, 100 * np.diag(c) / np.maximum(u, 1), np.nan)
m13 = lambda c: np.nanmean(iou(c)[1:])
D = {(b, s): nap(b, s) for b in ("a04_cvis", "tbsua_a04") for s in S}
ids = D["a04_cvis", 0][0]; assert all((v[0] == ids).all() for v in D.values()) and len(ids) == 228
df = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str}); df = df.set_index(df.image_id.astype(int)).loc[ids]
mo, dy, lab_mo = [], [], np.zeros((15, 2))       # lab_mo[c] = [pixel lớp c trong vùng cổng mở, tổng pixel lớp c]
for r in df.itertuples():
    yc = cv2.cvtColor(cv2.imread(str(ANH / "a04_cvis" / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    ya = cv2.cvtColor(cv2.imread(str(ANH / "tbsua_a04" / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    d = np.abs(yc - ya); m = d >= 1; mo.append(m.mean() * 100); dy.append(d.mean())
    L = cv2.imread(str(duong(r.label_path)), 0)
    lab_mo[:, 0] += np.bincount(L[m], minlength=15); lab_mo[:, 1] += np.bincount(L.ravel(), minlength=15)
mo, dy = np.array(mo), np.array(dy)
# chênh từng ảnh: độ chính xác pixel (bỏ lớp 0), TB 2 seed
acc = lambda c: np.array([np.trace(x[1:, 1:]) / max(x[1:].sum(), 1) for x in c])
dacc = np.mean([acc(D["a04_cvis", s][1]) - acc(D["tbsua_a04", s][1]) for s in S], 0) * 100
print(f"Dev228: cổng mở TB {mo.mean():.1f}% (trung vị {np.median(mo):.1f}, max {mo.max():.1f}); |dY| TB {dy.mean():.2f}; ảnh cổng đóng hẳn (<0,5%): {(mo<0.5).sum()}")
r, p = spearmanr(mo, dacc); print(f"Spearman (%cổng mở, chênh acc pixel cvis−a04): r={r:+.3f} p={p:.3f}")
r, p = spearmanr(dy, dacc); print(f"Spearman (|dY|, chênh acc pixel): r={r:+.3f} p={p:.3f}")
print("\nmIoU13 cvis−a04 theo nhóm mức cổng (cộng ma trận trong nhóm, TB 2 seed):")
q = np.quantile(mo, [0, 1/3, 2/3, 1]); nh = np.digitize(mo, q[1:-1])
for k, ten in enumerate(["cổng ít", "cổng vừa", "cổng nhiều"]):
    i = nh == k; h = [m13(D["a04_cvis", s][1][i].sum(0)) - m13(D["tbsua_a04", s][1][i].sum(0)) for s in S]
    print(f"  {ten:10s} {i.sum():3d} ảnh, mở {mo[i].min():5.1f}–{mo[i].max():5.1f}%: {h[0]:+.2f} / {h[1]:+.2f} TB {np.mean(h):+.2f}")
print("\nBỏ dần ảnh cổng mở nhiều nhất → mIoU13 cvis−a04 (TB 2 seed):")
o = np.argsort(-mo)
for k in (0, 20, 50, 100, 150):
    i = np.ones(228, bool); i[o[:k]] = False
    print(f"  bỏ {k:3d} ảnh: {np.mean([m13(D['a04_cvis', s][1][i].sum(0)) - m13(D['tbsua_a04', s][1][i].sum(0)) for s in S]):+.2f}")
print("\nTừng lớp: % pixel nằm trong vùng cổng mở | IoU cvis−a04 s0/s1:")
V = {k: iou(v[1].sum(0)) for k, v in D.items()}
for c in range(1, 15):
    if c == 13: continue
    print(f"  {T[c]:9s} {100*lab_mo[c,0]/max(lab_mo[c,1],1):5.1f}%  | {V['a04_cvis',0][c]-V['tbsua_a04',0][c]:+5.1f} / {V['a04_cvis',1][c]-V['tbsua_a04',1][c]:+5.1f}")
