"""25/11: Dev228 (và val151) giống Train992 tới đâu: độ sáng VIS, IR, tương phản, lớp. CPU, một luồng. KHÔNG đụng Test, không dùng để chọn phương pháp."""
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
import sys; sys.path.insert(0, f"{_TL}/b3")
import cv2, numpy as np, pandas as pd
from pathlib import Path
from chung import duong
cv2.setNumThreads(1)
B = Path(f"{_TL}/scene_v1/goc"); KP = Path(f"{_TL}/scene_v1/kham_pha/dev.csv")
T = ["0","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
rd = lambda p: pd.read_csv(p, dtype={"image_id": str}).sort_values("image_id")
v = set(rd(KP).image_id); R = []
for t in ("train", "dev"):
    for r in rd(B/f"{t}.csv").itertuples():
        y = cv2.imread(str(duong(r.vis_path)), cv2.IMREAD_REDUCED_GRAYSCALE_4); i = cv2.imread(str(duong(r.ir_path)), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        l = cv2.imread(str(duong(r.label_path)), -1); l = l[..., 0] if l.ndim == 3 else l
        R.append(dict(tap=t, id=int(r.image_id), val=r.image_id in v, y=y.mean(), ysd=y.std(), ir=i.mean(), irsd=i.std(), **{c: n for c, n in zip(T, np.bincount(l.ravel(), minlength=15)[:15])}))
d = pd.DataFrame(R); d.to_csv("scene_v1/dai_dien_dev.csv", index=False)
a, b = d[d.tap == "train"], d[d.tap == "dev"]; assert (len(a), len(b), d.val.sum()) == (992, 228, 151)
S = {"Train992": a, "Dev228": b, "val151": d[d.val]}
q = np.quantile(a.y, [1/3, 2/3]); print("ngưỡng tam phân vị sáng theo Train", q.round(1))
for n, x in S.items():
    g = np.histogram(x.y, [-1, *q, 999])[0]/len(x)*100
    print(f"{n:9s} tối/vừa/sáng % {g.round(0)} | VIS<60: {100*(x.y<60).mean():.0f}% >=100: {100*(x.y>=100).mean():.0f}% | TB VIS {x.y.mean():.0f} IR {x.ir.mean():.0f} sdVIS {x.ysd.mean():.0f} sdIR {x.irsd.mean():.0f}")
print("\nlớp: % ảnh có >100 px | % pixel (bỏ nhãn 0)")
print(f"{'':10s}"+"".join(f"{n:>19s}" for n in S))
for c in T[1:]:
    print(f"{c:10s}"+"".join(f"{100*(x[c]>100).mean():9.1f}{100*x[c].sum()/x[T[1:]].values.sum():10.2f}" for x in S.values()), f"  Dev có {(b[c]>100).sum()} ảnh, {b.loc[b[c]>100,'id'].diff().gt(5).sum()+1 if (b[c]>100).any() else 0} cụm")
k = (b.id.diff() > 5).cumsum()
print("\ncụm Dev: id, n, VIS TB"); print(b.groupby(k.values).agg(tu=("id","min"), den=("id","max"), n=("id","size"), y=("y","mean"), ir=("ir","mean"), truck=("truck", lambda x:(x>100).sum()), bus=("bus", lambda x:(x>100).sum()), moto=("moto", lambda x:(x>100).sum()), sign=("sign", lambda x:(x>100).sum())).round(0).to_string())
