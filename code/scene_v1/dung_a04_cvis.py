"""26/11: a04_cvis = a04 + cổng VIS. Y = Ya04 + x(Y_vis − Ya04), x = cong_hep(y, ir, s=8, a=0,2) của dung_2_nhanh.py
(tham số đã chọn trên Train 24/11 cho tbsua_cvis, KHÔNG dò lại). CrCb của VIS. 992 train + Dev228.
SAGE_MAN=scene_v1/goc SAGE_ANH=<thư mục ảnh ghép> python scene_v1/dung_a04_cvis.py"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, duong, thu_muc
from dung_2_nhanh import cong_hep
NEN, TEN, S, A = "tbsua_a04", "a04_cvis", 8.0, 0.2
df = pd.concat([pd.read_csv(MAN / f"{t}.csv", dtype={"image_id": str}) for t in ("train", "dev")])
assert len(df) == 1220 and df.image_id.is_unique
tk = []
for r in df.itertuples():
    ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]; yv = y.astype(np.float32); ir = cv2.imread(str(duong(r.ir_path)), 0)
    yt = cv2.cvtColor(cv2.imread(str(ANH / NEN / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    x = cong_hep(y, ir, S, A)
    yf = (yt + x * (yv - yt)).round().clip(0, 255).astype(np.uint8)
    out = ANH / TEN / thu_muc(r) / f"{r.image_id}.png"; out.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(out), cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
    tk.append((r.split, np.abs(yf - yt).mean(), (x > 0).mean() * 100, (yf >= 250).mean() * 100))
d = pd.DataFrame(tk, columns=["split", "|dY|", "mo_cong_%", "chay_%"])
print(TEN, len(d)); print(d.groupby("split").mean().round(3).to_string())
