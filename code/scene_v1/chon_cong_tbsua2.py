"""24/11 bước 2: cổng hẹp x = clip((chi_tiet_manh*(1-nóng) - a)/0,3) ở ảnh sáng. Chọn (sigma, a) trên nửa A train, kiểm trên nửa B.
LUẬT ĐẶT TRƯỚC: lấy ti_so (dich / pho_bien) lớn nhất trên A với điều kiện dich >= 0,25 và person <= 0,15; báo B nguyên trạng."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import MAN, TEN_LOP15, duong
from xuat_anh import cong_vnc
from chon_cong_tbsua import DICH, PHO_BIEN, nl
df = pd.read_csv(MAN / "train.csv", dtype=str).sort_values("image_id")
T, N = {}, {}
for i, r in enumerate(df.itertuples()):
    if i % 2: continue
    y = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if y.mean() < 100: continue
    ir = cv2.imread(str(duong(r.ir_path)), 0); lab = cv2.imread(str(duong(r.label_path)), 0)[::3, ::3].ravel()
    nong = cong_vnc(y, ir, g_min=0, toi_lo=-2, toi_hi=-1)
    nua = "AB"[i // 50 % 2]; N[nua] = N.get(nua, 0) + np.bincount(lab, minlength=15)[:15]
    for s in (4, 8):
        ev, ei = nl(y, s), nl(ir, s)
        m = ev / (ev + ei + 2) * np.clip(ev / 12, 0, 1) * (1 - nong)
        for a in (0, .1, .2, .3, .4, .5):
            x = np.clip((m - a) / .3, 0, 1)[::3, ::3].ravel()
            T[(s, a, nua)] = T.get((s, a, nua), 0) + np.bincount(lab, x, minlength=15)[:15]
R = []
for (s, a, u), t in T.items():
    m = t / np.maximum(N[u], 1)
    R.append(dict(nua=u, s=s, a=a, dich=m[DICH].mean(), pb=m[PHO_BIEN].mean(), ti_so=m[DICH].mean() / m[PHO_BIEN].mean(), **{TEN_LOP15[c]: m[c] for c in range(1, 15)}))
R = pd.DataFrame(R).sort_values(["nua", "s", "a"]); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.round(3).to_string(index=False))
A = R[(R.nua == "A") & (R.dich >= .25) & (R.person <= .15)]
print("CHỌN (theo luật):", A.loc[A.ti_so.idxmax(), ["s", "a", "dich", "pb", "ti_so"]].to_dict() if len(A) else "không bản nào đạt")
