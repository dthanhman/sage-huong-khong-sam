"""24/11: dựng 2 nhánh trên nền tbsua_992 (CHỈ dựng ảnh, không train B3), 992 train + Dev228.
  tbsua_vhn3 : Y = Yt + p_ng(R_ir − Yt) + p_vis(R_vis − Yt)  (ghep() của b3/vung_hai_nhanh.py, bản đồ p_thay_vhn3.npz nguồn chéo)
  tbsua_cvis : Y = Yt + x(Y_vis − Yt), x = X_MAX·(1 − tối)·clip((m − a)/0,3), m = VIS trội chi tiết × không nóng (chon_cong_tbsua2.py)
CrCb của VIS. Không nhãn, không SAM (vhn3 cần thầy hai nhánh lúc suy luận).
python scene_v1/dung_2_nhanh.py --s 4 --a 0.2 [--bt tbsua_vhn3 tbsua_cvis]"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, duong, thu_muc
from xuat_anh import cong_vnc
from vung_hai_nhanh import ghep
from chon_cong_tbsua import nl
NEN, X_MAX = "tbsua_992", 0.8            # 0,8 = 1 − g_min của vnc


def cong_hep(y, ir, s, a):
    nong = cong_vnc(y, ir, g_min=0, toi_lo=-2, toi_hi=-1)                      # chỉ cổng nhiệt
    toi = float(np.clip((110 - y.mean()) / 30, 0, 1))                           # cổng tối như vnc (80 -> 110)
    ev, ei = nl(y, s), nl(ir, s)
    m = ev / (ev + ei + 2) * np.clip(ev / 12, 0, 1) * (1 - nong)
    return X_MAX * (1 - toi) * np.clip((m - a) / .3, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, required=True); ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--bt", nargs="+", default=["tbsua_vhn3", "tbsua_cvis"])
    A = ap.parse_args()
    z = np.load(Path(__file__).resolve().parent / "_vhn3/p_thay_vhn3.npz")
    P, vt = z["p"], {k: i for i, k in enumerate(z["image_id"])}
    df = pd.concat([pd.read_csv(MAN / f"{t}.csv", dtype={"image_id": str}) for t in ("train", "dev")])
    assert len(df) == 1220 and df.image_id.is_unique
    tk = {b: [] for b in A.bt}
    for r in df.itertuples():
        ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]; yv = y.astype(np.float32)
        ir = cv2.imread(str(duong(r.ir_path)), 0)
        yt = cv2.cvtColor(cv2.imread(str(ANH / NEN / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        for bt in A.bt:
            if bt == "tbsua_vhn3":
                p = [cv2.resize(q.astype(np.float32) / 255, (800, 600), interpolation=cv2.INTER_LINEAR) for q in P[vt[r.image_id]]]
                yf = ghep(yt, yv, ir.astype(np.float32), p[0], p[1], False)
            else:
                yf = yt + cong_hep(y, ir, A.s, A.a) * (yv - yt)
            yf = yf.round().clip(0, 255).astype(np.uint8)
            out = ANH / bt / thu_muc(r) / f"{r.image_id}.png"; out.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(out), cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
            tk[bt].append((r.split, np.abs(yf - yt).mean(), (yf >= 250).mean() * 100, (yt >= 250).mean() * 100))
    for bt, v in tk.items():
        d = pd.DataFrame(v, columns=["split", "|dY|", "chay_%", "chay_nen_%"])
        print(bt, len(d)); print(d.groupby("split").mean().round(3).to_string())


if __name__ == "__main__":
    main()
