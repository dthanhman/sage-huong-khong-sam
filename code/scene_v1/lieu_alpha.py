"""24/11 tối: Y = TB + a*(tbsua - TB). Rào (MS-SSIM>=99, Q_ABF>=98 % SAGE) theo a, CHỈ trên Train (1/8 ảnh). Không train, không Dev/Test."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
sys.path[:0] = [f"{_TL}/b3", f"{_TL}/cai_tien"]
import numpy as np, pandas as pd, cv2
from chi_so_ghep import fusion_ms_ssim, q_abf, sum_of_correlation_differences, ANH, MAN, duong, thu_muc
from concurrent.futures import ProcessPoolExecutor
A = [0, .25, .4, .5, .6, .75, 1]
def doc(p): return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2YCrCb)[..., 0]
def lam(r):
    y, ir = doc(duong(r[0])), cv2.imread(str(duong(r[1])), 0)
    g = lambda b: doc(ANH / b / r[2] / f"{r[3]}.png")
    t = (y.astype(np.float32) + ir) / 2; u = g("tbsua_992").astype(np.float32); o = []
    for n, f in [("sage", g("sage"))] + [(str(a), (t + a * (u - t)).round().clip(0, 255).astype(np.uint8)) for a in A]:
        hp = f.astype(np.float32); hp = (hp - cv2.GaussianBlur(hp, (0, 0), 2)).std()
        o.append((n, fusion_ms_ssim(f, ir, y, aggregation="sum")["MS_SSIM"], q_abf(f, ir, y), sum_of_correlation_differences(f, ir, y), f.astype(np.float32).std(), hp))
    return o
if __name__ == "__main__":
    df = pd.read_csv(MAN / "train.csv", dtype={"image_id": str}).iloc[::8]
    with ProcessPoolExecutor(3) as ex: R = [x for o in ex.map(lam, [(r.vis_path, r.ir_path, thu_muc(r), r.image_id) for r in df.itertuples()]) for x in o]
    d = pd.DataFrame(R, columns=["a", "MS", "Q", "SCD", "SD", "cao_tan"]).groupby("a", sort=False).mean()
    for c in ("MS", "Q", "SCD", "SD"): d[c + "%"] = 100 * d[c] / d.loc["sage", c]
    print(len(df), "ảnh Train"); print(d.round(3).to_string())
