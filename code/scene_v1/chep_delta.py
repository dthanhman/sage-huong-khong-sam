"""Chẩn đoán chưng cất (23/11): Student chép được bao nhiêu phần độ chỉnh của thầy?
D_t = ref − nen (thầy chỉnh), D_s = S − nen (Student lệch khỏi nền). Theo lớp:
  chep = Σ D_s·D_t / Σ D_t²  (1 = chép đủ, 0 = không chép)
  nhieu = RMS(D_s − proj) ở mọi điểm ảnh lớp đó (lệch không giải thích bằng thầy)
  rms_t = RMS(D_t) (thầy chỉnh mạnh cỡ nào)
Chạy: python chep_delta.py dev|train ref nen S1 [S2 ...]"""
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
import sys; sys.path.insert(0, f"{_TL}/b3"); sys.stdout.reconfigure(encoding="utf-8")
import cv2, numpy as np, pandas as pd
from chung import MAN, ANH, duong, thu_muc
TEN = ['unl','road','sidewalk','building','light','sign','veg','sky','person','car','truck','bus','moto','bike','pole']
lop = [1,2,3,4,5,6,7,8,9,10,11,14]
tap, ref, nen, *ss = sys.argv[1:]
df = pd.read_csv(MAN / f"{tap}.csv", dtype={"image_id": str})
if tap == "train": df = df.iloc[::4]
NET = {}
def chay_ck(ten, r):
    """'ck:<ten>' = chạy thẳng checkpoint Student b3/sage_student/<ten>.pt (không cần xuất ảnh)."""
    import torch
    if ten not in NET:
        sys.path.insert(0, f"{_TL}/b3/sage_student")
        from model import Network
        sd = torch.load(f"{_TL}/b3/sage_student/{ten}.pt", map_location="cpu", weights_only=True)
        if any(k.startswith("vung_") for k in sd):
            from model_vung import NetworkVung; net = NetworkVung(32, rong=sd["vung_dau.0.weight"].shape[0])
        elif any(k.startswith("mat_") for k in sd):
            dim = sd["module1.0.weight"].shape[0]
            from model_mat import NetworkMat; net = NetworkMat(dim)
        else:
            net = Network(sd["module1.0.weight"].shape[0])
        net.load_state_dict(sd, strict=True); NET[ten] = net.cuda().eval()
    v = cv2.imread(str(duong(r.vis_path)))
    y = cv2.cvtColor(v, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY)
    t2 = lambda z: torch.from_numpy(z).float().div(255)[None, None].cuda()
    with torch.inference_mode():
        return (NET[ten](t2(y), t2(ir))[0].clamp(0, 1) * 255).round().cpu().numpy()[0, 0].astype(float)
def Y(b, r):
    if b.startswith("ck:"):
        return chay_ck(b[3:], r)
    return cv2.cvtColor(cv2.imread(str(ANH / b / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
st = {s: np.zeros((15, 4)) for s in ss}   # Σ Ds·Dt, Σ Dt², Σ Ds², n
for r in df.itertuples():
    lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE).ravel()
    n0 = Y(nen, r); dt = (Y(ref, r) - n0).ravel()
    for s in ss:
        ds = (Y(s, r) - n0).ravel()
        st[s][:, 0] += np.bincount(lab, ds * dt, 15); st[s][:, 1] += np.bincount(lab, dt * dt, 15)
        st[s][:, 2] += np.bincount(lab, ds * ds, 15); st[s][:, 3] += np.bincount(lab, minlength=15)
print(f"tập {tap}, {len(df)} ảnh, thầy {ref}, nền {nen}")
for s in ss:
    a = st[s][lop]; chep = a[:, 0] / a[:, 1]
    nhieu = np.sqrt(np.maximum(a[:, 2] - chep ** 2 * a[:, 1], 0) / a[:, 3])
    print(f"\n== {s}")
    print(pd.DataFrame({"rms_thay": np.sqrt(a[:, 1] / a[:, 3]), "chep": chep, "nhieu_rms": nhieu},
                       index=[TEN[c] for c in lop]).T.round(2).to_string())
