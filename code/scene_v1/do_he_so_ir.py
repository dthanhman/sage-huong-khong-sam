"""Đo Student thật sự kéo ảnh ghép về IR bao nhiêu theo từng lớp (val 151). (22/11, chép vào scene_v1 ngày 23/11, thêm S-mắt)
Chạy: python do_he_so_ir.py nosam_kp841 lint_nha lint_mat sage
a_lop = sum((F-Y)(IR-Y)) / sum((IR-Y)^2) trên điểm ảnh của lớp: 0 = giống VIS, 1 = giống IR.
Thêm: |F-Y| và |F-IR| trung bình (mức xám 0-255)."""
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
import sys, json
sys.path.insert(0, f"{_TL}/b3")
sys.stdout.reconfigure(encoding="utf-8")
import os
os.environ.setdefault("SAGE_MAN", f"{_TL}/scene_v1/kham_pha")
import cv2, numpy as np, pandas as pd, torch
from chung import MAN, STUDENT, duong
sys.path.insert(0, str(STUDENT))
from model import Network

TEN = ['unl', 'road', 'sidewalk', 'building', 'light', 'sign', 'veg', 'sky', 'person', 'car', 'truck', 'bus', 'moto', 'bicycle', 'pole']

def nap(ten):
    sd = torch.load(STUDENT / f"{ten}.pt", map_location="cpu", weights_only=True)
    sd.pop("_tb_scale", None)
    dim = sd["module1.0.weight"].shape[0]
    net = Network(dim)
    if any(k.startswith("mat_") for k in sd):
        from model_mat import NetworkMat
        net = NetworkMat(dim)
    if any(k.startswith("xa_") for k in sd):
        from model_xa import NetworkXa
        net = NetworkXa(dim, len({k.split(".")[1] for k in sd if k.startswith("xa_xuong.")}))
    net.load_state_dict(sd, strict=True)
    return net.cuda().eval()

bts = sys.argv[1:]
df = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str})
print("số ảnh:", len(df))
nets = {b: nap(b if b != "sage" else "checkpoints") for b in bts}
S = {b: np.zeros((15, 4)) for b in bts}   # sum (F-Y)(IR-Y), sum (IR-Y)^2, sum|F-Y|, sum|F-IR|
N = np.zeros(15)
t2 = lambda z: torch.from_numpy(z).float().div(255)[None, None].cuda()
for r in df.itertuples():
    vis = cv2.imread(str(duong(r.vis_path)))
    ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE)
    yf = y.astype(np.float64)
    N += np.bincount(lab.ravel(), minlength=15)
    for b, net in nets.items():
        with torch.inference_mode():
            f = (net(t2(y), t2(ir.astype(np.uint8)))[0].clamp(0, 1) * 255).round().cpu().numpy()[0, 0]
        for k, v in enumerate([(f - yf) * (ir - yf), (ir - yf) ** 2, np.abs(f - yf), np.abs(f - ir)]):
            S[b][:, k] += np.bincount(lab.ravel(), weights=v.ravel(), minlength=15)
w = json.load(open(f"{_TL}/scene_v1/w_lop.json"))
lop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
print("\nHệ số kéo về IR a (0 = VIS, 1 = IR); dòng w_lint = mục tiêu loss lint")
bang = {b: [S[b][c, 0] / S[b][c, 1] for c in lop] for b in bts}
bang["w_lint"] = [w[str(c)] for c in lop]
if os.environ.get("RA_CSV"):
    pd.DataFrame(bang, index=[TEN[c] for c in lop]).T.to_csv(os.environ["RA_CSV"])
print(pd.DataFrame(bang, index=[TEN[c] for c in lop]).T.round(2).to_string())
print("\n|F - Y_vis| trung bình (mức xám)")
print(pd.DataFrame({b: [S[b][c, 2] / N[c] for c in lop] for b in bts}, index=[TEN[c] for c in lop]).T.round(1).to_string())
print("\n|F - IR| trung bình (mức xám)")
print(pd.DataFrame({b: [S[b][c, 3] / N[c] for c in lop] for b in bts}, index=[TEN[c] for c in lop]).T.round(1).to_string())
