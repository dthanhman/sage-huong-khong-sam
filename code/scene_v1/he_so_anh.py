"""a theo lớp tính trực tiếp từ ẢNH đã xuất (dùng cho cả biến thể không phải Student như vhn). 23/11.
Chạy: python he_so_anh.py dev|train <bien_the...>   (train: lấy 1/4 số ảnh)"""
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
import sys; sys.path.insert(0, f"{_TL}/b3"); sys.stdout.reconfigure(encoding="utf-8")
import cv2, numpy as np, pandas as pd
from chung import MAN, ANH, duong, thu_muc
TEN = ['unl','road','sidewalk','building','light','sign','veg','sky','person','car','truck','bus','moto','bike','pole']
lop = [1,2,3,4,5,6,7,8,9,10,11,14]
bts = sys.argv[2:]; tap = sys.argv[1]
df = pd.read_csv(MAN / f"{tap}.csv", dtype={"image_id": str})
if tap == "train": df = df.iloc[::4]
S = {b: np.zeros((15, 2)) for b in bts}
for r in df.itertuples():
    y = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY).astype(float)
    lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE).ravel()
    for b in bts:
        f = cv2.cvtColor(cv2.imread(str(ANH / b / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
        S[b][:, 0] += np.bincount(lab, ((f - y) * (ir - y)).ravel(), 15); S[b][:, 1] += np.bincount(lab, ((ir - y) ** 2).ravel(), 15)
print(f"tập {tap}, {len(df)} ảnh")
print(pd.DataFrame({b: [S[b][c, 0] / S[b][c, 1] for c in lop] for b in bts}, index=[TEN[c] for c in lop]).T.round(2).to_string())
