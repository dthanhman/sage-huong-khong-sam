"""24/11: chọn tín hiệu (không nhãn) cho cổng VIS hẹp trên nền tbsua_992. CHỈ ảnh TRAIN 992, nhãn chỉ để đo, không vào phương pháp.
Mỗi tín hiệu x in [0,1] = "nên theo VIS". In trung bình theo lớp: muốn cao ở sidewalk/light/truck/bus, thấp ở 8 lớp phổ biến.
Chạy (từ tai_lap): SAGE_MAN=... python scene_v1/chon_cong_tbsua.py"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import MAN, TEN_LOP15, duong
from xuat_anh import cong_vnc
cv2.setNumThreads(4)
DICH = [2, 4, 10, 11]
PHO_BIEN = [1, 3, 5, 6, 7, 8, 9, 14]

def nl(a, s=4):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1)
    g = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)) / 8
    return cv2.GaussianBlur(g, (0, 0), s)

def tin_hieu(vis, ir):
    ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb).astype(np.float32); y = ycc[:, :, 0]
    ev, ei = nl(y), nl(ir)
    chroma = cv2.GaussianBlur(np.hypot(ycc[:, :, 1] - 128, ycc[:, :, 2] - 128), (0, 0), 4)
    hot = cv2.morphologyEx(cv2.GaussianBlur(ir, (0, 0), 2), cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    nong = np.clip((hot.astype(np.float32) - 15) / 30, 0, 1)
    tr = ev / (ev + ei + 2)
    return {"vnc(1-g)": 1 - cong_vnc(y.astype(np.uint8), ir, g_min=0), "troi_chi_tiet": tr,
            "chi_tiet_manh": tr * np.clip(ev / 12, 0, 1), "mau": np.clip(chroma / 40, 0, 1),
            "mau_khong_nong": np.clip(chroma / 40, 0, 1) * (1 - nong),
            "lam": np.clip((cv2.GaussianBlur(ir, (0, 0), 15).astype(np.float32) * -1 + cv2.GaussianBlur(y, (0, 0), 15)) / 80, 0, 1)}

def main():
    df = pd.read_csv(MAN / "train.csv", dtype=str).sort_values("image_id")
    T, N = {}, {}
    for i, r in enumerate(df.itertuples()):
        vis = cv2.imread(str(duong(r.vis_path))); ir = cv2.imread(str(duong(r.ir_path)), 0)
        lab = cv2.imread(str(duong(r.label_path)), 0)
        n = np.bincount(lab.ravel(), minlength=15)[:15]
        nua = "AB"[i // 50 % 2]
        sang = "sang" if cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0].mean() >= 100 else "toi"
        for k, x in tin_hieu(vis, ir).items():
            K = (k, nua, sang)
            T[K] = T.get(K, 0) + np.bincount(lab.ravel(), x.ravel(), minlength=15)[:15]
            N[K] = N.get(K, 0) + n
    R = []
    for nhom, sang in (("A", "sang"), ("B", "sang"), ("AB", "sang"), ("AB", "toi")):
        for k in dict.fromkeys(K[0] for K in T):
            m = sum(T[(k, u, sang)] for u in nhom) / np.maximum(1, sum(N[(k, u, sang)] for u in nhom))
            R.append({"nua": nhom, "nhom": sang, "tin_hieu": k, "dich": m[DICH].mean(), "pho_bien": m[PHO_BIEN].mean(),
                      "ti_so": m[DICH].mean() / m[PHO_BIEN].mean(), **{TEN_LOP15[c]: m[c] for c in range(1, 15)}})
    R = pd.DataFrame(R); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(R.round(2).to_string(index=False))

main() if __name__ == "__main__" else None
