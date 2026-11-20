"""Phân tích nosam thua ở đâu so với SAGE / TB (20/11). Chỉ đọc kết quả đã có, không train.

Dữ liệu: val 151 (scene_v1/kham_pha/dev.csv), B0, ma trận nhầm lẫn từng ảnh trong runs_b0_gop.
Mỗi biến thể lấy trung bình các seed đang có (nosam/sage: 0,1,2; tb: 0,1).

1. mIoU từng ảnh (10 lớp >= 20 ảnh, chỉ lớp có trong GT ảnh đó) -> ảnh nosam thua/hơn nhiều nhất,
   tương quan hiệu với độ sáng VIS và diện tích người.
2. Lớp person / sidewalk bị nhầm thành gì (hàng của ma trận nhầm lẫn cộng dồn, % pixel GT).
3. Person IoU theo kích thước người trong ảnh (nhỏ/vừa/lớn) và theo độ sáng.
4. Tương phản người/nền trên ẢNH GHÉP (kênh Y): |Y_người - Y_vành quanh người| (vành 15 px),
   so nosam / sage / tb / vis / ir. Dùng nhãn GT val 151 chỉ để PHÂN TÍCH (không đưa vào mô hình).

Chạy: cd code/b3 && SAGE_MAN=../scene_v1/kham_pha SAGE_RUNS=../scene_v1/runs_b0_gop \
        SAGE_ANH=<thư mục ảnh ghép> python ../scene_v1/phan_tich_nosam.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, RUNS, TEN_LOP, duong  # noqa: E402

RA = Path(__file__).resolve().parent / "phan_tich_nosam"
RA.mkdir(exist_ok=True)
BT = {"nosam_kp841": [0, 1, 2], "sage": [0, 1, 2], "tb": [0, 1]}
P = TEN_LOP.index("person")
pd.set_option("display.width", 200)


def doc(bt, seeds):
    cs = []
    for s in seeds:
        z = np.load(RUNS / f"{bt}_b0_s{s}" / "dev_nham_last.npz")
        cs.append(z["conf"].astype(np.float64))
        ids = [str(i) for i in z["image_id"]]
    return np.stack(cs), ids          # (seed, ảnh, 14, 14)


def iou(c):
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    gt, pr = c.sum(-1), c.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (gt + pr - tp), gt   # (..., lớp)


def main():
    C, ids = {}, None
    for bt, ss in BT.items():
        C[bt], i2 = doc(bt, ss)
        assert ids is None or i2 == ids
        ids = i2
    n = len(ids)
    gt_px = C["sage"][0].sum(-1)                     # (ảnh, lớp)
    so_anh = (gt_px > 0).sum(0)
    lop10 = np.where(so_anh >= 20)[0]
    print("10 lớp:", [TEN_LOP[k] for k in lop10])

    # ---- 1. mIoU từng ảnh
    mi = {}
    for bt in BT:
        I, g = iou(C[bt])                            # (seed, ảnh, lớp)
        I = np.where(g > 0, I, np.nan)[..., lop10]
        mi[bt] = np.nanmean(np.nanmean(I, -1), 0)    # TB lớp trong ảnh, rồi TB seed
    man = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str}).set_index("image_id").loc[ids]
    sang, nguoi = [], []
    for k, i in enumerate(ids):
        v = cv2.imread(str(duong(man.loc[i, "vis_path"])))
        sang.append(cv2.cvtColor(v, cv2.COLOR_BGR2YCrCb)[..., 0].mean())
        nguoi.append(gt_px[k, P] / gt_px[k].sum() * 100)
    df = pd.DataFrame({"id": ids, "sang_vis": sang, "nguoi_%": nguoi,
                       "nosam": mi["nosam_kp841"], "sage": mi["sage"], "tb": mi["tb"]})
    df["d_sage"] = df.nosam - df.sage
    df["d_tb"] = df.nosam - df.tb
    df.round(2).to_csv(RA / "miou_tung_anh.csv", index=False)
    print("\n=== 1. mIoU từng ảnh (10 lớp, TB seed) ===")
    print(f"nosam thua SAGE ở {(df.d_sage < 0).sum()}/{n} ảnh, thua TB ở {(df.d_tb < 0).sum()}/{n} ảnh")
    print("Tương quan Spearman của hiệu với: ")
    for c in ["sang_vis", "nguoi_%"]:
        print(f"  {c:9s}: d_sage {df.d_sage.corr(df[c], 'spearman'):+.2f} | d_tb {df.d_tb.corr(df[c], 'spearman'):+.2f}")
    print("\n10 ảnh nosam THUA SAGE nhiều nhất:")
    print(df.nsmallest(10, "d_sage").round(1).to_string(index=False))
    print("\n10 ảnh nosam HƠN SAGE nhiều nhất:")
    print(df.nlargest(10, "d_sage").round(1).to_string(index=False))
    # theo đoạn cảnh (khối id liền nhau, cách nhau > 5)
    so = np.array([int(i) for i in ids])
    doan = np.concatenate([[0], np.cumsum(np.diff(so) > 5)])
    df["doan"] = [f"{so[doan == d].min():05d}-{so[doan == d].max():05d}" for d in doan]
    g = df.groupby("doan").agg(so_anh=("id", "size"), sang=("sang_vis", "mean"),
                               d_sage=("d_sage", "mean"), d_tb=("d_tb", "mean"))
    print("\nTheo đoạn cảnh (TB hiệu mIoU ảnh):")
    print(g.round(2).to_string())

    # ---- 2. person / sidewalk nhầm thành gì
    print("\n=== 2. Pixel GT bị dự đoán thành (% pixel GT, cộng dồn ảnh, TB seed) ===")
    rows = []
    for ten in ["person", "sidewalk"]:
        k = TEN_LOP.index(ten)
        for bt in BT:
            hang = C[bt][:, :, k, :].sum(1).mean(0)
            hang = hang / hang.sum() * 100
            r = {"lop": ten, "bien_the": bt}
            r.update({TEN_LOP[j]: hang[j] for j in np.argsort(-hang)[:6]})
            rows.append(r)
    t = pd.DataFrame(rows).fillna(0).set_index(["lop", "bien_the"])
    print(t.round(1).to_string())
    t.round(2).to_csv(RA / "nham_person_sidewalk.csv")
    # thừa person: pixel dự đoán person mà GT khác
    for bt in BT:
        cot = C[bt][:, :, :, P].sum(1).mean(0)
        print(f"  {bt:12s}: dự đoán person sai (GT lớp khác) = {(cot.sum() - cot[P]) / cot.sum() * 100:.1f}% số pixel dự đoán person")

    # ---- 3. person IoU theo kích thước người & độ sáng
    print("\n=== 3. Person IoU (cộng dồn) theo nhóm ảnh ===")
    co = gt_px[:, P] > 0
    q = np.quantile(df.loc[co, "nguoi_%"], [1 / 3, 2 / 3])
    nhom_kt = np.where(df["nguoi_%"] < q[0], "nho", np.where(df["nguoi_%"] < q[1], "vua", "lon"))
    qs = np.quantile(df.sang_vis, [1 / 3, 2 / 3])
    nhom_s = np.where(df.sang_vis < qs[0], "toi", np.where(df.sang_vis < qs[1], "vua", "sang"))
    rows = []
    for ten_nhom, nh in [("kich_thuoc", nhom_kt), ("do_sang", nhom_s)]:
        for gi in pd.unique(nh):
            m = co & (nh == gi)
            r = {"nhom": ten_nhom, "gia_tri": gi, "so_anh": int(m.sum())}
            for bt in BT:
                c = C[bt][:, m].sum(1)          # (seed, 14, 14)
                I, _ = iou(c)
                r[bt] = np.nanmean(I[:, P]) * 100
            rows.append(r)
    t3 = pd.DataFrame(rows)
    t3["nosam-sage"] = t3.nosam_kp841 - t3.sage
    t3["nosam-tb"] = t3.nosam_kp841 - t3.tb
    print(f"(ngưỡng kích thước người: {q[0]:.2f}% / {q[1]:.2f}% diện tích ảnh)")
    print(t3.round(2).to_string(index=False))
    t3.round(2).to_csv(RA / "person_theo_nhom.csv", index=False)

    # ---- 4. Tương phản người / vành quanh người trên ảnh ghép
    print("\n=== 4. Tương phản người - vành 15 px (kênh Y, mức xám) trên ảnh có người ===")
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    nguon = {"nosam_kp841": None, "sage": None, "tb": None, "vis": "vis", "ir": "ir"}
    rows = []
    for k, i in enumerate(ids):
        if gt_px[k, P] < 200:
            continue
        lab = cv2.imread(str(duong(man.loc[i, "label_path"])), cv2.IMREAD_GRAYSCALE)
        mn = (lab == 8).astype(np.uint8)   # nhãn FMB 8 = person
        vanh = (cv2.dilate(mn, ker) > 0) & (mn == 0) & (lab != 0)
        if vanh.sum() < 100:
            continue
        r = {"id": i, "sang_vis": df.loc[k, "sang_vis"]}
        for bt, loai in nguon.items():
            if loai == "vis":
                y = cv2.cvtColor(cv2.imread(str(duong(man.loc[i, "vis_path"]))), cv2.COLOR_BGR2YCrCb)[..., 0]
            elif loai == "ir":
                y = cv2.imread(str(duong(man.loc[i, "ir_path"])), cv2.IMREAD_GRAYSCALE)
            else:
                f = ANH / bt / "train" / f"{i}.png"
                y = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2YCrCb)[..., 0]
            y = y.astype(np.float64)
            r[bt] = abs(y[mn > 0].mean() - y[vanh].mean())
        rows.append(r)
    t4 = pd.DataFrame(rows)
    t4.round(2).to_csv(RA / "tuong_phan_nguoi.csv", index=False)
    cot = list(nguon)
    print(f"{len(t4)} ảnh có người (>= 200 px). Trung vị:")
    print(t4[cot].median().round(2).to_string())
    ts = np.quantile(t4.sang_vis, [1 / 3, 2 / 3])
    t4["nhom_sang"] = np.where(t4.sang_vis < ts[0], "1_toi", np.where(t4.sang_vis < ts[1], "2_vua", "3_sang"))
    print("\nTrung vị theo độ sáng VIS:")
    print(t4.groupby("nhom_sang")[cot].median().round(2).to_string())
    print(f"\nnosam < sage ở {(t4.nosam_kp841 < t4.sage).mean() * 100:.0f}% ảnh có người; "
          f"nosam < tb ở {(t4.nosam_kp841 < t4.tb).mean() * 100:.0f}%")
    print(f"\nĐã ghi các bảng vào {RA}")


if __name__ == "__main__":
    main()
