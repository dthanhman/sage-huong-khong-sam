"""Phân tích kỹ một biến thể A2 so với nosam (và SAGE, TB) trên val 151, B0 seed 0 (20/11).

1. IoU từng lớp: lớp nào lên / xuống so với nosam.
2. Person IoU theo kích thước người & độ sáng (ngưỡng giống phan_tich_nosam.py).
3. Nhầm lẫn: person/car bị đoán thành gì; tỉ lệ dự đoán person sai (quá tay -> tăng).
4. Theo từng ảnh có người: độ nổi bật tăng (so nosam) có đi kèm person IoU tăng không (Spearman).
Ưu tiên run máy nhà (runs_b0), không có thì lấy Kaggle (runs_b0_gop) – ghi rõ máy.

Chạy: cd b3 && SAGE_MAN=../scene_v1/kham_pha SAGE_ANH=.../seg/anh python ../scene_v1/phan_tich_a2.py --a nosam_kp841_vung10
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, TEN_LOP, duong  # noqa: E402

GOC = Path(__file__).resolve().parent
NHA, GOP = GOC / "runs_b0", GOC / "runs_b0_gop"
P, CAR = TEN_LOP.index("person"), TEN_LOP.index("car")
pd.set_option("display.width", 200)


def doc(bt):
    for d, may in ((NHA, "nhà"), (GOP, "kaggle")):
        p = d / f"{bt}_b0_s0" / "dev_nham_last.npz"
        if p.exists():
            z = np.load(p)
            return z["conf"].astype(np.float64), [str(i) for i in z["image_id"]], may
    sys.exit(f"chưa có kết quả B0 s0 của {bt}")


def iou(c):
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (c.sum(-1) + c.sum(-2) - tp) * 100


def tuong_phan(bt, ids, man, ker):
    """|TB_Y(người) − TB_Y(vành 15 px)| từng ảnh (NaN nếu ít người)."""
    out = []
    for i in ids:
        lab = cv2.imread(str(duong(man.loc[i, "label_path"])), cv2.IMREAD_GRAYSCALE)
        mn = (lab == 8).astype(np.uint8)
        vanh = (cv2.dilate(mn, ker) > 0) & (mn == 0) & (lab != 0)
        if mn.sum() < 200 or vanh.sum() < 100:
            out.append(np.nan)
            continue
        y = cv2.cvtColor(cv2.imread(str(ANH / bt / "train" / f"{i}.png")), cv2.COLOR_BGR2YCrCb)[..., 0].astype(float)
        out.append(abs(y[mn > 0].mean() - y[vanh].mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="biến thể cần phân tích, vd nosam_kp841_vung10")
    ap.add_argument("--goc", default="nosam_kp841")
    ap.add_argument("--them", nargs="*", default=["sage", "tb"], help="mốc tham chiếu thêm")
    a = ap.parse_args()
    bts = [a.a, a.goc] + a.them
    C, ids, may = {}, None, {}
    for bt in bts:
        C[bt], i2, may[bt] = doc(bt)
        assert ids is None or i2 == ids
        ids = i2
    print("Máy:", ", ".join(f"{b} ({m})" for b, m in may.items()))
    gt = C[a.goc].sum(-1)                              # (ảnh, lớp)
    so_anh = (gt > 0).sum(0)
    lop10 = np.where(so_anh >= 20)[0]

    # 1. IoU từng lớp
    t = pd.DataFrame({bt: iou(C[bt].sum(0)) for bt in bts}, index=TEN_LOP)
    t["so_anh"] = so_anh
    t[f"{a.a} − {a.goc}"] = t[a.a] - t[a.goc]
    t["lop10"] = [k in lop10 for k in range(len(TEN_LOP))]
    print("\n=== 1. IoU từng lớp (cộng dồn 151 ảnh) ===")
    print(t.round(2).to_string())
    print("mIoU 10 lớp:", "  ".join(f"{bt} {t.loc[t.lop10, bt].mean():.2f}" for bt in bts))

    # 2. person theo kích thước / độ sáng
    man = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str}).set_index("image_id").loc[ids]
    sang = np.array([cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2YCrCb)[..., 0].mean() for p in man.vis_path])
    nguoi = gt[:, P] / gt.sum(1) * 100
    co = gt[:, P] > 0
    q = np.quantile(nguoi[co], [1 / 3, 2 / 3])
    qs = np.quantile(sang, [1 / 3, 2 / 3])
    nhom = {"kich_thuoc": np.where(nguoi < q[0], "1_nho", np.where(nguoi < q[1], "2_vua", "3_lon")),
            "do_sang": np.where(sang < qs[0], "1_toi", np.where(sang < qs[1], "2_vua", "3_sang"))}
    rows = []
    for ten, nh in nhom.items():
        for g in sorted(set(nh[co])):
            m = co & (nh == g)
            r = {"nhom": ten, "gia_tri": g, "so_anh": int(m.sum())}
            r.update({bt: iou(C[bt][m].sum(0))[P] for bt in bts})
            rows.append(r)
    t2 = pd.DataFrame(rows)
    t2[f"{a.a} − {a.goc}"] = t2[a.a] - t2[a.goc]
    print("\n=== 2. Person IoU theo nhóm ảnh ===")
    print(t2.round(2).to_string(index=False))

    # 3. nhầm lẫn
    print("\n=== 3. Nhầm lẫn (% pixel) ===")
    for k, ten in ((P, "person"), (CAR, "car")):
        rows = []
        for bt in bts:
            h = C[bt].sum(0)[k]
            h = h / h.sum() * 100
            r = {"bien_the": bt}
            r.update({TEN_LOP[j]: h[j] for j in np.argsort(-h)[:5]})
            rows.append(r)
        print(f"GT {ten} bị đoán thành:")
        print(pd.DataFrame(rows).set_index("bien_the").fillna(0).round(1).to_string())
    for bt in bts:
        cot = C[bt].sum(0)[:, P]
        print(f"  {bt:22s}: dự đoán person mà GT khác = {(cot.sum() - cot[P]) / cot.sum() * 100:.1f}%"
              f" | GT person bị bỏ sót = {(1 - C[bt].sum(0)[P, P] / C[bt].sum(0)[P].sum()) * 100:.1f}%")

    # 4. nổi bật tăng có kéo person IoU tăng không (từng ảnh)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tp_a, tp_g = tuong_phan(a.a, ids, man, ker), tuong_phan(a.goc, ids, man, ker)
    ia, ig = iou(C[a.a])[:, P], iou(C[a.goc])[:, P]
    d = pd.DataFrame({"id": ids, "d_noi_bat": tp_a - tp_g, "d_person_iou": ia - ig, "nguoi_%": nguoi}).dropna()
    print(f"\n=== 4. Từng ảnh có người ({len(d)} ảnh) ===")
    print(f"Độ nổi bật tăng (trung vị) {d.d_noi_bat.median():+.1f} mức xám; person IoU tăng ở "
          f"{(d.d_person_iou > 0).mean() * 100:.0f}% ảnh (trung vị {d.d_person_iou.median():+.2f})")
    print(f"Spearman(d_noi_bat, d_person_iou) = {d.d_noi_bat.corr(d.d_person_iou, 'spearman'):+.2f}; "
          f"Spearman(nguoi_%, d_person_iou) = {d['nguoi_%'].corr(d.d_person_iou, 'spearman'):+.2f}")
    ra = GOC / "phan_tich_nosam" / f"a2_{a.a}"
    ra.mkdir(parents=True, exist_ok=True)
    t.round(3).to_csv(ra / "iou_lop.csv"); t2.round(3).to_csv(ra / "person_nhom.csv", index=False)
    d.round(3).to_csv(ra / "tung_anh.csv", index=False)
    print(f"\nĐã ghi {ra}")


if __name__ == "__main__":
    main()
