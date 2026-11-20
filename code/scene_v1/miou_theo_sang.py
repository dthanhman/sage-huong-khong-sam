"""mIoU theo nhóm độ sáng (19/11). Quy tắc giống E3 (08/11): độ sáng = trung bình kênh Y của ảnh VIS;
chia 3 nhóm bằng nhau theo tam phân vị (tối / vừa / sáng). Không dùng nhãn, không dùng kết quả mô hình.

Với mỗi nhóm: mIoU_co_mat từ ma trận nhầm lẫn cộng dồn các ảnh trong nhóm, từng seed rồi lấy trung bình seed.
Hiệu A − B: khoảng tin cậy 95% bootstrap theo khối 10 ảnh id liền nhau (như so_sanh_cap.py), trong nhóm,
cùng bộ ảnh cho mọi seed, hiệu tính trên trung bình seed.

Chạy: cd b3 && SAGE_MAN=../scene_v1/kham_pha SAGE_RUNS=../scene_v1/runs_b0_gop python ../scene_v1/miou_theo_sang.py \
        --bien_the tb sage nosam_kp841 m11a_l5_e30_bn m11b_l15_e30_bn --model b0 --seeds 0 1 --goc sage tb
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import MAN, RUNS, cac_miou, duong  # noqa: E402
from so_sanh_cap import khoi_anh  # noqa: E402

B, NHOM = 2000, ["toi", "vua", "sang"]


def miou(c):
    return cac_miou(c)[0]["mIoU_co_mat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", nargs="+", required=True)
    ap.add_argument("--model", default="b0")
    ap.add_argument("--hau", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--goc", nargs="+", default=["sage", "tb"], help="các mốc để so hiệu")
    ap.add_argument("--ra", default=None, help="file csv kết quả")
    a = ap.parse_args()

    df = pd.read_csv(MAN / "dev.csv", dtype={"image_id": str})
    sang = {r.image_id: cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)[:, :, 0].mean()
            for r in df.itertuples()}

    C = {}   # (bt, seed) -> conf N x NC x NC, theo thứ tự ids
    ids = None
    for bt in a.bien_the:
        for s in a.seeds:
            z = np.load(RUNS / f"{bt}_{a.model}_s{s}{a.hau}" / "dev_nham_last.npz")
            i = [str(x) for x in z["image_id"]]
            if ids is None:
                ids = i
            assert i == ids, f"{bt} s{s}: thứ tự ảnh khác"
            C[bt, s] = z["conf"].astype(np.int64)
    y = np.array([sang[i] for i in ids])
    q = np.quantile(y, [1 / 3, 2 / 3])
    nhom = np.where(y < q[0], "toi", np.where(y < q[1], "vua", "sang"))
    print(f"{len(ids)} ảnh | ngưỡng Y_VIS: tối < {q[0]:.1f} ≤ vừa < {q[1]:.1f} ≤ sáng")
    for g in NHOM:
        print(f"  {g}: {(nhom == g).sum()} ảnh, Y_VIS {y[nhom == g].min():.0f}–{y[nhom == g].max():.0f}")

    rng = np.random.default_rng(0)
    dong = []
    for g in NHOM + ["chung"]:
        idx = np.arange(len(ids)) if g == "chung" else np.where(nhom == g)[0]
        khoi = [k for k in (np.intersect1d(kk, idx) for kk in khoi_anh([ids[j] for j in range(len(ids))])) if len(k)]
        m = {bt: np.mean([miou(C[bt, s][idx].sum(0)) for s in a.seeds]) for bt in a.bien_the}
        for bt in a.bien_the:
            dong.append({"nhom": g, "so_anh": len(idx), "bien_the": bt, "mIoU": m[bt],
                         **{f"s{s}": miou(C[bt, s][idx].sum(0)) for s in a.seeds}})
        # bootstrap theo khối trong nhóm
        mau = [np.concatenate([khoi[k] for k in rng.integers(0, len(khoi), len(khoi))]) for _ in range(B)]
        for goc in a.goc:
            for bt in a.bien_the:
                if bt == goc:
                    continue
                h = [np.mean([miou(C[bt, s][ix].sum(0)) - miou(C[goc, s][ix].sum(0)) for s in a.seeds]) for ix in mau]
                lo, hi = np.percentile(h, [2.5, 97.5])
                dong.append({"nhom": g, "so_anh": len(idx), "bien_the": f"{bt} − {goc}", "mIoU": m[bt] - m[goc],
                             "CI_lo": lo, "CI_hi": hi,
                             **{f"s{s}": miou(C[bt, s][idx].sum(0)) - miou(C[goc, s][idx].sum(0)) for s in a.seeds}})
    kq = pd.DataFrame(dong)
    pd.set_option("display.width", 200)
    for g in NHOM + ["chung"]:
        print(f"\n== {g}")
        print(kq[kq.nhom == g].drop(columns="nhom").round(2).to_string(index=False))
    if a.ra:
        kq.to_csv(a.ra, index=False)
        print("Ghi", a.ra)


if __name__ == "__main__":
    main()
