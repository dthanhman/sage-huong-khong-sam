"""Cổng 2: chỉ số ghép trên Dev (hoặc tập khác) cho nhiều biến thể, cùng một code đo.

Tính trên kênh Y: Y ảnh ghép (RGB->YCrCb), Y của VIS, IR 1 kênh, 800x600 uint8.
Code chỉ số: pr6_eval/src/evaluation/metrics.py (Lộc lớn). MS-SSIM kiểu TỔNG (như bài, có thể > 1).

Ví dụ:
  SAGE_ANH=<thư mục ảnh ghép> \
  python chi_so_ghep.py --bien_the sage nosam tb vis --tap dev
Ra: cai_tien/chi_so_ghep/{tap}_tung_anh.csv (từng ảnh) và {tap}_tong_hop.csv (trung bình + % so với SAGE).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DAY = Path(__file__).resolve().parent
sys.path.insert(0, str(DAY.parent / "b3"))
sys.path.insert(0, str(DAY.parent / "pr6_eval"))   # bản repo: code/pr6_eval
from chung import ANH, MAN, duong, thu_muc  # noqa: E402
from src.evaluation.metrics import (entropy, fusion_ms_ssim, q_abf,  # noqa: E402
                                    standard_deviation, sum_of_correlation_differences)

CHI_SO = ["SCD", "MS_SSIM", "Q_ABF", "EN", "SD", "MS_SSIM_IR", "MS_SSIM_VIS"]
NGUONG = {"SCD": 0.95, "MS_SSIM": 0.98, "Q_ABF": 0.93}   # cổng 2, chốt 17/11 (SAGE − 2×SD 5 seed), xem README


def do_mot(args):
    bt, r = args
    vis = cv2.imread(str(duong(r["vis_path"])))
    yv = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ir = cv2.imread(str(duong(r["ir_path"])), cv2.IMREAD_GRAYSCALE)
    f = cv2.imread(str(ANH / bt / r["thu_muc"] / f"{r['image_id']}.png"))
    assert f is not None, f"thiếu ảnh {bt}/{r['image_id']}"
    yf = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ms = fusion_ms_ssim(yf, ir, yv, aggregation="sum")
    return {
        "bien_the": bt, "image_id": r["image_id"],
        "SCD": sum_of_correlation_differences(yf, ir, yv),
        "MS_SSIM": ms["MS_SSIM"], "MS_SSIM_IR": ms["MS_SSIM_IR"], "MS_SSIM_VIS": ms["MS_SSIM_VIS"],
        "Q_ABF": q_abf(yf, ir, yv),
        "EN": entropy(yf),
        "SD": standard_deviation(yf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", nargs="+", required=True)
    ap.add_argument("--tap", default="dev")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hau", default="", help="hậu tố tên file ra, để không ghi đè bảng mốc (vd m8)")
    a = ap.parse_args()

    df = pd.read_csv(MAN / f"{a.tap}.csv", dtype={"image_id": str})
    df["thu_muc"] = [thu_muc(r) for r in df.itertuples()]
    viec = [(bt, r) for bt in a.bien_the for r in df.to_dict("records")]
    with ProcessPoolExecutor(a.workers) as ex:
        kq = pd.DataFrame(list(ex.map(do_mot, viec, chunksize=8)))

    ra = DAY / "chi_so_ghep"
    ra.mkdir(exist_ok=True)
    kq.to_csv(ra / f"{a.tap}{a.hau and '_' + a.hau}_tung_anh.csv", index=False)
    th = kq.groupby("bien_the")[CHI_SO].mean().reindex(a.bien_the)
    if "sage" in th.index:
        for c in ["SCD", "MS_SSIM", "Q_ABF"]:
            th[f"{c}_%sage"] = 100 * th[c] / th.loc["sage", c]
        th["qua_cong2"] = np.logical_and.reduce([th[f"{c}_%sage"] >= 100 * v for c, v in NGUONG.items()])
    th.to_csv(ra / f"{a.tap}{a.hau and '_' + a.hau}_tong_hop.csv")
    pd.set_option("display.width", 200)
    print(f"{a.tap}: {len(df)} ảnh")
    print(th.round(4).to_string())


if __name__ == "__main__":
    main()
