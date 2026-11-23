"""L_int theo lớp (21/11): tính trọng số IR cho từng lớp nhãn TỰ ĐỘNG theo luật đặt trước (không chọn tay).

Bằng chứng (val151 B0): mọi ảnh ghép được person (nhờ IR) nhưng mất lớp VIS mạnh (truck, biển, đèn, vỉa hè);
hai_nhanh lấy được cả hai phía. -> mỗi vùng lớp nên kéo về nguồn mạnh hơn cho lớp đó.

LUẬT (đặt trước khi train): d_c = IoU_VIS(c) − TB[IoU_SAGE, IoU_TB, IoU_nosam](c)  (val151 B0, mọi seed có sẵn)
  d_c >= +2  -> lớp nghiêng VIS: w_IR = 0,1
  d_c <= −5  -> lớp nghiêng IR : w_IR = 0,7
  còn lại    -> w_IR = 0,4 (= gốc SAGE 3·Y + 2·IR)
Nhãn 0 (unlabeled) và lớp có < 10 ảnh trong val151 (nhiễu: motorcycle 1, bicycle 2) -> 0,4. Chỉ dùng val151 (tập thiết kế); Dev228 giữ riêng.
Chạy: python scene_v1/tinh_w_lop.py  -> scene_v1/w_lop.json
"""
import json
from pathlib import Path

import numpy as np

DAY = Path(__file__).resolve().parent
GOP = DAY / "runs_b0_gop"
TEN = ["road", "sidewalk", "building", "traffic_light", "traffic_sign", "vegetation", "sky", "person", "car",
       "truck", "bus", "motorcycle", "bicycle", "pole"]   # nhãn 1..14


def iou_tb(bt):
    v = []
    for s in range(3):
        p = GOP / f"{bt}_b0_s{s}" / "dev_nham_last.npz"
        if p.exists():
            C = np.load(p)["conf"].sum(0).astype(float)
            d = np.diag(C)
            with np.errstate(invalid="ignore", divide="ignore"):
                v.append(100 * d / (C.sum(0) + C.sum(1) - d))
    return np.nanmean(v, 0), len(v)


def so_anh():
    C = np.load(GOP / "sage_b0_s0" / "dev_nham_last.npz")["conf"]   # (ảnh, gt, pred)
    return (C.sum(2) > 0).sum(0)


def main():
    vis, nv = iou_tb("vis")
    na = so_anh()
    ghep = [iou_tb(b) for b in ("sage", "tb", "nosam_kp841")]
    g = np.nanmean([x for x, _ in ghep], 0)
    w = {0: 0.4}
    print(f"VIS {nv} seed; ghép (sage/tb/nosam) {[n for _, n in ghep]} seed")
    for k, t in enumerate(TEN):
        d = vis[k] - g[k]
        ww = 0.4 if (np.isnan(d) or na[k] < 10) else (0.1 if d >= 2 else 0.7 if d <= -5 else 0.4)
        w[k + 1] = ww
        print(f"  {k + 1:2d} {t:14s} {na[k]:3d} ảnh  VIS {vis[k]:5.1f}  ghép {g[k]:5.1f}  d {d:+6.1f}  -> w_IR {ww}")
    (DAY / "w_lop.json").write_text(json.dumps({str(k): v for k, v in w.items()}, indent=1), encoding="utf-8")
    print("Ghi", DAY / "w_lop.json")


if __name__ == "__main__":
    main()
