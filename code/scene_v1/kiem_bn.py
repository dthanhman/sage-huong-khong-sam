"""Kiểm nghi vấn BatchNorm: Student ở chế độ train (thống kê theo batch) cho ảnh khác chế độ eval (thống kê chạy)?
So trên 16 ảnh train kham_pha, crop 384x512 (như m11) và 192x256 (như tb_sua), batch 4.
In: |ảnh train-mode - ảnh eval-mode| (mức xám) và rào chắn nền B ở mỗi chế độ.
Ví dụ: python kiem_bn.py tbsua_w150
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

TL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TL / "cai_tien"))
sys.path.insert(0, str(TL / "b3"))
from chung import duong  # noqa: E402
from train_student_seg import SOBEL_X, SOBEL_Y, Network, STUDENT  # noqa: E402
from m11_hoc_chung import rao_tb  # noqa: E402

def main():
    ten = sys.argv[1]
    sd = torch.load(STUDENT / f"{ten}.pt", map_location="cpu", weights_only=True)
    s = float(sd.pop("_tb_scale"))
    df = pd.read_csv(TL / "scene_v1/kham_pha/train.csv", dtype={"image_id": str})
    rng = np.random.default_rng(0)
    rows = df.iloc[rng.choice(len(df), 16, replace=False)]
    Y = np.stack([cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)[..., 0] for r in rows.itertuples()])
    IR = np.stack([cv2.imread(str(duong(r.ir_path)), 0) for r in rows.itertuples()])
    Y, IR = Y.astype(np.float32) / 255, IR.astype(np.float32) / 255
    kx, ky = SOBEL_X.cuda(), SOBEL_Y.cuda()
    for ch, cw in [(384, 512), (192, 256), (600, 800)]:
        net = Network(); net.load_state_dict(sd); net = net.cuda()
        d, r_tr, r_ev = [], [], []
        for b in range(0, 16, 4):
            i0, j0 = rng.integers(0, 600 - ch + 1), rng.integers(0, 800 - cw + 1)
            y = torch.from_numpy(Y[b:b + 4, None, i0:i0 + ch, j0:j0 + cw]).cuda()
            ir = torch.from_numpy(IR[b:b + 4, None, i0:i0 + ch, j0:j0 + cw]).cuda()
            with torch.no_grad():
                net.eval(); o_ev = ((y + ir) / 2 + s * net(y, ir)[0]).clamp(0, 1)
                net.train(); o_tr = ((y + ir) / 2 + s * net(y, ir)[0]).clamp(0, 1)
                net.load_state_dict(sd)   # khôi phục running stats sau forward train-mode
            d.append((o_tr - o_ev).abs().mean().item() * 255)
            r_tr.append(rao_tb(o_tr, y, ir, kx, ky, 150, 27).item())
            r_ev.append(rao_tb(o_ev, y, ir, kx, ky, 150, 27).item())
        print(f"crop {ch}x{cw}: |train - eval| = {np.mean(d):.2f} mức xám; rào train-mode {np.mean(r_tr):.3f}, "
              f"eval-mode {np.mean(r_ev):.3f}")


if __name__ == "__main__":
    main()
