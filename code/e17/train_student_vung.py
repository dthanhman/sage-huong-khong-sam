"""Train S-vùng (23/11, b3/sage_student/model_vung.py): chỉ học ĐẦU VÙNG, đích = bản đồ vùng của thầy vhn
(scene_v1/_vhn/p_thay.npz, tạo bằng b3/vung_hai_nhanh.py luu_p: nguồn CHÉO, A đoán nửa B và ngược lại).

Loss: BCE(logit, p_thay) ở 1/4 độ phân giải (đích mềm). Nền nosam_kp841 + ResNet18 đóng băng.
Chỉ train trên tập train (841, SAGE_MAN=scene_v1/kham_pha); dev chỉ để BÁO đo bản đồ, không chọn checkpoint (lưu bản cuối).

  SAGE_MAN=.../kham_pha SO_TRAIN=841 python train_student_vung.py --ten svung --khoi_tao nosam_kp841
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "b3"))
from chung import MAN, STUDENT, duong  # noqa: E402

sys.path.insert(0, str(STUDENT))
from model_vung import NetworkVung  # noqa: E402


def nap(tap, P):
    df = pd.read_csv(MAN / f"{tap}.csv", dtype={"image_id": str})
    Y = [cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2YCrCb)[:, :, 0] for p in df.vis_path]
    IR = [cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2GRAY) for p in df.ir_path]
    T = [P[i] for i in df.image_id]
    return df, Y, IR, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", required=True)
    ap.add_argument("--khoi_tao", default="nosam_kp841")
    ap.add_argument("--p_thay", default=str(Path(__file__).resolve().parent.parent / "scene_v1/_vhn/p_thay.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rong", type=int, default=64)
    ap.add_argument("--crop", type=int, nargs=2, default=[384, 512])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    z = np.load(a.p_thay)
    P = dict(zip(z["image_id"], z["p"]))
    nguon = dict(zip(z["image_id"], z["nguon"]))
    df, Y, IR, T = nap("train", P)
    assert len(df) == int(os.environ.get("SO_TRAIN", 841)) and not df.vis_path.str.startswith("test").any()
    assert all(nguon[i] in ("A", "B", "dem") for i in df.image_id), "bản đồ train phải là nguồn chéo / đệm"
    ddf, DY, DIR, DT = nap("dev", P)
    assert not set(df.image_id) & set(ddf.image_id)
    print(f"train {len(df)} ảnh, dev {len(ddf)} ảnh; nguồn bản đồ train {pd.Series([nguon[i] for i in df.image_id]).value_counts().to_dict()}", flush=True)

    net = NetworkVung(32, pretrained=True, rong=a.rong)
    sd = torch.load(STUDENT / f"{a.khoi_tao}.pt", map_location="cpu", weights_only=True)
    net.goc.load_state_dict(sd, strict=True)
    net = net.cuda().train()
    hoc = [p for p in net.parameters() if p.requires_grad]
    print(f"S-vùng: {sum(p.numel() for p in net.parameters()) / 1e6:.3f}M tham số, học {sum(p.numel() for p in hoc) / 1e6:.4f}M", flush=True)
    opt = torch.optim.Adam(hoc, lr=a.lr)
    so_buoc = a.epochs * math.ceil(len(df) / a.batch)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, so_buoc, eta_min=a.lr / 100)
    ch, cw = a.crop
    t2 = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float().div(255)

    @torch.no_grad()
    def do_dev():
        net.eval()
        bce, tu, mau = 0.0, np.zeros(2), np.zeros(2)
        for y, ir, t in zip(DY, DIR, DT):
            lg = net.logit_vung(t2(y)[None, None].cuda(), t2(ir)[None, None].cuda())[0]
            tt = t2(t).cuda()[:, :lg.shape[1], :lg.shape[2]]
            lg = lg[:, :tt.shape[1], :tt.shape[2]]
            bce += F.binary_cross_entropy_with_logits(lg, tt).item()
            p = torch.sigmoid(lg)
            tu += (p * tt).sum((1, 2)).cpu().numpy(); mau += (p + tt - p * tt).sum((1, 2)).cpu().numpy()
        net.train()
        return bce / len(DY), tu / mau           # IoU mềm (người, VIS) giữa bản đồ Student và thầy

    log, t0, buoc = [], time.time(), 0
    for ep in range(1, a.epochs + 1):
        thu_tu = rng.permutation(len(df))
        tong, n = 0.0, 0
        for bi in range(0, len(df), a.batch):
            ys, irs, ts = [], [], []
            for k in thu_tu[bi:bi + a.batch]:
                H, W = Y[k].shape
                i0 = rng.integers(0, (H - ch) // 4 + 1) * 4
                j0 = rng.integers(0, (W - cw) // 4 + 1) * 4
                y, ir = Y[k][i0:i0 + ch, j0:j0 + cw], IR[k][i0:i0 + ch, j0:j0 + cw]
                t = T[k][:, i0 // 4:(i0 + ch) // 4, j0 // 4:(j0 + cw) // 4]
                if rng.random() < 0.5:
                    y, ir, t = y[:, ::-1], ir[:, ::-1], t[:, :, ::-1]
                ys.append(t2(y)), irs.append(t2(ir)), ts.append(t2(t))
            y = torch.stack(ys)[:, None].cuda(); ir = torch.stack(irs)[:, None].cuda(); t = torch.stack(ts).cuda()
            lg = net.logit_vung(y, ir)
            loss = F.binary_cross_entropy_with_logits(lg, t)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); buoc += 1
            tong += loss.item(); n += 1
        dong = {"epoch": ep, "loss": tong / n, "phut": (time.time() - t0) / 60}
        if ep % 5 == 0 or ep == 1 or ep == a.epochs:
            b, iou = do_dev()
            dong |= {"dev_bce": b, "dev_iou_nguoi": iou[0], "dev_iou_vis": iou[1]}
            print(f"epoch {ep:3d} loss {dong['loss']:.4f} | dev bce {b:.4f} IoU mềm người {iou[0]:.3f} VIS {iou[1]:.3f} | {dong['phut']:.1f} phút", flush=True)
        log.append(dong)
    torch.save(net.state_dict(), STUDENT / f"{a.ten}.pt")
    pd.DataFrame(log).to_csv(STUDENT / f"{a.ten}_log.csv", index=False)
    print(f"Lưu {STUDENT / (a.ten + '.pt')}")


if __name__ == "__main__":
    main()
