"""M1/M2 – Dạy thêm Student của SAGE để ảnh ghép giúp phân đoạn tốt hơn.

M1: loss = loss nội dung (như SAGE: MSE + gradient)  +  lam * CE( SegFormer-B0 ĐÓNG BĂNG (ảnh ghép), nhãn )
M2: đổi tỉ lệ cường độ IR trong loss nội dung (SAGE gốc: 3*MSE(Y) + 2*MSE(IR) = 60% VIS / 40% IR) bằng --w_ir.

Chống rò dữ liệu / học tủ:
  - Chỉ dùng 1.045 ảnh TRAIN (manifests/train.csv). Dev 175 ảnh để chọn, test không đụng.
  - Mạng dẫn đường = SegFormer-B0 cũ (luu_tru/tim_gap_b0_0811/seg/runs/<run>/last.pt), train trên đúng 1.045 ảnh đó.
  - Chấm thử (proxy) trên Dev bằng một B0 KHÁC seed (không phải mạng dẫn đường). Kết quả thật: SegFormer-B3 train lại.

Ra: ../b3/sage_student/{ten}.pt, {ten}_log.csv, {ten}_proxy.json
Ví dụ:
  python train_student_seg.py --ten m1_l1 --lam 1
  python train_student_seg.py --ten m2_w05 --lam 0 --w_ir 0.5 --init none --epochs 100   (M2: train lại từ đầu)
  python train_student_seg.py --chi_proxy sage        (chỉ chấm proxy một checkpoint có sẵn)
"""
import argparse
import json
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

DAY = Path(__file__).resolve().parent
B3 = DAY if (DAY / "chung.py").exists() else DAY.parent / "b3"
sys.path.insert(0, str(B3))
from chung import IGNORE, MAN, MEAN, NC, STUDENT, cac_miou, doc_nhan, duong, nham  # noqa: E402

sys.path.insert(0, str(STUDENT))
from model import Network  # noqa: E402

B0_RUNS = DAY.parents[1] / "luu_tru/tim_gap_b0_0811/seg/runs"
# Thầy chéo: thư mục chia 2 nửa và nơi chứa runs_b0_{A,B}. Mặc định = split cũ (nua/, cai_tien/).
# scene_v1: SAGE_NUA=.../scene_v1/nua_kp  SAGE_THAY_RUNS=.../scene_v1
NUA = Path(os.environ.get("SAGE_NUA", DAY / "nua"))
THAY_RUNS = Path(os.environ.get("SAGE_THAY_RUNS", DAY))
SOBEL_X = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
SOBEL_Y = torch.tensor([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None]


def grad(x, kx, ky):
    return F.conv2d(x, kx, padding=1).abs() + F.conv2d(x, ky, padding=1).abs()


def ycrcb_sang_rgb(y, crcb):
    """Như cv2.COLOR_YCrCb2RGB (giá trị 0..1), khả vi. y: Bx1xHxW, crcb: Bx2xHxW."""
    cr, cb = crcb[:, :1] - 0.5, crcb[:, 1:] - 0.5
    r = y + 1.403 * cr
    g = y - 0.714 * cr - 0.344 * cb
    b = y + 1.773 * cb
    return torch.cat([r, g, b], 1).clamp(0, 1)


def chuan_hoa(rgb):
    m = torch.tensor(MEAN, device=rgb.device)[None, :, None, None]
    s = torch.tensor(np.array([0.229, 0.224, 0.225], np.float32), device=rgb.device)[None, :, None, None]
    return (rgb - m) / s


def nap_b0(run, dev, n_train=1045):
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=NC, id2label={i: str(i) for i in range(NC)},
        label2id={str(i): i for i in range(NC)}, ignore_mismatched_sizes=True)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg["n_train"] == n_train and cfg["model"] == "b0" and not cfg.get("lop0"), f"{run}: sai B0 ({cfg['n_train']} ảnh)"
    m.load_state_dict(torch.load(run / "last.pt", map_location="cpu", weights_only=True))
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# 14 lớp FMB (thứ tự TEN_LOP) -> chỉ số lớp Cityscapes (19 lớp)
CITY = [0, 1, 2, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 5]


class ThayCityscapes(torch.nn.Module):
    """SegFormer học sẵn trên Cityscapes, CHƯA từng thấy FMB. Trả logits 14 lớp FMB (lấy đúng các kênh tương ứng)."""

    def __init__(self, ten="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.m = SegformerForSemanticSegmentation.from_pretrained(ten).eval()
        for q in self.m.parameters():
            q.requires_grad_(False)

    def forward(self, pixel_values):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.m(pixel_values=pixel_values).logits[:, CITY])


class DauTron(torch.nn.Module):
    """M4: Y = w*VIS + (1-w)*IR + r. Thân mạng = Student SAGE; đầu mới dự đoán w (sigmoid) và r (nhỏ).
    Đầu khởi tạo bằng 0 -> w = 0.5, r = 0 -> lúc đầu ảnh ra ĐÚNG BẰNG ghép trung bình (TB)."""

    def __init__(self, than, dim=32, r_max=0.2):
        super().__init__()
        self.than, self.r_max = than, r_max
        self.dau = torch.nn.Conv2d(dim, 2, 3, padding=1)
        torch.nn.init.zeros_(self.dau.weight)
        torch.nn.init.zeros_(self.dau.bias)

    def forward(self, y, ir):
        x5 = self.than(y, ir)[1][4]
        a, b = self.dau(x5).split(1, 1)
        w = torch.sigmoid(a)
        return (w * y + (1 - w) * ir + self.r_max * torch.tanh(b)).clamp(0, 1), w


def nap_anh(split):
    df = pd.read_csv(MAN / f"{split}.csv", dtype={"image_id": str})
    Y, CC, IR, LAB = [], [], [], []
    for r in df.itertuples():
        ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)
        Y.append(ycc[:, :, 0]), CC.append(ycc[:, :, 1:])
        IR.append(cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY))
        LAB.append(doc_nhan(duong(r.label_path), False).astype(np.uint8))  # 0..13, 255 = bỏ qua
    return df, np.stack(Y), np.stack(CC), np.stack(IR), np.stack(LAB)


def t(z, dev):
    z = torch.from_numpy(np.ascontiguousarray(z)).float().div(255)
    return (z[:, None] if z.ndim == 3 else z.permute(0, 3, 1, 2)).to(dev)


@torch.no_grad()
def cham_proxy(net, b0, data, dev):
    if b0 is None:   # --proxy none
        return float("nan")
    """mIoU Dev (bỏ unlabeled, 14 lớp) khi đưa ảnh ghép của Student vào một B0 đóng băng."""
    _, Y, CC, IR, LAB = data
    if isinstance(net, torch.nn.Module):
        net.eval()
    c = np.zeros((NC, NC), np.int64)
    for i in range(len(Y)):
        y, ir, cc = t(Y[i:i + 1], dev), t(IR[i:i + 1], dev), t(CC[i:i + 1], dev)
        if net == "vis":
            o = y
        elif net == "tb":
            o = ((y + ir) / 2 * 255).round() / 255
        else:
            o = net(y, ir)[0].clamp(0, 1)
        o = (o * 255).round() / 255                     # như ảnh PNG 8 bit thật
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = b0(pixel_values=chuan_hoa(ycrcb_sang_rgb(o, cc))).logits
        p = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear").argmax(1)[0].cpu().numpy()
        c += nham(LAB[i].astype(np.int64), p)
    if isinstance(net, torch.nn.Module):
        net.train()
    return cac_miou(c)[0]["mIoU_co_mat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", default="m1")
    ap.add_argument("--lam", type=float, default=1.0, help="trọng số CE phân đoạn (0 = chỉ loss nội dung)")
    ap.add_argument("--w_ir", type=float, default=0.4, help="tỉ lệ IR trong MSE cường độ (SAGE gốc 0.4)")
    ap.add_argument("--w_grad", type=float, default=1.0, help="nhân thêm cho loss gradient (1 = như SAGE)")
    ap.add_argument("--init", default="sage", help="sage | nosam | none (khởi tạo ngẫu nhiên)")
    ap.add_argument("--dan_duong", default="sage_b0_s0", help="run B0 làm mạng dẫn đường (đóng băng)")
    ap.add_argument("--proxy", default="sage_b0_s1", help="run B0 khác seed để chấm thử trên Dev; none = bỏ proxy")
    ap.add_argument("--thay", default="fmb", help="fmb (B0 học FMB, --dan_duong) | cityscapes (học sẵn, chưa thấy FMB)")
    ap.add_argument("--dau", default="sage", help="sage (đầu Tanh gốc) | tb (M4: trộn w, r, khởi tạo = TB)")
    ap.add_argument("--thay_bt", default="tb", help="thầy chéo học trên ảnh loại nào (tb | sage)")
    ap.add_argument("--hai_nua", action="store_true",
                    help="2 thầy chéo: B0 học nửa A dẫn ảnh nửa B và ngược lại (thầy không thuộc lòng ảnh nó chấm)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, nargs=2, default=[384, 512])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kiem_thay", action="store_true",
                    help="đo CE của 2 thầy chéo trên ảnh TB: nửa đã học vs nửa chưa thấy (thầy còn chỉ ra lỗi không)")
    ap.add_argument("--chi_proxy", default="", help="chỉ chấm proxy checkpoint sage_student/<tên>.pt ('vis', 'tb' = ảnh VIS, ghép TB)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("Nạp Dev ...", flush=True)
    dev_data = nap_anh("dev")
    if a.proxy == "none":   # scene_v1: B0 proxy cũ đã học ảnh val 151 -> không dùng
        b0_proxy = None
    else:
        b0_proxy = ThayCityscapes().to(dev) if a.proxy == "cityscapes" else nap_b0(B0_RUNS / a.proxy, dev)

    if a.chi_proxy:
        net = a.chi_proxy
        if a.chi_proxy not in ("vis", "tb"):
            net = Network().to(dev)
            ck = "checkpoints.pt" if a.chi_proxy == "sage" else f"{a.chi_proxy}.pt"
            sd = torch.load(STUDENT / ck, map_location="cpu", weights_only=True)
            if any(k.startswith("dau.") for k in sd):
                net = DauTron(net).to(dev)
            net.load_state_dict(sd)
        print(f"proxy Dev ({a.proxy}) {a.chi_proxy}: {cham_proxy(net, b0_proxy, dev_data, dev):.2f}")
        return

    print("Nạp 1.045 ảnh train ...", flush=True)
    df, Y, CC, IR, LAB = nap_anh("train")
    if a.kiem_thay:
        chia = pd.read_csv(NUA / "chia_nua.csv", dtype={"image_id": str}).set_index("image_id").nua
        nua = df.image_id.map(chia).to_numpy()
        kq = {}
        for th in "AB":
            m = nap_b0(THAY_RUNS / f"runs_b0_{th}" / f"{a.thay_bt}_b0_s0", dev, len(pd.read_csv(NUA / th / "train.csv")))
            for n in "AB":
                ces = []
                with torch.no_grad():
                    for i in np.flatnonzero(nua == n):
                        y, ir, cc = t(Y[i:i + 1], dev), t(IR[i:i + 1], dev), t(CC[i:i + 1], dev)
                        o = ((y + ir) / 2 * 255).round() / 255
                        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                            lg = m(pixel_values=chuan_hoa(ycrcb_sang_rgb(o, cc))).logits
                        lg = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear")
                        ces.append(F.cross_entropy(lg, torch.from_numpy(LAB[i:i + 1].astype(np.int64)).to(dev),
                                                   ignore_index=IGNORE).item())
                kq[f"thay_{th}_tren_nua_{n}"] = float(np.mean(ces))
                print(f"thầy {th} trên nửa {n} ({'đã học' if th == n else 'CHƯA thấy'}): CE {np.mean(ces):.3f}", flush=True)
        (DAY / "kiem_thay.json").write_text(json.dumps(kq, indent=2), encoding="utf-8")
        return
    assert len(df) == int(os.environ.get("SO_TRAIN", 1045)) and not df.vis_path.str.startswith("test").any()
    N, H, W = Y.shape
    ch, cw = a.crop

    net = Network().to(dev).train()
    if a.init != "none":
        ck = "checkpoints.pt" if a.init == "sage" else f"{a.init}.pt"
        net.load_state_dict(torch.load(STUDENT / ck, map_location="cpu", weights_only=True))
    if a.dau == "tb":
        net = DauTron(net).to(dev).train()
    if a.lam > 0 and a.thay == "cityscapes":
        b0 = ThayCityscapes().to(dev)
    elif a.lam > 0 and a.hai_nua:
        chia = pd.read_csv(NUA / "chia_nua.csv", dtype={"image_id": str}).set_index("image_id").nua
        nua = df.image_id.map(chia).to_numpy()
        assert set(nua) == {"A", "B"}
        n_ab = {k: len(pd.read_csv(NUA / k / "train.csv")) for k in "AB"}
        # ảnh nửa A được chấm bởi thầy học nửa B, và ngược lại
        thay = {"A": nap_b0(THAY_RUNS / "runs_b0_B" / f"{a.thay_bt}_b0_s0", dev, n_ab["B"]),
                "B": nap_b0(THAY_RUNS / "runs_b0_A" / f"{a.thay_bt}_b0_s0", dev, n_ab["A"])}
        print("Thầy chéo:", n_ab, flush=True)
    elif a.lam > 0:
        b0 = nap_b0(B0_RUNS / a.dan_duong, dev)
    else:
        b0 = None
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    so_batch = N // a.batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * so_batch, eta_min=a.lr / 100)
    kx, ky = SOBEL_X.to(dev), SOBEL_Y.to(dev)

    p0 = cham_proxy(net, b0_proxy, dev_data, dev)
    print(f"proxy Dev lúc đầu: {p0:.2f}", flush=True)
    log, t0, tot = [], time.time(), (-1, None)
    for ep in range(1, a.epochs + 1):
        if a.lam > 0 and a.hai_nua:   # mỗi batch chỉ gồm ảnh của một nửa, xen kẽ hai nửa
            pa, pb = rng.permutation(np.flatnonzero(nua == "A")), rng.permutation(np.flatnonzero(nua == "B"))
            ba = [pa[i:i + a.batch] for i in range(0, len(pa) - a.batch + 1, a.batch)]
            bb = [pb[i:i + a.batch] for i in range(0, len(pb) - a.batch + 1, a.batch)]
            cac_batch = [x for cap in zip(ba, bb) for x in cap] + ba[len(bb):] + bb[len(ba):]
        else:
            thu_tu = rng.permutation(N)
            cac_batch = [thu_tu[bi * a.batch:(bi + 1) * a.batch] for bi in range(so_batch)]
        s = dict(loss=0., noidung=0., ce=0.)
        for idx in cac_batch:
            if a.lam > 0 and a.hai_nua:
                b0 = thay[nua[idx[0]]]
            i0 = rng.integers(0, H - ch + 1, len(idx))
            j0 = rng.integers(0, W - cw + 1, len(idx))
            lat = rng.random(len(idx)) < 0.5
            cat = lambda A, k, i, j, l: (A[k, i:i + ch, j:j + cw][:, ::-1] if l else A[k, i:i + ch, j:j + cw])
            y = t(np.stack([cat(Y, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]), dev)
            ir = t(np.stack([cat(IR, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]), dev)
            cc = t(np.stack([cat(CC, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]), dev)
            lab = torch.from_numpy(np.stack([cat(LAB, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]).astype(np.int64)).to(dev)

            o = net(y, ir)[0]
            if a.dau == "tb" and a.lam > 0:
                # M4: chỉ giữ ảnh không quá xa TB (thay cho loss nội dung kiểu SAGE kéo về 60/40)
                l_fuse = 5 * F.mse_loss(o, (y + ir) / 2)
                noidung = 3 * l_fuse
            l_fuse = 5 * ((1 - a.w_ir) * F.mse_loss(o, y) + a.w_ir * F.mse_loss(o, ir))   # w_ir 0.4 -> 3*Y + 2*IR như SAGE
            l_grad = 9 * a.w_grad * F.l1_loss(grad(o, kx, ky), torch.maximum(grad(y, kx, ky), grad(ir, kx, ky)))
            if not (a.dau == "tb" and a.lam > 0):
                noidung = 3 * l_fuse + 3 * l_grad
            ce = torch.zeros((), device=dev)
            if b0 is not None:
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    lg = b0(pixel_values=chuan_hoa(ycrcb_sang_rgb(o.clamp(0, 1), cc))).logits
                lg = F.interpolate(lg.float(), size=lab.shape[-2:], mode="bilinear", align_corners=False)
                ce = F.cross_entropy(lg, lab, ignore_index=IGNORE)
            loss = noidung + a.lam * ce
            if not torch.isfinite(loss):
                sys.exit(f"loss không hữu hạn ở epoch {ep}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            s["loss"] += loss.item(); s["noidung"] += noidung.item(); s["ce"] += ce.item()
        pd_ = cham_proxy(net, b0_proxy, dev_data, dev) if (ep % 5 == 0 or ep == a.epochs) else float("nan")
        dong = {"epoch": ep, **{k: v / len(cac_batch) for k, v in s.items()}, "proxy_dev": pd_,
                "phut": (time.time() - t0) / 60}
        log.append(dong)
        print(f"epoch {ep:3d} loss {dong['loss']:.4f} (nội dung {dong['noidung']:.4f}, CE {dong['ce']:.4f})"
              + (f" proxy Dev {pd_:.2f}" if not math.isnan(pd_) else "") + f"  {dong['phut']:.1f} phút", flush=True)
    torch.save(net.state_dict(), STUDENT / f"{a.ten}.pt")
    pd.DataFrame(log).to_csv(STUDENT / f"{a.ten}_log.csv", index=False)
    (STUDENT / f"{a.ten}_proxy.json").write_text(json.dumps(
        vars(a) | {"proxy_dau": p0, "proxy_cuoi": log[-1]["proxy_dev"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Lưu {STUDENT / (a.ten + '.pt')}  | proxy Dev {p0:.2f} -> {log[-1]['proxy_dev']:.2f}")


if __name__ == "__main__":
    main()
