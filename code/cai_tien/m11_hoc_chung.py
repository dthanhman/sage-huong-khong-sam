"""M11 – Student SAGE (KHÔNG SAM) học CÙNG LÚC với SegFormer-B0, có rào chắn loss nội dung.

Rút kinh nghiệm:
  M1  : CE qua thầy B0 ĐÓNG BĂNG -> thầy thuộc lòng/bị "lừa"            -> ở đây B0 học cùng (SeAFusion, MRFS, TDFusion).
  M4/6: khởi tạo từ Student SAGE (có SAM)                                -> ở đây khởi tạo nosam_1045 (không SAM, không thấy Dev).
  M4/5: trộn thẳng w*VIS + (1-w)*IR                                       -> ở đây mạng tự sinh ảnh (TDFusion ablation V).
  TDFusion ablation III: CE thẳng vào mạng ghép làm chỉ số ghép giảm     -> giữ loss nội dung của SAGE (beta) làm rào chắn,
                                                                            kiểm bằng cổng 2.
Loss Student = CE(B0(ảnh ghép), nhãn) * lam + beta * L_noi_dung
  L_noi_dung = 3 * 5 * [(1-w_ir) MSE(o, Y_vis) + w_ir MSE(o, IR)] + 3 * 9 * L1(|∇o|, max(|∇Y_vis|, |∇IR|))
               (w_ir = 0,4 -> đúng hệ số 3/2 của SAGE gốc, như train_student_nosam.py)
B0 học bằng CE. warm_seg epoch đầu: Student đứng yên (B0 còn ngẫu nhiên thì CE chưa có nghĩa).

M11-SCD (18/11, sau khi m11_b1 rớt cổng 2 ở SCD 85%): thêm vào loss Student
  + gamma * (2 - SCD(o))     SCD = r(o - IR, Y_vis) + r(o - Y_vis, IR), tương quan Pearson trên cả crop từng ảnh
                             (đúng công thức chi_so_ghep.py). Trộn tuyến tính w*VIS+(1-w)*IR có SCD KHÔNG ĐỔI theo w
                             (tb/ir30/ir70 đều 1,399) -> tăng w_ir vô ích; phải ép giữ cả hai nguồn kiểu cộng.
  + eta * L_chay             chống cháy trắng (nhược điểm của M8c vci). SCD bất biến khi cộng hằng số nên mạng
                             được tự do kéo độ sáng xuống mà không mất SCD; L_chay chỉ phạt phần vượt ngưỡng:
      L_chay = mean(relu(o - tran)^2) + mean(relu(o - max(Y_vis, IR) - le)^2)
               (tran = 0,92 ≈ 235/255: vùng sắp cháy; vế 2: chỗ đã sáng không bị đẩy sáng thêm quá le
                -> giữ màu CrCb của biển báo, chỗ M8c/M8d bị bạc)
  gamma = eta = 0 -> đúng M11 cũ (m11_b1).
M11 sửa (18/11 khuya, sau chẩn đoán: lam 5 cân lực mà ảnh vẫn ít đổi -> nghi B0 học cùng tự thích nghi):
  --b0_st F  : nhân lr của B0 với F khi Student đã học (sau warm_seg). 1 = cũ; 0 = B0 ĐÓNG BĂNG lúc Student học
               (B0 mới học warm_seg epoch, chưa thuộc ảnh như thầy đóng băng của M1) -> Student phải đổi ảnh để giảm CE.
  --bn_dong  : (19/11) BatchNorm Student ở chế độ eval khi train – sửa lệch train/eval (xem st_train).
  --nen tb   : nền B. Ảnh = clamp(TB + s * Student), rào chắn = w_tb * MSE(ảnh, TB) + w_g * L1 gradient
               (đúng loss tb_sua.py); --init là checkpoint tb_sua (có khóa _tb_scale). Lưu kèm _tb_scale cho xuat_anh.
Chỉ 1.045 ảnh train. Đánh giá thật: xuat_anh.py --bien_the <ten> -> cổng 2 (chi_so_ghep.py) -> cổng 3 (B0 MỚI, Kaggle).

Ví dụ: python m11_hoc_chung.py --thu 30                 (đo tốc độ, bộ nhớ)
       python m11_hoc_chung.py --ten m11_b1 --beta 1 --epochs 40
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DAY = Path(__file__).resolve().parent
sys.path.insert(0, str(DAY))
from train_student_seg import (SOBEL_X, SOBEL_Y, Network, STUDENT, chuan_hoa, grad, nap_anh, t,  # noqa: E402
                               ycrcb_sang_rgb)
from chung import IGNORE, NC, cac_miou, nham  # noqa: E402


def noi_dung(o, y, ir, kx, ky, w_ir):
    l_fuse = 5 * ((1 - w_ir) * F.mse_loss(o, y) + w_ir * F.mse_loss(o, ir))
    l_grad = 9 * F.l1_loss(grad(o, kx, ky), torch.maximum(grad(y, kx, ky), grad(ir, kx, ky)))
    return 3 * l_fuse + 3 * l_grad


def _tuong_quan(a, b, eps=1e-8):
    """Pearson từng ảnh trên cả crop. a, b: Bx1xHxW -> B."""
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / (a.square().sum(1).sqrt() * b.square().sum(1).sqrt() + eps)


def scd(o, y, ir):
    """SCD khả vi, giống sum_of_correlation_differences (pr6_eval). Trả B giá trị trong [-2, 2]."""
    return _tuong_quan(o - y, ir) + _tuong_quan(o - ir, y)


def chay(o, y, ir, tran, le):
    """Phạt cháy trắng: vượt trần tuyệt đối + vượt nguồn sáng hơn quá le."""
    return F.relu(o - tran).square().mean() + F.relu(o - torch.maximum(y, ir) - le).square().mean()


NEN = {"ten": "nosam", "s": 0.5}   # đặt trong main theo --nen
BN_DONG = [False]                    # --bn_dong: BatchNorm của Student luôn ở chế độ eval


def st_train(net):
    """Student về chế độ học. --bn_dong: BatchNorm dùng thống kê đã lưu (như lúc xuất ảnh), không cập nhật.
    Lý do (19/11, scene_v1/kiem_bn.py): tbsua_w150 crop 384x512 train-mode lệch eval 1,5 mức xám, rào 1,43 vs 0,74
    -> Student mất công sửa lệch BatchNorm thay vì học theo CE, và ảnh xuất (eval) khác ảnh lúc train."""
    net.train()
    if BN_DONG[0]:
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    return net


def tao_anh(net, y, ir):
    """Ảnh ghép (kênh Y). nosam: đầu ra Student. tb: TB + s * Student, kẹp [0, 1] như tb_sua.py."""
    r = net(y, ir)[0]
    return ((y + ir) / 2 + NEN["s"] * r).clamp(0, 1) if NEN["ten"] == "tb" else r


def rao_tb(o, y, ir, kx, ky, w_tb, w_g):
    """Rào chắn nền B, đúng loss tb_sua.py."""
    lg = F.l1_loss(grad(o, kx, ky), torch.maximum(grad(y, kx, ky), grad(ir, kx, ky)))
    return w_tb * F.mse_loss(o, (y + ir) / 2) + w_g * lg


@torch.no_grad()
def cham_dev(net, seg, data, dev):
    """mIoU Dev của chính B0 học chung (chỉ để theo dõi; số chính thức là B0 MỚI train trên ảnh xuất ra)."""
    _, Y, CC, IR, LAB = data
    net.eval(), seg.eval()
    c = np.zeros((NC, NC), np.int64)
    for i in range(len(Y)):
        y, ir, cc = t(Y[i:i + 1], dev), t(IR[i:i + 1], dev), t(CC[i:i + 1], dev)
        o = (tao_anh(net, y, ir).clamp(0, 1) * 255).round() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            lg = seg(pixel_values=chuan_hoa(ycrcb_sang_rgb(o, cc))).logits
        p = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear").argmax(1)[0].cpu().numpy()
        c += nham(LAB[i].astype(np.int64), p)
    st_train(net), seg.train()
    return cac_miou(c)[0]["mIoU_co_mat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", default="m11")
    ap.add_argument("--init", default="nosam_1045", help="checkpoint Student khởi tạo (KHÔNG dùng checkpoints.pt = SAGE có SAM)")
    ap.add_argument("--lam", type=float, default=1.0, help="trọng số CE lên Student")
    ap.add_argument("--beta", type=float, default=1.0, help="trọng số loss nội dung (rào chắn chỉ số ghép)")
    ap.add_argument("--w_ir", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.0, help="trọng số (2 - SCD); 0 = M11 cũ")
    ap.add_argument("--eta", type=float, default=0.0, help="trọng số phạt cháy trắng L_chay; 0 = tắt")
    ap.add_argument("--tran", type=float, default=0.92, help="ngưỡng sáng bắt đầu phạt (0..1)")
    ap.add_argument("--le", type=float, default=0.10, help="được sáng hơn nguồn sáng nhất tối đa bao nhiêu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warm_seg", type=int, default=5, help="số epoch đầu Student đứng yên, chỉ B0 học")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, nargs=2, default=[384, 512])
    ap.add_argument("--lr_seg", type=float, default=6e-5, help="như train_seg (head x10)")
    ap.add_argument("--lr_st", type=float, default=1e-4, help="Student")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thu", type=int, default=0, help="chỉ chạy N batch để đo tốc độ/bộ nhớ")
    ap.add_argument("--do_grad", type=int, default=0,
                    help="chẩn đoán: mỗi N batch đo riêng ||grad|| lên Student của lam*CE và beta*rào chắn (0 = tắt)")
    ap.add_argument("--b0_st", type=float, default=1.0,
                    help="hệ số lr B0 khi Student đã học (1 = cũ, 0 = đóng băng B0 lúc đó)")
    ap.add_argument("--bn_dong", action="store_true",
                    help="khóa BatchNorm Student ở chế độ eval khi train (khớp lúc xuất ảnh); mặc định = hành vi cũ")
    ap.add_argument("--nen", choices=["nosam", "tb"], default="nosam", help="nền A (nosam) hoặc nền B (TB + phần sửa)")
    ap.add_argument("--w_tb", type=float, default=150.0, help="nền tb: trọng số MSE(ảnh, TB)")
    ap.add_argument("--w_g", type=float, default=27.0, help="nền tb: trọng số loss gradient")
    a = ap.parse_args()
    assert a.init != "checkpoints", "cấm khởi tạo từ Student SAGE (có SAM) – xem cai_tien/README.md"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"

    print("Nạp ảnh ...", flush=True)
    dev_data = nap_anh("dev")
    df, Y, CC, IR, LAB = nap_anh("train")
    assert len(df) == int(os.environ.get("SO_TRAIN", 1045)) and not df.vis_path.str.startswith("test").any()
    N, H, W = Y.shape
    ch, cw = a.crop

    net = Network()
    sd0 = torch.load(STUDENT / f"{a.init}.pt", map_location="cpu", weights_only=True)
    s_tb = sd0.pop("_tb_scale", None)
    assert (s_tb is not None) == (a.nen == "tb"), "--nen tb cần --init là checkpoint tb_sua (có _tb_scale), và ngược lại"
    NEN["ten"] = a.nen
    if s_tb is not None:
        NEN["s"] = float(s_tb)
    net.load_state_dict(sd0)
    BN_DONG[0] = a.bn_dong
    net = st_train(net.to(dev))
    from transformers import SegformerForSemanticSegmentation
    seg = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=NC, id2label={i: str(i) for i in range(NC)},
        label2id={str(i): i for i in range(NC)}, ignore_mismatched_sizes=True).to(dev).train()
    kx, ky = SOBEL_X.to(dev), SOBEL_Y.to(dev)

    so_batch = N // a.batch
    tong = a.epochs * so_batch
    enc = [p for n, p in seg.named_parameters() if n.startswith("segformer.")]
    head = [p for n, p in seg.named_parameters() if not n.startswith("segformer.")]
    opt_seg = torch.optim.AdamW([{"params": enc, "lr": a.lr_seg}, {"params": head, "lr": a.lr_seg * 10}],
                                weight_decay=0.01)
    warm = max(1, tong // 10)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt_seg, lambda s: ((s + 1) / warm if s < warm else max(0.0, 1 - (s - warm) / max(1, tong - warm)))
        * (a.b0_st if s >= a.warm_seg * so_batch else 1.0))
    opt_st = torch.optim.Adam(net.parameters(), lr=a.lr_st)
    scaler = torch.amp.GradScaler()

    log, t0, buoc = [], time.time(), 0
    for ep in range(1, a.epochs + 1):
        hoc_st = ep > a.warm_seg
        thu_tu = rng.permutation(N)
        s = dict(ce=0., nd=0., scd=0., chay=0., ti_le_chay=0.)
        gd = []   # (||g_CE||, ||g_rào||, cos) khi --do_grad
        for bi in range(so_batch):
            idx = thu_tu[bi * a.batch:(bi + 1) * a.batch]
            i0 = rng.integers(0, H - ch + 1, len(idx))
            j0 = rng.integers(0, W - cw + 1, len(idx))
            lat = rng.random(len(idx)) < 0.5
            cat = lambda A, k, i, j, l: (A[k, i:i + ch, j:j + cw][:, ::-1] if l else A[k, i:i + ch, j:j + cw])
            y = t(np.stack([cat(Y, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]), dev)
            ir = t(np.stack([cat(IR, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]), dev)
            cc = t(np.stack([cat(CC, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)]), dev)
            lab = torch.from_numpy(np.stack([cat(LAB, k, i, j, l) for k, i, j, l in zip(idx, i0, j0, lat)])
                                   .astype(np.int64)).to(dev)

            with torch.set_grad_enabled(hoc_st):
                o = tao_anh(net, y, ir)
                nd = (rao_tb(o, y, ir, kx, ky, a.w_tb, a.w_g) if a.nen == "tb"
                      else noi_dung(o, y, ir, kx, ky, a.w_ir))
                sc = scd(o, y, ir).mean()
                ch_ = chay(o, y, ir, a.tran, a.le)
                rao = nd + a.gamma * (2 - sc) + a.eta * ch_    # rào chắn chỉ số ghép, nhân beta
            vao = o if hoc_st else o.detach()
            with torch.autocast("cuda", dtype=torch.float16):
                lg = seg(pixel_values=chuan_hoa(ycrcb_sang_rgb(vao.clamp(0, 1), cc))).logits
            lg = F.interpolate(lg.float(), size=lab.shape[-2:], mode="bilinear", align_corners=False)
            ce = F.cross_entropy(lg, lab, ignore_index=IGNORE)
            # B0 học bằng CE; Student học bằng lam*CE + beta*nội dung (cùng một lần lan ngược)
            loss = ce * (a.lam if hoc_st else 1.0) + (a.beta * rao if hoc_st else 0.0)
            if not torch.isfinite(loss):
                sys.exit(f"loss không hữu hạn ở epoch {ep}")
            if hoc_st and a.do_grad and bi % a.do_grad == 0:   # chẩn đoán, không đổi việc train
                ps = [q for q in net.parameters() if q.requires_grad]
                S = 4096.0   # nhân rồi chia lại: tránh tràn dưới FP16 trong B0 (như GradScaler)
                g1 = torch.autograd.grad(S * a.lam * ce, ps, retain_graph=True, allow_unused=True)
                g2 = torch.autograd.grad(S * a.beta * rao, ps, retain_graph=True, allow_unused=True)
                v1 = torch.cat([(x if x is not None else torch.zeros_like(q)).flatten() for x, q in zip(g1, ps)])
                v2 = torch.cat([(x if x is not None else torch.zeros_like(q)).flatten() for x, q in zip(g2, ps)])
                gd.append((v1.norm().item() / S, v2.norm().item() / S, F.cosine_similarity(v1, v2, dim=0).item()))
            opt_seg.zero_grad(), opt_st.zero_grad()
            scaler.scale(loss).backward()
            if hoc_st and a.lam != 1.0:          # B0 luôn nhận đúng gradient của CE (không nhân lam)
                scaler.unscale_(opt_seg)
                for g in opt_seg.param_groups:
                    for p in g["params"]:
                        if p.grad is not None:
                            p.grad.div_(a.lam)
            scaler.step(opt_seg)
            if hoc_st:
                scaler.step(opt_st)
            scaler.update()
            sch.step()
            buoc += 1
            s["ce"] += ce.item(); s["nd"] += nd.item(); s["scd"] += sc.item(); s["chay"] += ch_.item()
            s["ti_le_chay"] += (o.detach() >= 250 / 255).float().mean().item()
            if a.thu and buoc >= a.thu:
                print(f"thử {buoc} batch: CE {ce.item():.4f} nội dung {nd.item():.4f} SCD {sc.item():.3f} "
                      f"(gamma*(2-SCD) {a.gamma * (2 - sc.item()):.4f}) L_chay {ch_.item():.5f} "
                      f"(eta*L_chay {a.eta * ch_.item():.4f}) | {(time.time() - t0) / buoc:.2f} s/batch, "
                      f"GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
                      f"ước {tong * (time.time() - t0) / buoc / 3600:.2f} h cho {a.epochs} epoch")
                return
        m = cham_dev(net, seg, dev_data, dev) if (ep % 5 == 0 or ep == a.epochs) else float("nan")
        dong = {"epoch": ep, "hoc_st": hoc_st, **{k: v / so_batch for k, v in s.items()},
                "dev_miou_seg_chung": m, "phut": (time.time() - t0) / 60,
                **({"g_ce": np.mean([x[0] for x in gd]), "g_rao": np.mean([x[1] for x in gd]),
                    "cos_ce_rao": np.mean([x[2] for x in gd])} if gd else {})}
        log.append(dong)
        print(f"epoch {ep:3d} {'ST+B0' if hoc_st else 'B0   '} CE {dong['ce']:.4f} nội dung {dong['nd']:.4f}"
              f" SCD {dong['scd']:.3f} cháy {100 * dong['ti_le_chay']:.2f}%"
              + (f" | Dev (B0 học chung) {m:.2f}" if m == m else "")
              + (f" | g_CE {dong['g_ce']:.2e} g_rào {dong['g_rao']:.2e} tỉ lệ {dong['g_ce'] / dong['g_rao']:.3f}"
                 f" cos {dong['cos_ce_rao']:+.2f}" if gd else "") + f"  {dong['phut']:.1f} phút", flush=True)
    sd = net.state_dict()
    if a.nen == "tb":
        sd["_tb_scale"] = torch.tensor(NEN["s"])   # để xuat_anh.py dựng lại TB + s * Student
    torch.save(sd, STUDENT / f"{a.ten}.pt")
    pd.DataFrame(log).to_csv(STUDENT / f"{a.ten}_log.csv", index=False)
    (STUDENT / f"{a.ten}_cfg.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    print(f"Lưu {STUDENT / (a.ten + '.pt')}. Tiếp: b3/xuat_anh.py --bien_the {a.ten} -> chi_so_ghep.py -> B0 mới.")


if __name__ == "__main__":
    main()
