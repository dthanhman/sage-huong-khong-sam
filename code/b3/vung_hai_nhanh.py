"""VÙNG TỪ hai_nhanh – KIỂM SẠCH (22/11 khuya). CHẨN ĐOÁN, chưa phải phương pháp.

CÂU HỎI: hiểu biết vùng của mô hình thấy VIS + IR riêng (hai_nhanh), chuyển qua MỘT ảnh ghép, có làm B0 (chỉ nhìn ảnh ghép,
đúng giao thức) tốt hơn nosam KHÔNG CHỈ ở 10 lớp mà cả ĐỦ 14 LỚP không?
Lần thử nhanh (oracle_doan.py --nguon hai_nhanh): 10 lớp +1,19 vs nosam nhưng đủ 14 lớp thua (bus −16, đèn −8) vì
(1) nền trộn tuyến tính thay vì nosam, (2) nhãn cứng, (3) mô hình chấm học trên ảnh nhãn THẬT, chấm trên ảnh nhãn ĐOÁN.

SỬA (đặt trước, không chỉnh theo dev):
  - Nền = Y của nosam_kp841. Chỉ sửa ở vùng mục tiêu, theo XÁC SUẤT MỀM của hai_nhanh (softmax), làm mờ Gauss 4 px:
        Y = Y_nosam + p_nguoi·(R_ir − Y_nosam) + p_vis·(R_vis − Y_nosam)
        R_ir  = 0,3·Y_vis + 0,7·IR   (person – IR mạnh hơn, như w_lop 0,7)
        R_vis = 0,9·Y_vis + 0,1·IR   (lớp VIS hơn nosam ≥ +2 trên B0 val151 s0: vỉa hè, đèn, truck; KHÔNG bus vì nosam > VIS)
    Bản đối chứng `vhn_nguoc`: đổi vai R_ir <-> R_vis (cùng vùng, sai luật).
  - Nhãn đoán CHÉO: train chia 2 nửa theo khối 50 ảnh liền nhau xen kẽ, bỏ 2 ảnh đệm mỗi đầu khối khỏi CẢ HAI nửa;
    mô hình A (học nửa A) đoán nửa B và ngược lại; ảnh đệm và dev dùng TRUNG BÌNH xác suất của A và B.
    -> mô hình chấm học và bị chấm trên ảnh dựng từ nhãn đoán cùng loại (hết lệch train/dev).

Bước:
  python vung_hai_nhanh.py chia --ra <thư mục>          -> <ra>/man_A, <ra>/man_B (train = nửa, dev/test giữ nguyên), <ra>/nua.csv
  (train hai_nhanh B0 trên man_A, man_B: SAGE_MAN=<ra>/man_X SAGE_RUNS=<ra>/runs_X python train_seg.py --bien_the hai_nhanh --model b0 ...)
  python vung_hai_nhanh.py dung --ra <thư mục>          -> ảnh ANH/vhn, ANH/vhn_nguoc cho mọi ảnh train + dev (cần ANH/nosam_kp841)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from chung import ANH, MAN, duong, thu_muc

KHOI, DEM = 50, 2
LOP_NGUOI = [7]          # chỉ số 0..13 = lớp FMB 1..14: person = 8
LOP_VIS = [1, 3, 9]      # sidewalk (2), traffic_light (4), truck (10)
W_IR, W_VIS, SIGMA = 0.7, 0.1, 4.0


def chia(ra):
    df = pd.read_csv(MAN / "train.csv", dtype={"image_id": str})
    df = df.sort_values("image_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    vt = np.arange(len(df))
    k = vt // KHOI                                    # khối theo VỊ TRÍ trong danh sách id tăng dần
    trong = vt % KHOI
    nua = np.where(k % 2 == 0, "A", "B").astype(object)
    nua[(trong < DEM) | (trong >= KHOI - DEM)] = "dem"
    nua[(k == 0) & (trong < DEM)] = "A"               # đầu danh sách không giáp khối nào
    nua[(k == k.max()) & (trong >= KHOI - DEM)] = "A" if k.max() % 2 == 0 else "B"
    df["nua"] = nua
    ra.mkdir(parents=True, exist_ok=True)
    df[["image_id", "nua"]].to_csv(ra / "nua.csv", index=False)
    for x in ("A", "B"):
        d = ra / f"man_{x}"
        d.mkdir(exist_ok=True)
        df[df.nua == x].drop(columns="nua").to_csv(d / "train.csv", index=False)
        for s in ("dev", "test"):
            (d / f"{s}.csv").write_bytes((MAN / f"{s}.csv").read_bytes())
    dem = df.nua.value_counts().to_dict()
    a, b = set(df.image_id[df.nua == "A"]), set(df.image_id[df.nua == "B"])
    dev = set(pd.read_csv(MAN / "dev.csv", dtype={"image_id": str}).image_id)
    assert not (a & b) and not ((a | b) & dev), "nửa A/B/dev trùng nhau"
    # ảnh liền kề (khoảng cách id ≤ 1) giữa A và B: phải bằng 0 nhờ ảnh đệm
    ia, ib = np.array(sorted(map(int, a))), np.array(sorted(map(int, b)))
    gan = int((np.abs(ia[:, None] - ib[None, :]) <= 1).sum())
    print(f"chia: {dem} | A∩B = 0, (A∪B)∩dev = 0 | cặp A–B id liền nhau: {gan}", flush=True)
    assert gan == 0


MODEL_VUNG = "b0"        # (23/11) cỡ mô hình hai_nhanh của thầy: --model b0 (gốc) | b3 (thầy mạnh hơn)
TEN_RA = "vhn"           # tên biến thể ảnh ra: --ten (b3 -> vhn3)


def nap(run):
    from train_seg import tao_model
    m = tao_model(MODEL_VUNG, 14, 3, "hai_nhanh")
    m.load_state_dict(torch.load(run / "last.pt", map_location="cpu", weights_only=True))
    return m.cuda().eval()


@torch.inference_mode()
def xac_suat(m, x):
    with torch.autocast("cuda", dtype=torch.float16):
        lg = m(pixel_values=x[None].cuda()).logits
    lg = torch.nn.functional.interpolate(lg.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
    return lg.softmax(1)[0].cpu().numpy()          # 14 x H x W


def ghep(y_n, y_v, ir, p_ng, p_vis, nguoc):
    r_ir = (1 - W_IR) * y_v + W_IR * ir
    r_vis = (1 - W_VIS) * y_v + W_VIS * ir
    if nguoc:
        r_ir, r_vis = r_vis, r_ir
    return y_n + p_ng * (r_ir - y_n) + p_vis * (r_vis - y_n)


def dung(ra, gioi_han=0):
    from train_seg import FMBSeg
    nua = pd.read_csv(ra / "nua.csv", dtype={"image_id": str}).set_index("image_id").nua
    mA, mB = (nap(ra / f"runs_{x}" / f"hai_nhanh_{MODEL_VUNG}_s0") for x in "AB")
    ds = FMBSeg(["train", "dev"], "hai_nhanh", False)
    if gioi_han:                                      # chỉ để chạy thử: N ảnh train đầu + N ảnh dev đầu
        ds.df = pd.concat([ds.df[ds.df.split == "train"].head(gioi_han), ds.df[ds.df.split == "dev"].head(gioi_han)])
    df = ds.df
    n = {"A->B": 0, "B->A": 0, "TB": 0}
    tk = {"p_nguoi": 0.0, "p_vis": 0.0, "dY_vhn": 0.0, "px": 0}
    for i in range(len(df)):
        r = df.iloc[i]
        x, _ = ds[i]
        h = nua.get(r.image_id, "dev")
        if h == "A":                                      # A đã học ảnh này -> dùng B
            p = xac_suat(mB, x); n["B->A"] += 1
        elif h == "B":
            p = xac_suat(mA, x); n["A->B"] += 1
        else:                                             # ảnh đệm, dev
            p = (xac_suat(mA, x) + xac_suat(mB, x)) / 2; n["TB"] += 1
        p_ng = cv2.GaussianBlur(p[LOP_NGUOI].sum(0).astype(np.float32), (0, 0), SIGMA)
        p_vis = cv2.GaussianBlur(p[LOP_VIS].sum(0).astype(np.float32), (0, 0), SIGMA)
        vis = cv2.imread(str(duong(r.vis_path)))
        ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
        nos = cv2.imread(str(ANH / "nosam_kp841" / thu_muc(r) / f"{r.image_id}.png"))
        assert nos is not None, f"thiếu ảnh nosam_kp841 {r.image_id}: chạy xuat_anh.py --bien_the nosam_kp841 trước"
        y_n = cv2.cvtColor(nos, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        for bt, ng in ((TEN_RA, False), (TEN_RA + "_nguoc", True)):
            yf = ghep(y_n, ycc[:, :, 0].astype(np.float32), ir, p_ng, p_vis, ng).round().clip(0, 255).astype(np.uint8)
            if not ng:
                tk["dY_vhn"] += float(np.abs(yf.astype(np.float32) - y_n).sum())
            out = ANH / bt / thu_muc(r) / f"{r.image_id}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
        tk["p_nguoi"] += float(p_ng.sum()); tk["p_vis"] += float(p_vis.sum()); tk["px"] += p_ng.size
    px = tk.pop("px")
    print(f"dung: {len(df)} ảnh, nguồn xác suất {n} | TB p_nguoi {tk['p_nguoi'] / px:.4f}, "
          f"p_vis {tk['p_vis'] / px:.4f}, |Y_vhn − Y_nosam| TB {tk['dY_vhn'] / px:.3f}", flush=True)


def luu_p(ra, gioi_han=0):
    """(23/11) Lưu bản đồ vùng THẦY (p_nguoi, p_vis đã mờ, y như dung) ở 1/4 độ phân giải, uint8 (x255),
    cho Student S-vùng học thẳng bản đồ. Cùng nguồn chéo như dung: A đoán nửa B, B đoán nửa A, đệm + dev = TB(A,B).
    Ra: <ra>/p_thay.npz {image_id, p (N x 2 x H/4 x W/4 uint8), nguon}."""
    from train_seg import FMBSeg
    nua = pd.read_csv(ra / "nua.csv", dtype={"image_id": str}).set_index("image_id").nua
    mA, mB = (nap(ra / f"runs_{x}" / f"hai_nhanh_{MODEL_VUNG}_s0") for x in "AB")
    ds = FMBSeg(["train", "dev"], "hai_nhanh", False)
    if gioi_han:
        ds.df = pd.concat([ds.df[ds.df.split == "train"].head(gioi_han), ds.df[ds.df.split == "dev"].head(gioi_han)])
    ids, P, ng = [], [], []
    for i in range(len(ds.df)):
        r = ds.df.iloc[i]
        x, _ = ds[i]
        h = nua.get(r.image_id, "dev")
        p = xac_suat(mB, x) if h == "A" else xac_suat(mA, x) if h == "B" else (xac_suat(mA, x) + xac_suat(mB, x)) / 2
        p_ng = cv2.GaussianBlur(p[LOP_NGUOI].sum(0).astype(np.float32), (0, 0), SIGMA)
        p_vis = cv2.GaussianBlur(p[LOP_VIS].sum(0).astype(np.float32), (0, 0), SIGMA)
        H, W = p_ng.shape
        q = [cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA) for m in (p_ng, p_vis)]
        P.append((np.stack(q).clip(0, 1) * 255).round().astype(np.uint8)); ids.append(r.image_id); ng.append(h)
    ra_p = ra / ("p_thay.npz" if TEN_RA == "vhn" else f"p_thay_{TEN_RA}.npz")
    np.savez_compressed(ra_p, image_id=np.array(ids), p=np.stack(P), nguon=np.array(ng))
    print(f"luu_p: {len(ids)} ảnh -> {ra_p}; TB p_nguoi {np.stack(P)[:, 0].mean() / 255:.4f}, "
          f"p_vis {np.stack(P)[:, 1].mean() / 255:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buoc", choices=["chia", "dung", "luu_p"])
    ap.add_argument("--ra", required=True)
    ap.add_argument("--gioi_han", type=int, default=0, help="chỉ để chạy thử")
    ap.add_argument("--model", default="b0", help="cỡ hai_nhanh của thầy (thư mục runs_X/hai_nhanh_<model>_s0)")
    ap.add_argument("--ten", default="vhn", help="tên biến thể ảnh ra (mặc định vhn = như cũ)")
    a = ap.parse_args()
    global MODEL_VUNG, TEN_RA
    MODEL_VUNG, TEN_RA = a.model, a.ten
    {"chia": lambda: chia(Path(a.ra)), "dung": lambda: dung(Path(a.ra), a.gioi_han),
     "luu_p": lambda: luu_p(Path(a.ra), a.gioi_han)}[a.buoc]()


if __name__ == "__main__":
    sys.exit(main())
