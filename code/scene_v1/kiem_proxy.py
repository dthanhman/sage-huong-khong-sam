"""Kiểm phép đo RẺ thay B0 train lại (20/11): B0 đã train sẵn (đóng băng, "giám khảo") dự đoán trên ảnh ghép mới.

Nếu thứ hạng mIoU theo giám khảo đóng băng KHỚP thứ hạng mIoU thật (B0 train lại trên chính ảnh đó, seed 0)
trên các biến thể đã có -> dùng được để sàng lọc nhanh các núm của A2 (vài phút/cấu hình thay vì 1h15).
Tránh thiên vị "sân nhà": giám khảo không chấm biến thể nó được train trên (bỏ cặp đó khi lấy trung bình).
Số: mIoU 10 lớp >= 20 ảnh trên val 151 (lớp cố định từ GT), chạy CPU để không tranh GPU.

Chạy: cd b3 && SAGE_MAN=../scene_v1/kham_pha SAGE_ANH=.../seg/anh python ../scene_v1/kiem_proxy.py
       thêm biến thể mới để chấm (không cần mIoU thật): --chi_cham nosam_kp841_vung10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import NC, nham  # noqa: E402
from train_seg import FMBSeg, tao_model  # noqa: E402
import torch.nn.functional as F  # noqa: E402

GOC = Path(__file__).resolve().parent
NHA, GOP = GOC / "runs_b0", GOC / "runs_b0_gop"
RA = GOC / "phan_tich_nosam" / "proxy"
GIAM_KHAO = ["nosam_kp841", "m11b_l15_e30_bn", "tdfusion"]
BIEN_THE = ["nosam_kp841", "nosam_kp841_w05", "nosam_kp841_w06", "nosam_kp841_w07", "m11a_l5_e30_bn",
            "m11a_l0_e30_bn", "m11b_l15_e30_bn", "tdfusion", "m11_kp", "tbsua_w150", "sage", "tb"]


def miou10(conf, lop):
    c = conf.sum(0).astype(np.float64)
    tp = np.diag(c)
    return float(np.mean(tp[lop] / (c.sum(1)[lop] + c.sum(0)[lop] - tp[lop])) * 100)


def that(bt):
    for d in (NHA, GOP):
        p = d / f"{bt}_b0_s0" / "dev_nham_last.npz"
        if p.exists():
            return np.load(p)["conf"], ("nha" if d == NHA else "kaggle")
    return None, None


@torch.inference_mode()
def cham(model, bt):
    ds = FMBSeg(["dev"], bt, False)
    conf = np.zeros((len(ds), NC, NC), np.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        p = F.interpolate(model(pixel_values=x[None]).logits, size=x.shape[-2:], mode="bilinear",
                          align_corners=False).argmax(1)[0].numpy()
        conf[i] = nham(y.numpy(), p, NC)
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi_cham", nargs="*", default=[], help="biến thể mới chỉ cần điểm giám khảo")
    a = ap.parse_args()
    torch.set_num_threads(10)
    RA.mkdir(parents=True, exist_ok=True)
    c0, _ = that("nosam_kp841")
    lop = np.where((c0.sum(2) > 0).sum(0) >= 20)[0]

    bts = a.chi_cham or BIEN_THE
    diem = {}
    for gk in GIAM_KHAO:
        model = tao_model("b0", NC)
        model.load_state_dict(torch.load(NHA / f"{gk}_b0_s0" / "last.pt", map_location="cpu", weights_only=True))
        model.eval()
        for bt in bts:
            f = RA / f"{gk}__{bt}.json"
            if f.exists():
                diem[(gk, bt)] = json.loads(f.read_text())["miou10"]
                continue
            diem[(gk, bt)] = miou10(cham(model, bt), lop)
            f.write_text(json.dumps({"miou10": diem[(gk, bt)]}))
            print(f"  giám khảo {gk:18s} chấm {bt:22s}: {diem[(gk, bt)]:.2f}", flush=True)

    rows = []
    for bt in bts:
        r = {"bien_the": bt}
        for gk in GIAM_KHAO:
            r[f"gk_{gk}"] = diem[(gk, bt)]
        r["proxy"] = np.mean([diem[(gk, bt)] for gk in GIAM_KHAO if gk != bt])   # bỏ "sân nhà"
        c, nguon = that(bt)
        r["that_s0"], r["may"] = (miou10(c, lop), nguon) if c is not None else (np.nan, "")
        rows.append(r)
    d = pd.DataFrame(rows).set_index("bien_the")
    pd.set_option("display.width", 200)
    print(d.round(2).to_string())
    d.round(3).to_csv(RA / ("proxy_moi.csv" if a.chi_cham else "proxy_vs_that.csv"))
    if not a.chi_cham:
        print(f"\nSpearman proxy ~ mIoU thật (s0, {len(d)} biến thể): {d.proxy.corr(d.that_s0, 'spearman'):+.2f}")
        for gk in GIAM_KHAO:
            m = d.index != gk
            print(f"  riêng giám khảo {gk:18s}: {d.loc[m, f'gk_{gk}'].corr(d.loc[m, 'that_s0'], 'spearman'):+.2f}")
        print("Lưu ý: chênh giữa các biến thể ~±1 điểm, cỡ nhiễu seed của mIoU thật -> tương quan sẽ bị kéo thấp.")


if __name__ == "__main__":
    main()
