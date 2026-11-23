"""ORACLE NHÃN ĐOÁN (22/11) – CHẨN ĐOÁN, không phải phương pháp.
CÂU HỎI: oracle theo lớp (orc_dung 64,83 vs nosam 56,99, B0 s0 val151) có còn lợi khi mask KHÔNG hoàn hảo không?
  Nếu lợi chủ yếu nhờ hình nhãn thật -> Student + đầu phân đoạn tạm (mask tự đoán) sẽ không ăn được -> bỏ.
BƯỚC 1 (script này, không train): dev dùng NHÃN ĐOÁN của B0 nosam_kp841 s0 (last.pt, dev 56,99);
  chấm lại B0 orc_dung / orc_nguoc ĐÃ TRAIN (train vẫn nhãn thật) trên ảnh dev mới.
  Lệch phân phối train (nhãn thật) / dev (nhãn đoán) -> nếu kết quả thấp thì CHƯA kết luận được, cần bước 2
  (nhãn đoán cả train, train lại B0). Nhãn đoán không có lớp 0 (unlabeled) – khác nhỏ so với nhãn thật.
ĐẶT TRƯỚC (10 lớp >= 20 ảnh, s0):
  TÍN HIỆU MẠNH: orc_doan >= nosam + 1,0 (57,99) VÀ orc_doan − orc_doan_nguoc >= +1,0.
  Ngược lại: chưa kết luận -> bước 2.
Chạy: cd b3 && (env như chay_oracle.sh) python ../scene_v1/oracle_doan.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, RUNS, duong, thu_muc  # noqa: E402
from train_seg import FMBSeg, du_doan, so_kenh, tao_model  # noqa: E402
from xuat_anh import oracle_lop  # noqa: E402

# --nguon: B0 dùng để đoán nhãn (mặc định nosam_kp841 = bước 1 ban đầu; hai_nhanh = B0 thấy VIS và IR riêng, 22/11 khuya)
_ap = argparse.ArgumentParser()
_ap.add_argument("--nguon", default="nosam_kp841")
_ap.add_argument("--hau", default="", help="hậu tố tên biến thể ra, vd _hn")
A = _ap.parse_args()
NGUON = A.nguon
RUN_NGUON = RUNS / f"{NGUON}_b0_s0"
NHAN_DOAN = ANH / f"_nhan_doan_{NGUON}_b0_s0"


def doan_nhan():
    cfg = json.loads((RUN_NGUON / "config.json").read_text(encoding="utf-8"))
    assert cfg["bien_the"] == NGUON and cfg["model"] == "b0"
    m = tao_model("b0", 14, so_kenh(NGUON), NGUON)
    m.load_state_dict(torch.load(RUN_NGUON / "last.pt", map_location="cpu", weights_only=True))
    m.cuda().eval()
    ds = FMBSeg(["dev"], NGUON, False)
    NHAN_DOAN.mkdir(parents=True, exist_ok=True)
    for i in range(len(ds)):
        x, _ = ds[i]
        p = du_doan(m, x[None].cuda()).argmax(1)[0].cpu().numpy() + 1      # 0..13 -> lớp 1..14
        cv2.imwrite(str(NHAN_DOAN / f"{ds.df.image_id.iloc[i]}.png"), p.astype(np.uint8))
    return ds.df


def main():
    df = doan_nhan()
    dung = [0, 0]
    for bt, nguoc in (("orc_doan" + A.hau, False), ("orc_doan_nguoc" + A.hau, True)):
        for r in df.itertuples():
            out = ANH / bt / thu_muc(r) / f"{r.image_id}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            vis = cv2.imread(str(duong(r.vis_path)))
            ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY)
            ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
            lab = cv2.imread(str(NHAN_DOAN / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            if not nguoc:
                gt = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE)
                co = gt > 0
                dung[0] += int((lab[co] == gt[co]).sum())
                dung[1] += int(co.sum())
            yf = oracle_lop(ycc[:, :, 0], ir, lab, nguoc=nguoc)
            cv2.imwrite(str(out), cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
        # run chấm: dùng lại B0 đã train trên ảnh oracle nhãn thật
        goc = RUNS / f"{'orc_nguoc' if nguoc else 'orc_dung'}_b0_s0"
        run = RUNS / f"{bt}_b0_s0"
        run.mkdir(exist_ok=True)
        cfg = json.loads((goc / "config.json").read_text(encoding="utf-8"))
        cfg["bien_the"] = bt
        cfg["ghi_chu"] = f"KHÔNG train: last.pt chép từ {goc.name}; dev = oracle nhãn đoán {NGUON} B0 s0"
        (run / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copy2(goc / "last.pt", run / "last.pt")
        print(f"{bt}: {len(df)} ảnh dev, run {run.name}", flush=True)
    print(f"độ đúng pixel nhãn đoán (bỏ nhãn 0): {dung[0] / dung[1]:.3f}")


if __name__ == "__main__":
    main()
