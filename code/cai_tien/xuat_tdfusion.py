"""B1 (19/11): xuất ảnh ghép TDFusion (CVPR 2025, checkpoint FMB công khai) theo đúng định dạng b3/xuat_anh.py.

Làm giống test.py của TDFusion:
  - đầu vào: Y của IR và Y của VIS (chia 255), ghép kênh theo thứ tự (IR, VIS);
  - đầu ra: sigmoid, rồi chuẩn hóa min-max trên TỪNG ảnh, nhân 255.
Màu: IR là ảnh xám nên test.py lấy Cr/Cb của VIS -> giống cách của nhóm (Y mới + CrCb VIS qua OpenCV).
Lưu: $SAGE_ANH/tdfusion/{train|test}/{id}.png, cho mọi ảnh trong manifest (train, dev, test).
Lưu ý: checkpoint TDFusion_FMB.pth học trên toàn bộ FMB train có nhãn -> đã thấy val 151 (như SAGE).

Chạy: cd cai_tien && SAGE_MAN=... python xuat_tdfusion.py --repo ../../external/TDFusion
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b3"))
from chung import ANH, MAN, duong, thu_muc  # noqa: E402

TEN = "tdfusion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="thư mục code TDFusion (có nets/ReFusion.py, models/)")
    ap.add_argument("--gioi_han", type=int, default=0, help="chỉ để thử: xuất N ảnh đầu")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    from nets.ReFusion import ReFusion

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ReFusion()
    net.load_state_dict(torch.load(repo / "models" / "TDFusion_FMB.pth", map_location="cpu", weights_only=True))
    net.to(dev).eval()

    df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str}) for s in ("train", "dev", "test")])
    if a.gioi_han:
        df = df.head(a.gioi_han)
    t = lambda z: torch.from_numpy(z.astype(np.float32) / 255)[None, None].to(dev)
    n = 0
    with torch.inference_mode():
        for r in df.itertuples():
            out = ANH / TEN / thu_muc(r) / f"{r.image_id}.png"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            vis = cv2.imread(str(duong(r.vis_path)))
            ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
            ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            f = net(torch.cat((t(ir), t(ycc[:, :, 0])), 1))
            f = (f - f.min()) / (f.max() - f.min())
            yf = (f * 255).round().byte().cpu().numpy()[0, 0]
            cv2.imwrite(str(out), cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
            n += 1
    print(f"{TEN}: ghi {n} ảnh mới vào {ANH / TEN}", flush=True)


if __name__ == "__main__":
    main()
