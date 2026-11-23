"""In IoU từng lớp (dev val151) cho nhiều run, từ dev_nham_last.npz. Dùng: python iou_lop_bang.py ten=duong ..."""
import sys, numpy as np, pandas as pd
TEN = ["road","sidewalk","building","traffic_light","traffic_sign","vegetation","sky","person","car","truck","bus","motorcycle","bicycle","pole"]
L10 = ["road","sidewalk","building","traffic_sign","vegetation","sky","person","car","truck","pole"]
out = {}
for a in sys.argv[1:]:
    n, p = a.split("=", 1)
    c = np.load(p + "/dev_nham_last.npz")["conf"].sum(0).astype(float)
    iou = np.diag(c) / (c.sum(0) + c.sum(1) - np.diag(c)) * 100
    s = pd.Series(iou, index=TEN)
    s["mIoU10"] = s[L10].mean(); s["mIoU14"] = s[TEN].mean()
    out[n] = s
d = pd.DataFrame(out)
print(d.round(1).to_string())
