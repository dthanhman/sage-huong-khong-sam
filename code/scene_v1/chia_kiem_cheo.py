"""25/11: phần 1 của kiểm chéo theo cảnh (TB vs a04). Chia bằng id/cụm + thống kê lớp, độ sáng của TRAIN; KHÔNG xem điểm mô hình nào, không Test.
Luật: cụm = đoạn Train992 có khoảng hở id <= 12 thì gộp. Phần kiểm = tập cụm 330-420 ảnh, >= 5 cụm, lệch ít nhất so với Train992 (tỉ lệ ảnh có lớp hiếm, độ sáng).
Train = 1.220 - kiểm, bỏ ảnh cách ảnh kiểm <= 10 id (đệm), bỏ thêm ảnh trùng cảnh theo ORB (bước sau)."""
import itertools, sys; import numpy as np, pandas as pd
from pathlib import Path
S = Path(__file__).resolve().parent; d = pd.read_csv(S/"dai_dien_dev.csv").sort_values("id").reset_index(drop=True)
t = d[d.tap == "train"].reset_index(drop=True); t["cum"] = (t.id.diff() > 12).cumsum()
# đoạn Dev chen giữa hai đoạn train thì không gộp hai đoạn đó (khoảng hở có Dev)
R = ["truck", "bus", "moto", "light", "sign", "person", "sidewalk"]
X = lambda q: np.r_[[(q[c] > 100).mean() for c in R], (q.y < 97.9).mean(), (q.y >= 145.9).mean()]
goc = X(t); tot = []
cs = sorted(t.cum.unique()); n = t.groupby("cum").size()
for k in range(5, len(cs)):
    for s in itertools.combinations(cs, k):
        m = n[list(s)].sum()
        if 330 <= m <= 420:
            x = X(t[t.cum.isin(s)]); tot.append((float((np.abs(x-goc)/(goc+.05)).mean()), m, s))
tot.sort(); print("số cụm", len(cs), "phương án", len(tot))
for lech, m, s in tot[:4]: print(round(lech, 3), m, [(int(t[t.cum == c].id.min()), int(t[t.cum == c].id.max())) for c in s])
chon = tot[0][2]; val = set(t[t.cum.isin(chon)].id); kiem = np.array(sorted(val))
tr = d[~d.id.isin(val)]; gan = tr.id.map(lambda i: np.abs(kiem-i).min() <= 10)
print("kiểm", len(val), "| train trước đệm", len(tr), "| đệm bỏ", int(gan.sum()))
q = t[t.cum.isin(chon)]; print("lớp: % ảnh Train992 / kiểm:", {c: (round(100*(t[c] > 100).mean(), 1), round(100*(q[c] > 100).mean(), 1), int((q[c] > 100).sum())) for c in R}, "| tối/sáng", goc[-2:].round(2), X(q)[-2:].round(2))
pd.DataFrame(dict(image_id=[f"{i:05d}" for i in d.id], phan=["kiem" if i in val else ("dem" if g else "train") for i, g in zip(d.id, d.id.map(lambda i: i not in val and np.abs(kiem-i).min() <= 10))])).to_csv(S/"kiem_cheo_p1.csv", index=False)
