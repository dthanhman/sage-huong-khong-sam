import re, glob, itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
K = r"D:/CS406/DoAn/Sage/tai_lap/_kaggle_out"; G = r"D:/CS406/DoAn/Sage/tai_lap/cai_tien/chi_so_ghep"
OUT = r"D:/CS406/DoAn/Sage/hinh_cai_tien"

# ---- TN1: B3, test 280, train 1.220, 3 seed, mIoU có mặt 13 lớp (quy ước chính) ----
r = pd.read_csv(f"{K}/runs_gop/tong_hop.csv"); r = r[r.so_anh == 280]
m1 = r[["bien_the", "seed", "mIoU_co_mat"]].rename(columns={"mIoU_co_mat": "mIoU"})
for f in sorted(glob.glob(f"{K}/e17/runs/log_nosam_b3_s*_1220_15lop.txt")):
    s = int(re.search(r"_s(\d)_1220", f).group(1))
    t = open(f, encoding="utf8").read()
    v = re.search(r"280 ảnh: mIoU có mặt \(13 lớp\) ([\d.]+)", t).group(1)
    m1 = pd.concat([m1, pd.DataFrame([{"bien_the": "nosam", "seed": s, "mIoU": float(v)}])])
f1 = pd.read_csv("test_tong_hop.csv", index_col=0)

# ---- TN2: B0, Dev 175, train 1.045, 2 seed, mIoU 15 lớp (LOP0), last ----
d = pd.concat([pd.read_csv(f"{K}/m5m4/runs/dev_tong_hop.csv"), pd.read_csv(f"{K}/m8/runs/dev_tong_hop.csv")])
d["bien_the"] = d.run.str.replace(r"_b0_s\d", "", regex=True); d["seed"] = d.run.str[-1].astype(int)
m2 = d[["bien_the", "seed", "dev_last"]].rename(columns={"dev_last": "mIoU"})
f2 = pd.concat([pd.read_csv(f"{G}/{x}_tong_hop.csv", index_col=0) for x in ["dev", "dev_m8b", "dev_m8c"]])
f2 = f2[~f2.index.duplicated()]

def bang(m, f, ten):
    g = m.groupby("bien_the").mIoU.agg(["mean", "std", "count", list])
    b = g.join(f[["SCD", "MS_SSIM", "Q_ABF", "SD"]], how="inner")
    bo = sorted(set(g.index) - set(b.index))
    print(f"\n== {ten}  (bỏ vì không có ảnh để đo chỉ số ghép: {bo})"); print(b.round(3).to_string())
    return b

def tq(b, c):
    rho, _ = spearmanr(b[c], b["mean"]); pr, _ = pearsonr(b[c], b["mean"])
    # p hoán vị chính xác cho Spearman (n nhỏ)
    x = b[c].rank().values; y = b["mean"].rank().values
    ds = [abs(spearmanr(x, p)[0]) for p in itertools.permutations(y)]
    p = np.mean(np.array(ds) >= abs(rho) - 1e-12)
    return rho, pr, p

b1 = bang(m1, f1, "TN1 B3 test"); b2 = bang(m2, f2, "TN2 B0 Dev")
rows = []
for ten, b in [("TN1", b1), ("TN2", b2)]:
    for c in ["SCD", "MS_SSIM", "Q_ABF", "SD"]:
        rho, pr, p = tq(b, c); rows.append({"tn": ten, "chi_so": c, "n": len(b), "spearman": rho, "p_hoanvi": p, "pearson": pr})
    if ten == "TN1":  # chỉ các pp ghép thật (bỏ vis, ir)
        bb = b.drop(["vis", "ir"])
        for c in ["SCD", "MS_SSIM", "Q_ABF"]:
            rho, pr, p = tq(bb, c); rows.append({"tn": "TN1_chi_ghep", "chi_so": c, "n": len(bb), "spearman": rho, "p_hoanvi": p, "pearson": pr})
tqd = pd.DataFrame(rows); print("\n", tqd.round(3).to_string())
b1.to_csv("tn1_b3_test.csv"); b2.to_csv("tn2_b0_dev.csv"); tqd.to_csv("tuong_quan.csv", index=False)

mau = {"vis": "#4C78A8", "ir": "#9D755D", "tb": "#54A24B", "sage": "#E45756", "nosam": "#F58518", "tn": "#72B7B2", "tbcd": "#B279A2", "vci": "#EECA3B", "m4_l1": "#888"}
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for i, (b, m, ten, nh) in enumerate([(b1, m1, "TN1: SegFormer-B3, test 280 ảnh, 3 seed\nmIoU 13 lớp có mặt", "test"),
                                     (b2, m2, "TN2: SegFormer-B0, Dev 175 ảnh, 2 seed\nmIoU 15 lớp", "Dev")]):
    for j, c in enumerate(["SCD", "MS_SSIM", "Q_ABF"]):
        a = ax[i, j]
        for bt, row in b.iterrows():
            ys = row["list"]
            a.scatter([row[c]] * len(ys), ys, s=14, color=mau.get(bt, "k"), alpha=.45)
            a.errorbar(row[c], row["mean"], yerr=row["std"], fmt="o", ms=8, color=mau.get(bt, "k"), capsize=4)
            a.annotate(bt, (row[c], row["mean"]), xytext=(6, 4), textcoords="offset points", fontsize=10)
        t = tqd[(tqd.tn == ("TN1" if i == 0 else "TN2")) & (tqd.chi_so == c)].iloc[0]
        a.set_title(f"{c} ({nh})   Spearman ρ={t.spearman:+.2f} (p={t.p_hoanvi:.2f}, n={int(t.n)})", fontsize=10)
        a.set_xlabel(f"{c} (trung bình {nh}, kênh Y)"); a.grid(alpha=.3)
        if j == 0: a.set_ylabel(ten)
fig.suptitle("Chỉ số ghép ảnh vs mIoU segmentation (chấm trung bình ± SD theo seed; chấm nhạt = từng seed)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/danh_doi_chi_so_ghep_vs_miou.png", dpi=130)
print("saved")
