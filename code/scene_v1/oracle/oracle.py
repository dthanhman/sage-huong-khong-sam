"""Oracle chọn biến thể từng ảnh (B0, val 151). Oracle giả = chọn seed tốt hơn của CÙNG biến thể."""
import itertools, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "b3"))
from chung import cac_miou
R = __import__("os").environ.get("SAGE_RUNS", str(__import__("pathlib").Path(__file__).resolve().parents[3] / "ket_qua/kham_pha/sang_loc_b0/runs")) + "/"
BT = ["tb", "sage", "nosam_kp841", "m11a_l5_e30_bn", "m11b_l15_e30_bn"]
C, ids = {}, None
for b in BT:
    for s in (0, 1):
        z = np.load(R + f"{b}_b0_s{s}/dev_nham_last.npz")
        i = [str(x) for x in z["image_id"]]
        ids = ids or i
        assert i == ids
        C[b, s] = z["conf"].astype(np.int64)
m = lambda c: cac_miou(c)[0]["mIoU_co_mat"]
diem = {k: np.array([m(x) for x in v]) for k, v in C.items()}   # mIoU từng ảnh

def oracle(keys):
    d = np.stack([diem[k] for k in keys])           # chọn theo điểm từng ảnh
    chon = d.argmax(0)
    tong = sum(C[keys[j]][n] for n, j in enumerate(chon))
    don = max(m(C[k].sum(0)) for k in keys)
    return m(tong), m(tong) - don, np.bincount(chon, minlength=len(keys))

print("mIoU từng run:", {f"{b}_s{s}": round(m(C[b, s].sum(0)), 2) for b in BT for s in (0, 1)})
print("\n== ORACLE GIẢ (cùng biến thể, chọn seed 0/1 từng ảnh) – mức 'ăn may'")
gia = []
for b in BT:
    o, g, _ = oracle([(b, 0), (b, 1)]); gia.append(g)
    print(f"  {b:18s} oracle {o:.2f}  lợi so run tốt hơn +{g:.2f}")
print(f"  -> TB lợi giả: +{np.mean(gia):.2f} (min {min(gia):.2f}, max {max(gia):.2f})")
print("\n== ORACLE THẬT (2 biến thể khác nhau, cùng seed)")
for a, b in itertools.combinations(BT, 2):
    r = [oracle([(a, s), (b, s)]) for s in (0, 1)]
    print(f"  {a:16s}+ {b:16s} lợi s0 +{r[0][1]:.2f}  s1 +{r[1][1]:.2f}  TB +{(r[0][1]+r[1][1])/2:.2f}")
print("\n== ORACLE 5 BIẾN THỂ (cùng seed) – chỉ để tham khảo, nhiều lựa chọn hơn nên lợi tự cao hơn")
for s in (0, 1):
    o, g, n = oracle([(b, s) for b in BT])
    print(f"  s{s}: oracle {o:.2f}  lợi +{g:.2f}  số ảnh chọn {dict(zip(BT, n.tolist()))}")
