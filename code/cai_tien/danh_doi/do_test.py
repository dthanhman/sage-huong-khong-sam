"""Đo chỉ số ghép trên test 280 cho vis, ir, tb, sage, nosam. 'ir' = ảnh IR làm ảnh ghép (đầu vào thật của run ir_b3)."""
import sys, cv2, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, r"D:/CS406/DoAn/Sage/tai_lap/cai_tien")
import chi_so_ghep as C

def do(args):
    bt, r = args
    if bt != "ir":
        return C.do_mot(args)
    vis = cv2.imread(str(C.duong(r["vis_path"])))
    yv = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ir = cv2.imread(str(C.duong(r["ir_path"])), cv2.IMREAD_GRAYSCALE)
    yf = ir.copy()
    ms = C.fusion_ms_ssim(yf, ir, yv, aggregation="sum")
    return {"bien_the": bt, "image_id": r["image_id"],
            "SCD": C.sum_of_correlation_differences(yf, ir, yv),
            "MS_SSIM": ms["MS_SSIM"], "MS_SSIM_IR": ms["MS_SSIM_IR"], "MS_SSIM_VIS": ms["MS_SSIM_VIS"],
            "Q_ABF": C.q_abf(yf, ir, yv), "EN": C.entropy(yf), "SD": C.standard_deviation(yf)}

if __name__ == "__main__":
    df = pd.read_csv(C.MAN / "test.csv", dtype={"image_id": str})
    df["thu_muc"] = [C.thu_muc(r) for r in df.itertuples()]
    viec = [(bt, r) for bt in ["sage", "nosam", "tb", "vis", "ir"] for r in df.to_dict("records")]
    with ProcessPoolExecutor(6) as ex:
        kq = pd.DataFrame(list(ex.map(do, viec, chunksize=8)))
    kq.to_csv("test_tung_anh.csv", index=False)
    th = kq.groupby("bien_the")[C.CHI_SO].mean()
    th["n"] = kq.groupby("bien_the").size()
    th.to_csv("test_tong_hop.csv"); print(th.round(4).to_string())
