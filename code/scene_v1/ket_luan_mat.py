"""Kết luận S-mắt theo luật ĐẶT TRƯỚC trong chay_mat.sh (đọc he_so_ir_mat.csv do do_he_so_ir.py ghi)."""
_TL = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()   # thư mục code/
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
a = pd.read_csv(f"{_TL}/scene_v1/he_so_ir_mat.csv", index_col=0)
VIS = ["sidewalk", "light", "sign", "truck", "bus"]
LOP = VIS + ["person"]


def D(b):
    return (a.loc[b, LOP] - a.loc["w_lint", LOP]).abs().mean()


m = a.loc["lint_mat"]
dk = {"a_person >= 0,6": m["person"] >= 0.6,
      "TB a lớp VIS <= 0,2": m[VIS].mean() <= 0.2,
      "D(mat) <= ½ D(nha)": D("lint_mat") <= 0.5 * D("lint_nha")}
print(f"D(lint_nha) {D('lint_nha'):.3f} | D(lint_mat) {D('lint_mat'):.3f} | "
      f"a_person: nha {a.loc['lint_nha', 'person']:.2f}, mat {m['person']:.2f} | "
      f"TB a VIS: nha {a.loc['lint_nha', VIS].mean():.2f}, mat {m[VIS].mean():.2f}")
for k, v in dk.items():
    print(f"  {k}: {'ĐẠT' if v else 'không'}")
print("KẾT LUẬN:", "GO – gắn mắt vào chưng cất" if all(dk.values()) else "KHÔNG ĐẠT luật đặt trước")
