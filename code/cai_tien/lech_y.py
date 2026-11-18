# Trung bình |Y_a - Y_b| (mức xám) trên val 151 giữa các biến thể ảnh ghép
import sys, cv2, numpy as np, pandas as pd
import os
A = os.environ["SAGE_ANH"]
ids = pd.read_csv(os.path.join(os.environ["SAGE_MAN"], "dev.csv"), dtype={"image_id": str}).image_id
goc = sys.argv[1]
for b in sys.argv[2:]:
    d = [np.abs(cv2.cvtColor(cv2.imread(f"{A}/{b}/train/{i}.png"), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
                - cv2.cvtColor(cv2.imread(f"{A}/{goc}/train/{i}.png"), cv2.COLOR_BGR2YCrCb)[:, :, 0]).mean() for i in ids]
    print(f"|{b} - {goc}| = {np.mean(d):.2f} mức xám")
