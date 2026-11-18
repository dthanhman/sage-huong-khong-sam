# E17 – SAM có làm ảnh ghép SAGE tốt hơn cho phân đoạn không?

Bài SAGE (Bảng 3): bỏ SAM thì mIoU FMB 61,2 → 56,5 (**−4,7**). Nhóm kiểm lại bằng đúng thiết lập B3.

## Cách làm

1. `train_student_nosam.py`: train lại Student SAGE (cùng kiến trúc, 0,136M tham số) trên 1.220 ảnh train FMB,
   **chỉ loss nội dung** (MSE + gradient, hệ số như code gốc), bỏ Teacher, SAM, loss tương phản, loss đặc trưng.
   Adam 2e-3, cosine, batch 4, crop 192×256, 100 epoch, chỉ cập nhật ở batch chẵn như code gốc. Seed 0. Train trên máy
   Mân (RTX 3050, ~22 phút) → `sage_student/nosam.pt`.
2. Kaggle notebook `thanhmn123/sage-e17-nosam-b3` (dataset `thanhmn123/fmb-ivif-sage` + `thanhmn123/sage-e17-nosam`):
   xuất ảnh `nosam` (Y = Student_nosam(Y_vis, IR), ghép CrCb của VIS) → SegFormer-B3 y hệt B3, seed 0, 1, 2.
3. So với 12 run B3 đã khóa bằng `so_sanh_b3.py` (thêm `nosam` vào danh sách biến thể), 273 ảnh, mIoU_co_mat.

## Tiêu chí đặt trước (ghi 14/11/2026, trước khi có kết quả)

Gọi d = mIoU(sage) − mIoU(nosam); CI95 bootstrap theo khối, gộp 3 seed (cùng bộ chỉ số bootstrap, như B3).

| Kết luận | Điều kiện |
|---|---|
| **SAM có ích** | d > 0 ở cả 3 seed **và** CI gộp > 0 |
| **Không thấy SAM có ích** | CI gộp chứa 0, hoặc d gộp ≤ 0 |
| **SAM làm hại** | d < 0 ở cả 3 seed **và** CI gộp < 0 |
| Lưng chừng | còn lại |

Kèm (không đổi kết luận trên):
- **Tái lập ablation của bài?** Có nếu d gộp ≥ +3 (bài +4,7); không nếu cận trên CI gộp < +4,7.
- nosam − tb và nosam − vis (Student không SAM có hơn trộn trung bình không).

Giới hạn: chỉ 1 lần train Student (seed 0) – SAGE cũng chỉ công bố 1 checkpoint; Student gốc còn được chưng cất từ Teacher
nên "bỏ SAM" ở đây = bỏ toàn bộ Teacher (SAM + SPA + X-Decoder), giống cách bài làm ablation "w/o SAM" hay không thì bài không ghi rõ.

## Kết quả (15/11/2026)

Chạy: `PYTHONIOENCODING=utf-8 SAGE_RUNS=_kaggle_out/runs_gop python e17/so_sanh_e17.py` → `e17/so_sanh_e17.txt`.
Output Kaggle: `_kaggle_out/e17` (chỉ tải npz/json/log, bỏ ảnh dự đoán và last.pt). Số sage/vis/tb trùng 12 run B3 đã khóa.

mIoU_co_mat, 273 ảnh:

| Biến thể | s0 | s1 | s2 | TB ± SD |
|---|---|---|---|---|
| vis | 62,07 | 62,56 | 63,02 | 62,55 ± 0,48 |
| ir | 54,66 | 56,64 | 57,93 | 56,41 ± 1,64 |
| sage | 62,62 | 63,07 | 64,42 | 63,37 ± 0,94 |
| tb | 64,27 | 63,91 | 65,09 | 64,42 ± 0,60 |
| **nosam** | 62,89 | 64,07 | 64,46 | **63,81 ± 0,81** |

Gộp 3 seed (CI95 bootstrap theo khối):

| Cặp | Chênh | CI95 |
|---|---|---|
| **d = sage − nosam** | **−0,44** | [−1,02; +0,20] |
| nosam − tb | −0,61 | [−1,33; +0,06] |
| nosam − vis | +1,26 | [−0,25; +3,22] |
| tb − sage | +1,05 | [+0,08; +2,16]* |

d theo seed: −0,28 / −1,00 / −0,04 (cả 3 âm). Theo độ sáng: tối −1,17, vừa −0,84, sáng −0,13 (đều CI chứa 0).

**Kết luận theo tiêu chí:** **Không thấy SAM có ích** (d gộp ≤ 0 và CI chứa 0). Không đạt mức "SAM làm hại" vì CI gộp chưa < 0.
**Tái lập ablation của bài: KHÔNG** (cận trên CI +0,20 < +4,7). Student không SAM chưa hơn trộn trung bình (tb) và chỉ nhỉnh VIS (CI chứa 0).

## Bản 2 – mỗi seed một Student riêng (nosamk, tải về 16/11/2026)

Student không SAM train lại với seed 1, 2 (nosam_s1.pt, nosam_s2.pt), ghép với SegFormer seed tương ứng. Seed 0 dùng lại nosam.pt. Chi tiết: `e17/so_sanh_e17_ban2.txt`.

| Seed | sage | nosamk | d = sage − nosamk |
|---|---|---|---|
| 0 | 62,62 | 62,89 | −0,28 |
| 1 | 63,07 | 64,04 | −0,96 |
| 2 | 64,42 | 64,64 | −0,22 |
| **Gộp (273 ảnh)** | | | **−0,49 [−1,35; +0,35]** |

**Kết luận theo tiêu chí: vẫn "Không thấy SAM có ích"** (d âm cả 3 seed, CI chứa 0). Đổi seed Student không làm đổi kết luận bản 1 (−0,44 [−1,02; +0,20]). nosamk − tb −0,56 [−1,09; +0,12]; nosamk − vis +1,31 [−0,14; +3,27] (đều CI chứa 0).
