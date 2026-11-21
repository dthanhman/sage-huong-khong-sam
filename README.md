# Ghép ảnh IR–VIS không cần SAM: tổng kết hướng thử nghiệm

Người làm: Đào Thanh Mân. Cập nhật: **21/11/2026** (thêm mục 4.13–4.16: đo cận trên, chốt A2 và chạy 3 seed, tìm chỗ ảnh ghép làm mất thông tin, A4; mục 2.4 đổi mục tiêu; mục 5 thêm kiểm tra tổng quát trên dataset khác).

**Tóm tắt kết quả đến 21/11:**
- Bỏ SAM rẻ hơn khoảng 250 lần lúc train, và mIoU **không thua** SAGE (mục 2.1, 4.8).
- Trên 10 lớp đủ ảnh, ba cách (bỏ SAM, SAGE, TB) **ngang nhau** (mục 4.8).
- Điểm SAGE thật sự hơn: làm **người nổi bật có chọn lọc** (mục 4.11). Loss A2 làm được điều này mà không cần SAM, nhưng mIoU mới hơn khoảng +0,5 ở 1 seed (mục 4.12, 4.14).
- Còn chỗ để cải thiện: một SegFormer nhận thẳng cả VIS lẫn IR (không ghép) hơn SAGE và TB **+2,3 điểm**. Phần hơn nằm ở vỉa hè, biển báo, người (mục 4.13).
- Ảnh ghép không mất cạnh, mà mất **thông tin vùng** của IR (ví dụ IR phân biệt đường với vỉa hè tốt gấp đôi VIS, ảnh bỏ SAM làm mất phần này) (mục 4.15). A4 thêm loss nhắm vào chỗ này (mục 4.16).
- **Đang chạy (21/11):** A2 chạy đủ 3 seed và A4 seed 0 trên Kaggle; cận trên chạy thêm 2 seed. Chưa có kết luận.

File này đọc riêng là đủ hiểu. Các thư mục đi kèm chỉ chứa số liệu gốc và code, dùng khi muốn kiểm lại.

## Dòng thời gian

| Ngày | Việc chính | Mục |
|---|---|---|
| 13/11 | Chỉ số ghép (SCD, Q_ABF, MS-SSIM) xếp hạng khác mIoU | 2.3 |
| 15–16/11 | Train lại Student **bỏ SAM**, chấm SegFormer-B3 trên test 280, 3 seed: không thua SAGE | 2.1 |
| 16–18/11 | Thử M1, M4, M5, M8, M11 trên split cũ: không hơn TB | 3 |
| 17–18/11 | Đo chi phí trên T4: bỏ SAM rẻ hơn ~250 lần lúc train | 2.2 |
| 18/11 | Chuyển sang split scene_v1, tạo tập khám phá (train 841 / val 151), đặt sàn ảnh ghép, dựng 2 nền A/B | 4.1–4.5 |
| 19/11 | Tìm ra lỗi BatchNorm; sàng lọc mIoU bằng SegFormer-B0; đối chứng CE; thử router | 4.6–4.10 |
| 19–20/11 | Nhiều seed: phần hơn của bỏ SAM nằm ở lớp hiếm; chọn số chính là 10 lớp đủ ảnh | 4.8 |
| 20/11 | TDFusion, A1 rớt; tìm ra SAGE hơn ở người nhỏ; A2 λ = 10 | 4.10–4.12 |
| 20–21/11 (đêm) | Đo cận trên (SegFormer nhận thẳng VIS + IR); A2 λ = 30, 100 | 4.13, 4.14 |
| 21/11 | Chốt A2 λ = 10, train seed 1, 2; đo cạnh và độ tách vùng; viết A4; đẩy Kaggle | 4.14–4.16 |

---

## 1. Bối cảnh trong một phút

- **SAGE** (CVPR 2025) ghép ảnh hồng ngoại (IR) với ảnh thường (VIS) thành một ảnh. Mạng chạy thật là một **Student** nhỏ (0,136M tham số). Lúc train, Student được dạy bởi một Teacher dùng **SAM**, mô hình phân đoạn rất nặng.
- Nhóm đánh giá ảnh ghép bằng cách train **SegFormer** trên ảnh ghép, rồi đo **mIoU** phân đoạn trên dataset FMB. mIoU càng cao thì ảnh ghép càng giúp hiểu cảnh.
- **TB** là cách ghép đơn giản nhất: lấy trung bình (VIS + IR) / 2 trên kênh độ sáng, màu giữ từ VIS. Không cần học, không tốn gì.
- Hướng này **khác** hướng Q06 của nhóm. Q06 fine-tune Student SAGE có sẵn, nên vẫn dựa gián tiếp vào SAM. Hướng này **bỏ hẳn SAM**.

**Câu hỏi:** bỏ SAM thì có giữ được mIoU không? Có làm được phương pháp không SAM mà mIoU hơn cả SAGE lẫn TB không?

---

## 2. Vì sao chọn hướng này

### 2.1. Bỏ SAM không làm giảm mIoU (15–16/11)

Mình train lại Student của SAGE (giữ nguyên kiến trúc) **chỉ với loss nội dung**: giữ ảnh ghép gần VIS, gần IR và giữ cạnh. Bỏ Teacher, SAM và các loss dùng SAM. Sau đó chấm bằng SegFormer-B3 trên 280 ảnh test FMB, 3 seed, cùng cấu hình cho mọi loại ảnh.

| Ảnh đưa vào SegFormer | mIoU (13 lớp, TB ± SD, 3 seed) |
|---|---|
| TB | **64,2 ± 0,6** |
| Student bỏ SAM | 63,5 ± 0,8 |
| SAGE | 63,2 ± 1,0 |
| Chỉ VIS | 62,2 ± 0,5 |
| Chỉ IR | 56,3 ± 1,5 |

- Bảng trên tính trên đủ 280 ảnh. Nếu bỏ 7 ảnh test bị rò (khung liền kề ảnh train), còn 273 ảnh, thì số là TB 64,42 / bỏ SAM 63,81 / SAGE 63,37 / VIS 62,55. Thứ tự không đổi.
- SAGE thấp hơn bản bỏ SAM ở **cả 3 seed** (trung bình −0,44 điểm, khoảng tin cậy 95% từ −1,02 đến +0,20). Train lại bản bỏ SAM bằng 3 seed khác, kết luận vẫn vậy.
- Bài báo SAGE ghi bỏ SAM thì mIoU giảm từ 61,2 xuống 56,5. **Nhóm không tái lập được mức giảm này.**
- Cách nói đúng: "trong thiết lập của nhóm, **không thấy** SAM giúp mIoU". Chưa nói được là SAM vô dụng.

### 2.2. Bỏ SAM rẻ hơn rất nhiều (đo 17–18/11)

Đo trên GPU Kaggle T4:

| | SAGE có SAM | Bỏ SAM | TB |
|---|---|---|---|
| SAM tạo mask cho 1.220 cặp ảnh | 99,2 giờ | 0 | 0 |
| Train 100 epoch | 26,3 giờ | 0,5 giờ | 0 |
| **Tổng** | **≈ 125 giờ** | **0,5 giờ** | 0 |
| Bộ nhớ GPU lúc train | 11,9 GB | 1,1 GB | – |
| Thời gian chạy một ảnh | 54,3 ms | 54,3 ms | 0,05 ms |

Bỏ SAM rẻ hơn khoảng 250 lần lúc train. Lúc chạy thì như nhau, vì cùng một Student.

### 2.3. Chỉ số ghép không đoán được mIoU (13/11, kiểm lại 18/11)

Các chỉ số đo chất lượng ảnh ghép (SCD, Q_ABF, MS-SSIM) **xếp hạng các phương pháp khác với mIoU**. Ví dụ: SAGE có SCD cao nhất (1,71), nhưng mIoU thấp hơn TB (SCD chỉ 1,38).

Vì vậy mình dùng chỉ số ghép làm **sàn**, chỉ để bảo đảm ảnh không bị hỏng. **Chọn phương án thì dựa vào mIoU.**

### 2.4. Mục tiêu đặt ra (17/11, sửa 21/11)

Phương pháp của mình phải:
1. **không cần SAM**, kể cả không khởi tạo từ Student SAGE;
2. có mIoU **hơn nhất quán cả SAGE và TB**: cùng cách chấm, so từng cặp seed, hơn ở mọi seed (theo thang trong `code/QUY_TAC_CHAM.md`);
3. **qua sàn** chỉ số ghép (MS-SSIM ≥ 99%, Q_ABF ≥ 98% của SAGE).

TB là đối thủ khó nhất: không tốn gì mà mIoU đang cao nhất.

**Lịch sử:** ngày 20/11 nhóm đặt đích "hơn 5% tương đối". Ngày 21/11 đổi lại thành "hơn nhất quán", vì chưa biết trần đạt được là bao nhiêu. Độ lớn cần đạt sẽ đặt sau khi đo xong cận trên (mục 4.13).

---

## 3. Đã thử gì trên split cũ (16–18/11), và vì sao chưa đạt

Thiết lập: train 1.045 ảnh, chấm 175 ảnh Dev, SegFormer-B0. Đạt khi hơn TB ít nhất +1,0 điểm ở cả 2 seed. TB đạt 53,13 (seed 0) và 53,27 (seed 1).

| Phương án (ngày) | Ý tưởng | Kết quả | Vì sao dừng |
|---|---|---|---|
| M1 (16/11) | Fine-tune Student SAGE bằng loss phân đoạn từ một SegFormer đóng băng | Loss phân đoạn gần như không giảm | Vẫn khởi tạo từ SAGE; SegFormer đóng băng đã học thuộc ảnh train nên không chỉ được hướng sửa |
| M4 (17/11) | Giữ TB, học thêm một đầu trộn nhỏ | 52,80 / 53,07 (≈ TB −0,3) | Đầu trộn gần như không học |
| M5 (17/11) | Trộn IR theo độ sáng ảnh (luật tay) | 52,10 / 51,27 | Thêm IR vào ảnh sáng làm hại |
| M8b (17–18/11) | TB cộng thêm chi tiết cạnh | 52,63 / 53,13 | Không hơn TB |
| M8c (18/11) | VIS + IR − trung bình IR (để đẩy SCD lên) | 52,19 / 51,46 | SCD rất cao mà mIoU thấp; nhà cửa bị cháy trắng |
| M11 bản đầu (18/11) | Student bỏ SAM và SegFormer **học cùng nhau** | Ảnh nghiêng về VIS, rớt SCD | Chưa chấm mIoU |

**Rút ra:**
- Trộn VIS/IR bằng luật tay hay theo tỉ lệ cố định đều không hơn TB.
- Dùng SegFormer đóng băng làm "thầy" không hiệu quả. Phải cho SegFormer học cùng mạng ghép.
- Đẩy chỉ số ghép lên không kéo mIoU lên theo.

---

## 4. Đang làm: split mới của nhóm (scene_v1)

### 4.1. Chia dữ liệu (18/11)

Nhóm chốt split **scene_v1**: 992 ảnh Train và 228 ảnh Dev, chia theo cảnh. Quy tắc:
- **Dev 228 dùng để chọn phương án.** Mỗi phương án phải khóa hết tham số trước, rồi chấm Dev đúng một lần. Không chỉnh rồi chấm lại.
- **Test 280 chỉ cho số cuối** của phương án đã được chọn. Không dùng Test để chọn.

Để thử nhiều lần mà không đụng Dev, mình cắt thêm một **tập khám phá** từ 992 ảnh Train:
- **val 151**: nguyên 3 đoạn ảnh liền số (00136–00156, 00649–00700, 01223–01300), có cả ảnh tối và ảnh ban ngày. Lấy nguyên đoạn để không cắt đôi một cảnh.
- **train 841**: phần còn lại.

Vì vậy có **hai tầng chấm**:

| | Tầng 1: sàng lọc | Tầng 2: chọn |
|---|---|---|
| Mục đích | Loại bớt phương án, chỉnh tham số | Chọn phương án (rồi chấm Test 280 cho phương án được chọn) |
| SegFormer train trên | 841 ảnh | 992 ảnh |
| Chấm trên | val 151 | Dev 228 |
| Loại SegFormer | B0 (nhỏ, chạy máy nhà) | B3 (như bài báo, chạy Kaggle) |
| Số lần chấm | Nhiều lần | Một lần |

Số của hai tầng **không so trực tiếp với nhau**, vì khác mạng, khác dữ liệu train, khác tập chấm.

### 4.2. Sàn ảnh ghép (18/11)

Mình train Student bỏ SAM 5 lần với 5 seed, đo trên val 151, rồi đặt ngưỡng = SAGE − 2 × độ lệch chuẩn giữa các seed:
- **MS-SSIM ≥ 99%** của SAGE;
- **Q_ABF ≥ 98%** của SAGE;
- SCD chỉ báo kèm, vì chính Student bỏ SAM cũng chỉ đạt 86–92%.

### 4.3. Hai nền phương án (18/11)

**Nền A – M11 (học cùng):** Student khởi tạo từ bản bỏ SAM. Ảnh ghép đi thẳng vào SegFormer-B0, và hai mạng học cùng nhau. Loss của Student = λ × (loss phân đoạn) + β × (loss nội dung). 5 epoch đầu chỉ SegFormer học, để nó đủ tốt trước khi "dạy" Student.

**Nền B – TB + phần sửa:** ảnh ghép = TB + một phần sửa nhỏ do Student học. Lúc đầu phần sửa bằng 0, nên ảnh đúng bằng TB. Loss = giữ gần TB + giữ cạnh. Lý do: TB có mIoU tốt nhất nhưng mất cạnh (Q_ABF chỉ 60% SAGE), nên giữ TB và chỉ học cách trả cạnh. Bước sau sẽ thêm loss phân đoạn.

### 4.4. Kết quả chỉ số ghép trên val 151 (% so với SAGE) (18–19/11)

| Ảnh ghép | Loại | MS-SSIM | Q_ABF | SCD (kèm) | Qua sàn? |
|---|---|---|---|---|---|
| SAGE | mốc | 100 | 100 | 100 | – |
| TB | mốc | 105,0 | 60,3 | 82,7 | ❌ (Q_ABF) |
| TB + chi tiết (2 kiểu) | đối chứng không học | 100,5–100,9 | 99,9–100,8 | 88–91 | ✅ |
| Student bỏ SAM | nền | 101,4 | 108,2 | 88,9 | ✅ |
| M11, λ = 1 | nền A | 101,1 | 109,9 | 89,1 | ✅ |
| TB + phần sửa, w_tb = 15 | nền B | 98,9 | 110,1 | 91,0 | ❌ (MS-SSIM) |
| **TB + phần sửa, w_tb = 150** | nền B | **101,4** | **109,2** | 89,9 | ✅ |
| **Nền A + CE, sửa BN** (`m11a_l5_e30_bn`, mục 4.6) | nền A | **100,6** | **110,4** | 90,6 | ✅ |
| **Nền B + CE, sửa BN** (`m11b_l15_e30_bn`, mục 4.6) | nền B | **101,5** | **110,3** | 90,0 | ✅ |

- Với w_tb = 15, phần sửa quá tay (trung bình 11,3 mức xám), ảnh trôi về phía VIS. Tăng trọng số giữ TB lên 150 thì phần sửa còn 4,9 mức xám và qua sàn.
- Qua sàn chỉ có nghĩa là ảnh không bị hỏng. mIoU ở mục 4.7.

### 4.5. Chẩn đoán M11: vì sao học phân đoạn mà ảnh gần như không đổi (18/11)

**Dấu hiệu:** ảnh M11 chỉ khác ảnh bỏ SAM trung bình 6,7 mức xám. Để so sánh, SAGE khác ảnh bỏ SAM tới 17,3 mức xám.

**Cách đo:** chạy lại M11, cứ 10 batch đo riêng hai lực kéo lên Student: lực từ loss phân đoạn và lực từ loss nội dung (độ lớn gradient), cùng góc giữa hai lực (cos).

| Epoch | Lực phân đoạn / lực nội dung | cos |
|---|---|---|
| 6 | 0,15 | +0,05 |
| 7 | 0,25 | +0,05 |
| 8 | 0,21 | +0,08 |

**Đọc:**
- cos ≈ 0: hai loss **không kéo ngược nhau**.
- **Lực phân đoạn yếu hơn 4–7 lần.** Tốc độ học của Student cũng nhỏ (1e-4, trong khi bản bỏ SAM train với 2e-3).
- Ngay khi được học, loss nội dung tụt từ 1,76 xuống 1,09. Student lo khớp loss nội dung, gần như không học phân đoạn.

**Đang sửa:** tăng λ lên 5 và tốc độ học Student lên 5e-4. Đặt trước cách đọc kết quả:
- lực hai bên gần cân bằng (tỉ lệ 0,7–1,5), ảnh khác bản bỏ SAM ≥ 10 mức xám, vẫn qua sàn → train đầy đủ rồi đem chấm mIoU;
- ảnh đổi nhiều nhưng rớt sàn → giảm λ xuống 2–3;
- ảnh vẫn đổi ít → cho Student học quen loss nội dung trước, rồi mới thêm loss phân đoạn.

**Kết quả lần sửa (λ = 5, lr Student 5e-4, 10 epoch):**

| Epoch | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| Lực phân đoạn / lực nội dung | 0,41 | 0,76 | 1,17 | 0,78 | 0,51 |

- **Lực đã gần cân bằng** (phần lớn trong khoảng 0,5–1,2), và hai loss vẫn không kéo ngược nhau (cos +0,07 đến +0,20).
- **Nhưng ảnh vẫn gần như không đổi.** So công bằng, cùng khoảng 8–10 epoch: ảnh lệch bản bỏ SAM 3,6 mức xám với λ = 1 và 4,2 mức xám với λ = 5. Tăng lực phân đoạn gấp khoảng 4 lần mà ảnh chỉ đổi thêm 0,6 mức xám.
- Chỉ số ghép (% SAGE): MS-SSIM 99,9, Q_ABF 108,6 → vẫn qua sàn; SCD giảm còn 81,3.

**Đọc:** đây là nhánh "ảnh vẫn đổi ít". Kết hợp với việc lực đã cân bằng, khả năng cao là **SegFormer học cùng đã tự thích nghi**: nó tự sửa để phân đoạn tốt trên ảnh hiện có, nên Student không cần đổi ảnh. Phần lớn độ lệch so với bản bỏ SAM đến từ việc đổi loss nội dung, không phải từ học phân đoạn.

**Hướng tiếp (chưa chạy, cần chọn):**
1. Cho Student học quen loss nội dung của M11 trước, để tách riêng phần thay đổi do phân đoạn gây ra.
2. Hạn chế SegFormer tự thích nghi: giảm tốc độ học của nó sau giai đoạn khởi động, hoặc dùng một SegFormer "thầy" train trên **nửa dữ liệu khác** (thầy chéo, để thầy không thuộc ảnh đang dạy).
3. Nếu cả hai không làm ảnh đổi rõ: kết luận M11 không tạo khác biệt, và dồn sức vào nền B (TB + phần sửa).

### 4.6. Tìm ra lỗi BatchNorm, chạy lại cả hai nền với CE (19/11)

**Lỗi:** trong code học cùng (`m11_hoc_chung.py`), Student để ở chế độ train suốt quá trình. Khi đó lớp BatchNorm dùng thống kê của từng batch nhỏ. Nhưng lúc xuất ảnh thật, Student chạy ở chế độ eval (dùng thống kê đã lưu). Hai chế độ cho ảnh khác nhau.

**Bằng chứng** (`code/scene_v1/kiem_bn.py`, trên Student `tbsua_w150`): cùng một ảnh, hai chế độ lệch nhau trung bình 1,51 mức xám; loss giữ ảnh là 1,43 ở chế độ train nhưng chỉ 0,74 ở chế độ eval. Con số này khớp với log cũ: loss nội dung tụt từ 1,28 xuống 0,76 ngay epoch Student bắt đầu học.

**Hệ quả:** trong các lần chạy M11 trước (mục 4.5), Student mất phần lớn sức để **sửa chỗ lệch BatchNorm**, chứ không học từ loss phân đoạn. Đây là một lý do khiến ảnh gần như không đổi. Lần chạy `m11_kp` trong mục 4.7 vẫn là bản có lỗi này.

**Sửa:** thêm cờ `--bn_dong`. Khi bật cờ, BatchNorm của Student để ở chế độ eval cả lúc train. Sau khi sửa, loss nội dung lúc khởi động là 0,747, khớp với chế độ eval (0,74).

**Chạy lại** (30 epoch: 15 epoch đầu chỉ SegFormer-B0 học, 15 epoch sau Student học cùng; sau khởi động B0 học chậm lại 10 lần):

| Bản | Khởi tạo | λ (CE) | lr Student | Ảnh đổi so với bản khởi tạo |
|---|---|---|---|---|
| `m11a_l5_e30_bn` (nền A) | bỏ SAM `nosam_kp841` | 5 | 5e-4 | trung bình 4,57 mức xám |
| `m11b_l15_e30_bn` (nền B) | `tbsua_w150` | 15 | 5e-4 | trung bình 0,81, nhưng person 3,15, car 1,83, đèn giao thông 1,63 |

- Nền B đổi ít nếu tính trung bình cả ảnh, nhưng đổi tập trung ở vật thể. Tức là CE sửa đúng chỗ. **Lọc theo độ lệch trung bình cả ảnh là tiêu chí sai**, nên mình bỏ tiêu chí này.
- Cả hai vẫn qua sàn ảnh ghép (mục 4.4).

### 4.7. Sàng lọc bằng mIoU (tầng 1), seed 0 (19/11)

SegFormer-B0 train trên 841 ảnh, 100 epoch, batch 8, 14 lớp (bỏ nhãn 0). Chấm val 151 bằng checkpoint cuối. TB, SAGE, bỏ SAM, `m11_kp`, `tbsua_w150` chạy trên Kaggle (T4). Hai bản sửa BN chạy trên máy nhà (RTX 3050), cùng code và cùng cấu hình.

| Ảnh ghép | mIoU (14 lớp) | mIoU 7 lớp in trong bài |
|---|---|---|
| **Nền A + CE, sửa BN** | **50,34** | 63,04 |
| `m11_kp` (nền A + CE, còn lỗi BN) | 49,81 | 61,57 |
| Bỏ SAM `nosam_kp841` | 49,78 | 61,73 |
| **Nền B + CE, sửa BN** | **49,69** | **63,17** |
| TB + phần sửa `tbsua_w150` | 49,26 | 62,62 |
| TB | 48,14 | 61,57 |
| SAGE | 47,92 | 61,74 |

So theo cặp (cùng 151 ảnh; khoảng tin cậy bootstrap theo khối 10 ảnh liền số):

| Cặp (bản sau trừ bản trước) | Hiệu | Khoảng tin cậy 95% | Kết luận |
|---|---|---|---|
| Nền B + CE − TB | +1,55 | [+0,54; +2,75] | hơn chắc chắn |
| Nền B + CE − SAGE | +1,76 | [−0,65; +3,02] | chưa phân biệt |
| Nền A + CE − TB | +2,21 | [−0,60; +2,99] | chưa phân biệt |
| Nền A + CE − SAGE | +2,42 | [−0,67; +3,13] | chưa phân biệt |
| Nền A + CE − bỏ SAM (tác dụng của CE) | +0,57 | [−0,11; +1,57] | chưa phân biệt |
| Nền B + CE − TB + phần sửa (tác dụng của CE) | +0,43 | [−0,28; +1,56] | chưa phân biệt |
| Bỏ SAM − SAGE | +1,85 | [−1,63; +2,31] | chưa phân biệt |

Một số khoảng tin cậy không đối xứng quanh hiệu, vì mIoU tính trên cả tập chứ không phải trung bình từng ảnh.

**Đọc:**
- Ở seed 0, cả hai bản có CE đều hơn **max(TB, SAGE) từ 1 điểm trở lên**: nền A +2,21, nền B +1,55. Đây là tiêu chí đã đặt trước.
- **Phần lớn mức tăng đến từ việc bỏ SAM**, chưa phải từ CE. Bản bỏ SAM chưa có CE đã hơn TB 1,64 điểm. CE chỉ thêm khoảng +0,5 so với nền của nó, và khoảng tin cậy vẫn qua 0.
- CE nâng rõ nhất **mIoU 7 lớp** (lớp phổ biến): nền A +1,31 so với bỏ SAM, nền B +0,55 so với tbsua_w150.
- Lớp hiếm dao động rất mạnh giữa các lần chạy. Ví dụ SAGE chỉ có motorcycle 19,7 (các bản khác 45–51), TB chỉ có traffic_light 30,8 (SAGE 47,8). Chừng đó đủ kéo mIoU lên xuống 1–2 điểm.

**Giới hạn của bảng này:** mới **1 seed**; hai bản sửa BN chạy khác máy với các mốc. Kết quả nhiều seed ở mục 4.8.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/` (ma trận nhầm lẫn từng ảnh, IoU từng lớp, log train). Tính lại bảng: `cd code/b3 && SAGE_RUNS=../../ket_qua/kham_pha/sang_loc_b0/runs python so_sanh_cap.py --bien_the tb sage nosam_kp841 m11_kp tbsua_w150 m11b_l15_e30_bn m11a_l5_e30_bn --model b0 --tap dev`.

### 4.8. Nhiều seed: bỏ SAM vẫn hơn, nhưng phần hơn nằm ở lớp hiếm (19–20/11)

**Lớp hiếm trong val 151:** motorcycle chỉ có 1 ảnh, bicycle 2 ảnh, traffic_light 13, bus 14, truck 22. IoU của lớp có 1–2 ảnh gần như là ngẫu nhiên, nhưng vẫn kéo mIoU 14 lớp lên xuống vài điểm.

Vì vậy từ đây mình báo **hai số**:
- **mIoU 10 lớp đủ ảnh** (lớp có từ 20 ảnh trong val 151 trở lên). Đây là **số chính** để chọn phương án.
- mIoU 14 lớp: báo kèm.

**SegFormer-B0, val 151** (Kaggle T4; riêng seed 0 của hai bản có CE chạy trên máy nhà, cùng code):

| Ảnh ghép | Số seed | mIoU 14 lớp | **mIoU 10 lớp đủ ảnh** | IoU person |
|---|---|---|---|---|
| Bỏ SAM `nosam_kp841` | 3 | **49,91** | **57,50** | 44,59 |
| Nền B + CE | 2 | 49,15 | **57,51** | 48,21 |
| SAGE | 3 | 47,85 | 57,25 | 46,84 |
| Nền A + CE | 2 | 49,80 | 57,18 | 44,87 |
| TB | 2 | 47,73 | 56,91 | 46,73 |

So theo cặp (bootstrap theo khối ảnh và theo seed):

| Cặp | 14 lớp | 10 lớp đủ ảnh | Kết luận |
|---|---|---|---|
| Bỏ SAM − SAGE (3 seed) | +2,06 [−0,78; +2,86] | +0,25 [−0,60; +1,13] | chưa phân biệt |
| Bỏ SAM − TB (2 seed) | +2,04 [−1,13; +3,58] | +0,41 | chưa phân biệt |

- Bỏ SAM hơn SAGE ở cả 3 seed (14 lớp: +1,85 / +1,72 / +2,59).
- Nhưng trên 10 lớp đủ ảnh, phần hơn chỉ còn +0,25. Tức là **gần như toàn bộ phần hơn đến từ motorcycle và bicycle**, mỗi lớp chỉ 1–2 ảnh. Ví dụ motorcycle: bỏ SAM 41,6, SAGE 20,0, mà chỉ trên **1 ảnh**.
- Kết luận đúng: trên lớp đủ ảnh, **bỏ SAM ≈ SAGE ≈ TB**. Bỏ SAM vẫn có lợi vì rẻ hơn rất nhiều (mục 2.2), nhưng chưa hơn về mIoU.

**SegFormer-B3, val 151** (Kaggle, cùng cách chấm, checkpoint cuối):

| Ảnh ghép | Seed | mIoU 14 lớp | mIoU 10 lớp đủ ảnh | IoU person |
|---|---|---|---|---|
| Bỏ SAM | 0 / 1 | 61,20 / 60,37 | 64,18 / 63,82 | 50,64 / 50,49 |
| TB | 0 | 58,34 | 64,09 | 55,51 |
| SAGE | 0 | 57,88 | 63,63 | 54,73 |

- Seed 0, 14 lớp: bỏ SAM − SAGE +3,32 [−0,35; +5,61], bỏ SAM − TB +2,85. 10 lớp đủ ảnh: +0,55 và +0,09. Cùng hình ảnh với B0: hơn nhờ lớp hiếm.
- **Person:** bỏ SAM kém SAGE và TB khoảng 4–5 điểm trên B3, và kém khoảng 2 điểm trên B0. Đây là điểm yếu thật, lặp lại ở cả hai loại SegFormer (phân tích ở mục 4.11).
- SAGE và TB mới có 1 seed trên B3, nên chưa kết luận được.

**Mục tiêu nhóm chốt 20/11 (đã đổi ngày 21/11, xem mục 2.4):** phương pháp không SAM phải hơn **cả SAGE và TB ít nhất 5% tương đối**, tính trên mIoU 10 lớp đủ ảnh. Trên B0 nghĩa là khoảng **60,1** (mốc cao nhất hiện là SAGE 57,25). Bỏ SAM đang ở 57,50, **còn thiếu khoảng 2,6 điểm**.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/` (`runs/`, `so_sanh_cap_dev_b0.csv`, `so_sanh_lop_du_dev_b0.csv`) và `ket_qua/kham_pha/sang_loc_b3/`. Tính lại bảng 10 lớp: `cd code/b3 && SAGE_RUNS=../../ket_qua/kham_pha/sang_loc_b0/runs python so_sanh_lop_du.py --bien_the nosam_kp841 sage --model b0 --tap dev`.

### 4.9. CE (học cùng SegFormer) thêm được bao nhiêu (19/11)

Đối chứng: chạy lại nền A **giống hệt**, chỉ đặt λ = 0 (không có loss phân đoạn). Hiệu giữa hai bản là tác dụng thật của CE.

| | 14 lớp | 10 lớp đủ ảnh |
|---|---|---|
| Nền A, λ = 5 (có CE), seed 0 | 50,34 | 57,24 |
| Nền A, λ = 0 (không CE), seed 0 | 49,68 | 57,08 |
| Hiệu | +0,67 [+0,06; +1,24] | +0,16 |

- Ở seed 0, CE có tác dụng nhưng nhỏ, và phần lớn nằm ở lớp hiếm.
- Ở seed 1, nền A có CE (49,25) lại **thấp hơn** bỏ SAM (49,75). Gộp 2 seed, nền A − bỏ SAM = −0,03.
- Ảnh của bản λ = 0 cũng lệch bản bỏ SAM 4,34 mức xám, gần bằng bản có CE (4,57). Vậy phần lớn thay đổi ảnh đến từ loss nội dung, không phải từ CE.

**Kết luận:** với cách học cùng hiện tại, CE gần như đã chạm trần. Không đầu tư thêm vào hướng này.

### 4.10. Ba hướng đã thử và rớt (19–20/11)

Ngưỡng đặt trước cho cả ba: hơn bản bỏ SAM **cùng máy, cùng seed** ít nhất +1,0 điểm.

| Hướng | Ý tưởng | Kết quả | Vì sao dừng |
|---|---|---|---|
| Router: chọn cách ghép theo từng ảnh (19/11) | Bạn trong nhóm đề xuất | Kể cả chọn **hoàn hảo** từng ảnh giữa 5 bản (dùng luôn đáp án), chỉ hơn bản tốt nhất +1,32 (seed 0) / +1,41 (seed 1) | Chỉ chọn giữa 2 seed của **cùng một bản** đã được +0,45 đến +1,28. Tức là phần lớn lợi là ăn may. Router thật còn kém hơn, lại phải chạy nhiều mạng ghép |
| TDFusion, CVPR 2025 (20/11) | Mạng ghép có trọng số loss học từ tác vụ phân đoạn. Dùng checkpoint FMB công bố | 10 lớp: 57,30 (seed 0) / 57,98 (seed 1), so với bỏ SAM 56,99 (seed 0, máy nhà): +0,31 [−0,33; +1,18]. 14 lớp: −0,08 | Không đạt +1,0. Thêm nữa, checkpoint này train trên đủ 1.220 ảnh train FMB, nên có thể đã thấy val 151 |
| A1: tăng trọng số IR (20/11) | Bản bỏ SAM nghiêng về VIS (IR chiếm 0,4 trong loss nội dung). Thử 0,5 / 0,6 / 0,7 | 10 lớp 56,80 / 56,72 / 56,90, đều **thấp hơn** 56,99. Person không tăng đều (45,6 / 43,8 / 46,2 so với 45,1). Bản 0,6 còn rớt sàn MS-SSIM (98,7%) | Tăng IR chỉ tăng tương phản **đều cả ảnh**, không làm người nổi bật riêng (mục 4.11) |

Ảnh TDFusion vẫn qua sàn chỉ số ghép (SCD 103,5 / MS-SSIM 101,3 / Q_ABF 105,7 % SAGE).

### 4.11. Bỏ SAM thua SAGE ở đâu: người, nhất là người nhỏ (20/11)

Phân tích trên B0 val 151 (bỏ SAM và SAGE 3 seed, TB 2 seed). Ảnh có người được chia 3 nhóm bằng nhau theo kích thước người và theo độ sáng ảnh.

| Nhóm ảnh (IoU person) | Bỏ SAM | SAGE | TB | Bỏ SAM − SAGE |
|---|---|---|---|---|
| Người nhỏ | 24,9 | 33,4 | 28,3 | **−8,4** |
| Người vừa | 53,6 | 56,0 | 54,6 | −2,5 |
| Người lớn | 46,0 | 47,4 | 48,3 | −1,4 |
| Ảnh tối | 38,2 | 38,6 | 39,3 | −0,4 |
| Ảnh vừa | 59,5 | 62,9 | 62,4 | −3,3 |
| Ảnh sáng | 52,1 | 58,1 | 55,3 | −6,0 |

- Bỏ SAM thua nặng nhất ở **người nhỏ**. Ở ảnh tối thì gần bằng (trước đây mình đoán sai là thua ở ảnh tối).
- Pixel person bị đoán thành car: bỏ SAM 29,4%, SAGE 26,2%.

**Đo tương phản người với vùng xung quanh** (độ sáng trung bình của người trừ vành 15 px quanh người, trên kênh Y của ảnh ghép, trung vị 116 ảnh có người):

| | VIS | IR | TB | Bỏ SAM | SAGE |
|---|---|---|---|---|---|
| Tương phản người – vành | 9,3 | 61,2 | 26,8 | 23,5 | **44,5** |
| Tỉ lệ so với độ tương phản chung của ảnh | – | – | 1,05 | 0,97 | **1,32** |

- Tỉ lệ > 1 nghĩa là người nổi bật **hơn** phần còn lại của ảnh. SAGE làm người nổi bật **có chọn lọc**. Bỏ SAM thì không. Bỏ SAM thấp hơn SAGE ở **100%** số ảnh.
- Tăng trọng số IR (A1, bản 0,7) chỉ nâng tỉ lệ lên 1,08, vì nó tăng tương phản đều cả ảnh. Điều này giải thích vì sao A1 rớt.
- **Kết luận:** lợi thế thật của SAGE (nhờ SAM) là **làm nổi vật thể có chọn lọc**. Muốn bỏ SAM mà vẫn giữ được điều này, cần một loss nhắm vào vùng vật thể.

Thêm (chỉ ghi kèm): mIoU chia theo độ sáng cho thấy bỏ SAM hơn SAGE +3,11 ở nhóm ảnh độ sáng vừa, thua −1,50 ở ảnh tối và −0,79 ở ảnh sáng. **Hạn chế:** trong val 151, mỗi nhóm độ sáng gần như trùng với một đoạn video (ví dụ 51/51 ảnh sáng thuộc đoạn 01223–01300), nên chưa tách được ảnh hưởng của độ sáng khỏi ảnh hưởng của cảnh.

Số liệu gốc: `ket_qua/kham_pha/phan_tich/` (`person_theo_nhom.csv`, `tuong_phan_nguoi.csv`, `nham_person_sidewalk.csv`), `ket_qua/kham_pha/sang_loc_b0/miou_theo_sang_b0.csv`.

### 4.12. A2: loss tương phản theo vùng vật thể, không SAM (20/11)

**Ý tưởng:** khi train Student bỏ SAM, thêm một loss buộc vật thể trong ảnh ghép **nổi bật ít nhất bằng nguồn nổi bật hơn** (VIS hoặc IR).

- Vùng vật thể lấy từ **nhãn phân đoạn** của 841 ảnh train: mỗi người, xe hơi, xe tải, xe buýt, xe máy, xe đạp là một vùng (tối đa 16 vùng mỗi ảnh cắt, ưu tiên vật nhỏ).
- Với mỗi vùng: c(ảnh) = |độ sáng trung bình của vật − độ sáng trung bình vành 7 px quanh vật|.
- Loss = λ × trung bình trên các vật của max(0, max(c(VIS), c(IR)) − c(ảnh ghép)). Tức là chỉ phạt khi ảnh ghép làm vật **mờ hơn** nguồn tốt nhất.
- Nhãn chỉ dùng lúc train. Lúc chạy, Student y hệt cũ (0,136M tham số, không cần nhãn, không cần SAM). Teacher của SAGE cũng dùng nhãn lúc train, nên điều kiện so sánh tương đương.

**Kết quả λ = 10 (seed 0, máy nhà, so với bỏ SAM cùng máy):**

| | Bỏ SAM | A2, λ = 10 | SAGE (tham chiếu) |
|---|---|---|---|
| Tương phản người – vành | 23,5 | **55,7** | 44,5 |
| Tỉ lệ so với tương phản chung | 0,97 | **1,79** | 1,32 |
| SCD / MS-SSIM / Q_ABF (% SAGE) | 88,9 / 101,4 / 108,2 | 95,9 / 100,5 / 105,1 | 100 / 100 / 100 |
| **mIoU 10 lớp đủ ảnh** | 56,99 | **57,44** | – |
| mIoU 14 lớp | 50,33 | 49,63 | – |
| IoU person | 45,08 | 46,88 | – |
| IoU person, người nhỏ | 27,3 | **34,1** | 33,6 |

- Loss làm người nổi bật có chọn lọc, **mạnh hơn cả SAGE**, mà không cần SAM. Ảnh vẫn qua sàn, kể cả SCD.
- 10 lớp đủ ảnh: **+0,45** [−0,22; +1,64], chưa phân biệt. **Chưa đạt** ngưỡng +1,0 đặt trước.
- Person tăng +1,8, người nhỏ tăng +6,9 (nay ngang SAGE). Sidewalk +4,1. Traffic_sign −1,8.
- 14 lớp giảm 0,70, chủ yếu do traffic_light (13 ảnh, −10,9) và motorcycle (1 ảnh, −4,1), đều là lớp ít ảnh.
- Tỉ lệ pixel bị đoán nhầm thành person **giảm** (30,6% → 28,4%), tức là loss không làm mạng đoán "person" bừa.

**Đọc:** cơ chế đúng như mục 4.11 dự đoán, nhưng mức tăng còn nhỏ và mới có 1 seed.

Số liệu gốc: `ket_qua/kham_pha/phan_tich/a2_nosam_kp841_vung10/`, `ket_qua/kham_pha/dev_kp_vung_tong_hop.csv`, log ở `ket_qua/kham_pha/log_train_a1_a2/`. Checkpoint Student: `code/b3/sage_student/nosam_kp841_vung10.pt`.

### 4.13. Cận trên: SegFormer nhận thẳng VIS và IR, không ghép (20–21/11)

**Câu hỏi:** nếu SegFormer được nhìn **cả VIS lẫn IR** (không phải một ảnh ghép), mIoU lên tới đâu? Nếu cao hơn ảnh ghép nhiều thì ảnh ghép đang làm mất thông tin, và còn chỗ để cải thiện.

Ba cách cho SegFormer-B0 nhận thẳng VIS + IR (cùng giao thức sàng lọc: train 841, chấm val 151, seed 0, máy nhà):

| Cách | Mô tả | 10 lớp đủ ảnh |
|---|---|---|
| `vis_ir4` (20/11) | Ghép VIS (3 kênh) + IR (1 kênh) thành ảnh 4 kênh, sửa tầng đầu nhận 4 kênh | 56,80 |
| `vis_ir4_lr10` (21/11) | Như trên, tầng đầu học nhanh gấp 10 lần | 56,27 |
| **`hai_nhanh`** (21/11) | Hai bộ mã hóa riêng cho VIS và IR, trộn đặc trưng ở mỗi tầng, chung bộ giải mã. 7,2M tham số (gấp đôi B0) | **59,27** |
| Mốc: SAGE / TB / bỏ SAM / chỉ VIS | | 56,90 / 56,98 / 56,99 / 56,27 |

- Cách 4 kênh (`vis_ir4`) **không phải trần**: tầng đầu chỉ quen ảnh màu, kênh IR thêm vào học rất chậm. Tăng tốc độ học cũng không giúp.
- `hai_nhanh` hơn max(SAGE, TB) **+2,30**, "hơn chắc chắn" so với cả SAGE [+1,12; +3,14] và TB [+1,17; +3,34]. Tức là **IR còn thông tin mà mọi ảnh ghép hiện tại đang làm rơi**.
- `hai_nhanh` **không phải phương pháp nộp** (to gấp đôi, không tạo ảnh ghép). Nó chỉ cho biết còn chỗ, và chỗ đó ở lớp nào.

IoU từng lớp (seed 0), `hai_nhanh` trừ max(SAGE, TB):

| sidewalk | traffic_sign | person | road | building | car | traffic_light | bus |
|---|---|---|---|---|---|---|---|
| **+7,3** | **+5,4** | **+5,1** | +2,2 | −1,0 | −1,3 | −3,2 | −10,9 |

Bus chỉ có 14 ảnh và traffic_light 13 ảnh, nên hai số âm cuối dao động mạnh. **Đang chạy seed 1, 2** để xác nhận (máy nhà, 21/11).

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs_may_nha/{vis_ir4,vis_ir4_lr10,hai_nhanh,vis}_b0_s0/`, `ket_qua/kham_pha/sang_loc_b0/so_sanh_lop_du_dev_b0_tran_v2.csv`. Code: `code/b3/train_seg.py` (biến thể `vis_ir4`, `vis_ir4_lr10`, `hai_nhanh`), script `code/scene_v1/chay_tran_v2.sh`.

### 4.14. A2: chọn λ, rồi chạy đủ 3 seed (21/11)

**Thử thêm λ (đêm 20/11):** λ = 10 và 30 không được dò có hệ thống, nên thêm λ = 100. Luật chọn đặt **trước khi** có kết quả λ = 100: lấy λ có mIoU 10 lớp cao nhất và qua sàn ảnh; nếu hai λ cách nhau dưới 0,3 điểm thì lấy λ nhỏ hơn (ít ép ảnh hơn); chỉ chọn một lần.

| | Bỏ SAM | λ = 10 | λ = 30 | λ = 100 | λ = 10, thêm đèn/biển/cột |
|---|---|---|---|---|---|
| 10 lớp đủ ảnh | 56,99 | 57,44 | **57,66** | 56,02 | 57,02 |
| IoU person | 45,1 | 46,9 | 47,4 | 45,7 | 43,8 |
| Tương phản người – vành | 23,5 | 55,7 | 64,5 | 81,1 | – |
| MS-SSIM / Q_ABF (% SAGE) | 101,4 / 108,2 | 100,5 / 105,1 | 100,0 / 102,4 | **95,6 / 92,7** (rớt sàn) | – |

- λ = 100 ép quá tay: người rất nổi nhưng ảnh rớt sàn và mIoU giảm.
- Thêm đèn, biển báo, cột vào danh sách vật làm person giảm, nên bỏ.
- λ = 30 hơn λ = 10 chỉ 0,22 điểm (< 0,3), nên theo luật **chốt λ = 10**.

**Chạy đủ 3 seed (21/11):** train thêm Student λ = 10 seed 1 và 2 (máy nhà, khoảng 14 phút mỗi seed). Cả 3 seed đều qua sàn và rất ổn định:

| Seed | SCD | MS-SSIM | Q_ABF (% SAGE) |
|---|---|---|---|
| 0 | 95,9 | 100,5 | 105,1 |
| 1 | 96,8 | 101,3 | 104,7 |
| 2 | 96,8 | 100,7 | 104,2 |

**Đang chạy trên Kaggle (21/11):** SegFormer-B0 cho A2 seed 0, 1, 2 (Student seed s đi cặp SegFormer seed s), thêm TB seed 2 để mốc TB cũng đủ 3 seed. Chạy trên Kaggle để **cùng máy** với mốc SAGE và TB. Đích: "hơn nhất quán" so với cả SAGE và TB.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs_may_nha/nosam_kp841_vung{30,100,10mr}_b0_s0/`, `so_sanh_lop_du_a2_lam.csv`, `ket_qua/kham_pha/dev_kp_vung_*.csv`, `dev_a2_seed_*.csv`. Checkpoint: `code/b3/sage_student/nosam_kp841_vung10{,_s1,_s2}.pt`.

### 4.15. Ảnh ghép làm mất thông tin gì: không phải cạnh, mà là vùng (21/11)

Phép đo (b), (c), (d) dưới đây làm trên **ảnh train** (khoảng 210–420 ảnh), không dùng val 151.

**(a) Ảnh ghép thua VIS ở lớp nào** (IoU B0 trên val 151, trung bình các seed có sẵn; VIS chỉ 1 seed):

| Lớp | VIS | TB | SAGE | Bỏ SAM | A2 λ = 10 |
|---|---|---|---|---|---|
| person | 28,0 | 46,7 | 46,8 | 44,6 | 46,9 |
| traffic_light | **54,8** | 29,3 | 45,2 | 46,3 | 35,3 |
| sidewalk | **24,1** | 18,3 | 23,6 | 19,8 | 19,8 |
| truck | **13,0** | 2,0 | 3,7 | 4,7 | 2,9 |

Ảnh ghép giúp người, nhưng thua VIS ở vài lớp mà VIS thấy rõ. Một bạn trong nhóm thấy hiện tượng giống vậy trên SegFormer-B3 và Dev 228 (người tăng, xe tải và xe buýt giảm).

**(b) Có mất cạnh ở ranh giới giữa hai lớp không?** Đo độ mạnh cạnh (Sobel) ngay tại ranh giới hai lớp trong nhãn, so với nguồn có cạnh mạnh hơn (VIS hoặc IR):

| | Bỏ SAM | A2 λ = 10 | SAGE | TB |
|---|---|---|---|---|
| Giữ được bao nhiêu % cạnh | 87–94% | 88–96% | 70–94% | ~50% |

Ảnh ghép giữ cạnh tốt. TB mất một nửa cạnh mà mIoU vẫn ngang SAGE. Vậy **mất cạnh không phải nguyên nhân**, và hướng "loss giữ cạnh ở ranh giới" (gọi tạm A3) bị **bỏ trước khi chạy**.

**(c) Tương phản vật – vành theo lớp** (trung vị): với đèn và biển báo, bỏ SAM và A2 giữ tương phản **bằng VIS** (đèn 11,2 và 10,3 so với VIS 10,9; biển báo 19,1 và 19,6 so với 18,7). Chỉ TB làm giảm một nửa. Vậy A2 tụt ở đèn (35,3) nhiều khả năng là dao động seed (đèn chỉ có 13 ảnh; riêng bỏ SAM đã dao động 42,7–52,1 giữa các seed).

**(d) Độ tách đường – vỉa hè** = |độ sáng TB của đường − của vỉa hè| / độ lệch chuẩn chung, đo trong dải 20 px quanh ranh giới, trung vị 245 ảnh:

| VIS | IR | SAGE | TB | A2 λ = 10 | Bỏ SAM |
|---|---|---|---|---|---|
| 0,45 | **1,06** | 0,83 | 0,69 | 0,66 | **0,49** |

- IR tách đường với vỉa hè **tốt gấp đôi VIS** (hai mặt đường có nhiệt khác nhau).
- SAGE giữ được phần lớn. Bỏ SAM **mất gần hết**, chỉ còn ngang VIS. Khớp với IoU sidewalk: SAGE 23,6 > bỏ SAM 19,8; `hai_nhanh` (thấy thẳng IR) 28,4.
- Kết luận: thứ bị mất là **thông tin cả vùng** (độ sáng trung bình) của IR, không phải cạnh.

Số liệu gốc: `ket_qua/kham_pha/kiem_vung/canh_bien_train.csv`. Code: `code/scene_v1/kiem_a3/do_canh_bien.py`, `do_tuong_phan_lop.py`.

### 4.16. A4: A2 + loss giữ độ tách giữa hai vùng kề nhau (21/11)

**Ý tưởng:** A2 lo cho **vật** (người, xe). A4 thêm phần lo cho **vùng nền** (đường, vỉa hè, nhà…). Với mỗi cặp vùng kề nhau trong nhãn, độ tách trên ảnh ghép phải **ít nhất bằng** nguồn tách tốt hơn (VIS hoặc IR).

- Độ tách s = |độ sáng TB phía a − phía b| / căn của trung bình hai phương sai, chỉ tính trong dải khoảng 10 px mỗi bên ranh giới.
- Loss = trung bình trên các cặp của max(0, max(s(VIS), s(IR)) − s(ảnh ghép)). Lấy **mọi cặp lớp**, không gắn với lớp cụ thể nào (tối đa 8 cặp mỗi ảnh cắt), để dùng được trên dataset khác.
- Nhãn chỉ dùng lúc train, giống A2. Lúc chạy vẫn là Student 0,136M tham số.
- Trọng số λ_tách = 1, **đặt một lần** sao cho phần loss này chiếm khoảng 8% tổng lúc bắt đầu train (ngang phần loss của A2). Không dò thêm.

**Tiêu chí đặt trước** (A4 seed 0 so với A2 λ = 10 seed 0, cùng chấm trên Kaggle): (1) độ tách đường – vỉa hè tăng; (2) IoU sidewalk tăng; (3) 10 lớp ≥ A2 + 0,5; (4) qua sàn ảnh. Đạt cả 4 thì chạy seed 1, 2.

**Kết quả đến giờ:**
- Qua sàn: SCD 100,1 / MS-SSIM 99,9 / Q_ABF 104,4 (% SAGE).
- Độ tách đường – vỉa hè (ảnh train): 0,66 → **0,68**, chỉ nhích nhẹ (SAGE 0,83). Loss tách giảm chậm (0,47 → 0,38 sau 100 epoch), tức là Student khó làm theo.
- Phụ: biển báo, đèn, xe tải nổi hơn (tương phản biển báo 23,1 so với A2 19,6; đèn 12,3 so với 10,3; xe tải 18,1 so với 15,8).
- **mIoU đang chạy trên Kaggle (21/11).**

Checkpoint: `code/b3/sage_student/nosam_kp841_vung10_tach1.pt`. Code: `code/e17/train_student_nosam.py` (tùy chọn `--lam_tach`), script `code/scene_v1/chay_a4.sh`. Chỉ số ghép: `ket_qua/kham_pha/dev_a4_*.csv`.

---

## 5. Bước tiếp theo (21/11)

1. **Đọc kết quả Kaggle:** A2 có "hơn nhất quán" cả SAGE và TB qua 3 seed không. A4 có đạt 4 tiêu chí ở mục 4.16 không.
2. **Cận trên `hai_nhanh` seed 1, 2:** xác nhận mức +2,3 và các lớp còn chỗ.
3. Nếu A4 không tăng sidewalk: kiểm cách chọn cặp vùng (đang ưu tiên cặp diện tích lớn). Hướng sửa vẫn phải **không gắn với lớp cụ thể**: ưu tiên cặp đang tách kém nhất, hoặc cân đều các cặp.
4. **Vòng chọn trên Dev 228:** phương án đã khóa → train lại Student trên 992 ảnh → SegFormer-B3, 3 seed, chấm Dev 228 một lần. Mốc SAGE, VIS, TB × 3 seed đã có notebook Kaggle. Lần chạy đầu lỗi do quên bật Internet, đang chạy lại. Nếu được chọn thì chấm thêm Test 280.
5. **Kiểm tổng quát (chốt 21/11):** mạng ghép **chỉ train trên FMB**, rồi đem chạy thẳng trên dataset khác, không train lại. SegFormer vẫn train và chấm trên dataset mới, như nhau cho mọi phương pháp.
   - Bộ chính: **MSRS** (bản căn chỉnh lại của MFNet; bài SAGE có báo số trên MFNet; có lớp người, xe, lề đường). Phải kiểm rò train/test trước.
   - Bộ phụ: **SemanticRT** (nhiều ảnh đêm; split gốc bị rò nặng nên phải chia lại theo góc máy).
   - Không train mạng ghép trên MSRS, để MSRS còn là dữ liệu "lạ" thật.

## 6. Cần nhóm chốt

| Việc | Đề xuất |
|---|---|
| Số chính để chọn phương án | mIoU **10 lớp đủ ảnh** (lớp có từ 20 ảnh trong tập chấm). mIoU 14 lớp báo kèm, vì lớp chỉ có 1–2 ảnh làm số dao động mạnh |
| Mốc trên Dev 228 | SAGE, VIS, TB × 3 seed, cùng code, cùng Kaggle (notebook đã có). Seed 0 dùng đối chiếu với số đã có của nhóm (SAGE 59,05; VIS 59,52; TB 60,04) |
| Cách train SegFormer-B3 | Nhãn 0 (unlabeled) có là một lớp lúc train không. Cách chấm mIoU 13 lớp thì giống nhau |
| Dataset kiểm tổng quát | MSRS là bộ chính, SemanticRT phụ (mục 5) |

## 7. Giới hạn

- Số trên tập khám phá chỉ để so các phương án với nhau. Không báo cáo như kết quả cuối.
- Tập khám phá (val 151) đã được nhìn nhiều lần để định hướng (ví dụ ý tưởng A4 xuất phát một phần từ IoU sidewalk trên val 151). Vì vậy **kết quả để báo cáo phải lấy từ Dev 228, Test 280 và dataset khác**.
- SegFormer-B0 yếu hơn B3. Trước đây, chênh lệch đo trên B0 thường lớn hơn trên B3.
- Chênh lệch khoảng 1 điểm mIoU giữa TB, bỏ SAM và SAGE xấp xỉ mức dao động do seed. Phải có nhiều seed và so theo cặp seed mới kết luận được.
- Một số lần chạy trên máy nhà (RTX 3050), mốc trên Kaggle (T4). Cùng code, nhưng khác máy có thể lệch khoảng 0,5 điểm. Vì vậy từ 21/11, SegFormer chấm A2/A4 đều chạy trên Kaggle. Mạng ghép (Student) train ở đâu cũng được, vì chỉ cần file trọng số.
- Checkpoint SAGE (và TDFusion) được train trên đủ 1.220 ảnh train FMB, nên đã thấy val 151 và Dev 228. Bản bỏ SAM của mình chỉ train trên 841 ảnh. Nếu có lợi thế thì đang nghiêng về SAGE. Repo SAGE chỉ có code train cho FMB, nên nhiều khả năng SAGE chưa thấy MSRS.

---

## Phụ lục A: dữ liệu gốc đi kèm (không cần đọc để hiểu)

| Thư mục | Có gì |
|---|---|
| `ket_qua/e17_bo_sam/` | So SAGE với bỏ SAM (mục 2.1). `runs_b3/` chứa 17 lần chạy SegFormer-B3 (VIS, IR, TB, SAGE, bỏ SAM × 3 seed, thêm 2 Student bỏ SAM khác): ma trận nhầm lẫn từng ảnh (`test_nham_last.npz`), cấu hình, log. Tính lại bảng bằng: `cd code/b3 && SAGE_RUNS=../../ket_qua/e17_bo_sam/runs_b3 python so_sanh_cap.py --bien_the vis tb sage nosam --model b3 --hau _1220_15lop` |
| `ket_qua/e18_chi_phi/` | Bảng chi phí đo trên T4 (mục 2.2) |
| `ket_qua/danh_doi/` | Chỉ số ghép so với mIoU (mục 2.3) |
| `ket_qua/split_cu/` | Số của các phương án mục 3 |
| `ket_qua/kham_pha/` | Chỉ số ghép trên val 151, sàn, log train (mục 4.2, 4.4, 4.6). `dev_kp_td_*` (TDFusion), `dev_kp_wir_*` (A1), `dev_kp_vung_*` (A2) cho mục 4.10, 4.12 |
| `ket_qua/kham_pha/sang_loc_b0/` | mIoU sàng lọc B0 (mục 4.7–4.9): `runs/` có 15 lần chạy, chủ yếu trên Kaggle (ma trận nhầm lẫn `dev_nham_last.npz`, IoU từng lớp, log); `runs_may_nha/` có các lần chạy trên máy nhà (bỏ SAM seed 0, A1, A2 λ = 10/30/100/mở rộng lớp, TDFusion, chỉ VIS, cận trên `vis_ir4`/`vis_ir4_lr10`/`hai_nhanh` – mục 4.10, 4.12–4.14); `so_sanh_lop_du_dev_b0_tran_v2.csv` (cận trên, mục 4.13); `so_sanh_cap_dev_b0.csv` (14 lớp), `so_sanh_lop_du_dev_b0.csv` (10 lớp đủ ảnh), `miou_theo_sang_b0.csv` |
| `ket_qua/kham_pha/sang_loc_b3/` | SegFormer-B3 trên val 151 (mục 4.8): 4 lần chạy, so cặp 14 lớp và 10 lớp |
| `ket_qua/kham_pha/phan_tich/` | Phân tích person, tương phản người, nhầm lớp (mục 4.11); `a2_nosam_kp841_vung10/` cho mục 4.12; `proxy/` là thử chấm nhanh bằng SegFormer đóng băng (không dùng, vì phạt oan ảnh khác kiểu bỏ SAM) |
| `ket_qua/kham_pha/log_train_a1_a2/` | Log train Student A1, A2 (mọi λ, 3 seed), A4 và log chạy cả quy trình (`*.out.txt`: cận trên, A2 λ, A2 seed, A4) |
| `ket_qua/kham_pha/kiem_vung/` | Độ mạnh cạnh tại ranh giới hai lớp trên ảnh train (mục 4.15) |
| `ket_qua/chan_doan_m11/` | Log đo lực kéo (mục 4.5), các lần chạy nền B trước khi sửa BN |
| `hinh/` | Hình minh họa |
| `code/` | Code chạy được, xem Phụ lục B |

## Phụ lục B: chạy lại

### Cấu trúc `code/`

| Thư mục | Có gì |
|---|---|
| `b3/` | `chung.py` (đường dẫn, danh sách lớp, mIoU), `xuat_anh.py` (xuất ảnh ghép), `train_seg.py` / `eval_seg.py` (train, chấm SegFormer B0/B3), `so_sanh_cap.py` (so theo cặp seed), `so_sanh_lop_du.py` (so cặp chỉ trên lớp đủ ảnh). `train_seg.py` có thêm biến thể `vis_ir4`, `vis_ir4_lr10`, `hai_nhanh` (SegFormer nhận thẳng VIS + IR, để đo cận trên, mục 4.13) |
| `b3/sage_student/` | `model.py` (kiến trúc Student của SAGE) và **checkpoint Student đã train** (mỗi file khoảng 0,6 MB): `nosam*` (E17, 5 seed), `nosam_kp841*` (tập khám phá, 5 seed), `m11_kp`, `tbsua_kp841`, `tbsua_w150`, `m11a_l0_e30_bn` (đối chứng λ = 0), `nosam_kp841_w05/w06/w07` (A1), `nosam_kp841_vung10{,_s1,_s2}` (A2 đã chốt, 3 seed), `nosam_kp841_vung30/vung100/vung10mr` (A2 các λ khác), `nosam_kp841_vung10_tach1` (A4) |
| `e17/` | Train Student bỏ SAM (`train_student_nosam.py`; thêm tùy chọn `--w_ir` cho A1, `--lam_vung` và `--lop_vat` cho A2, `--lam_tach` cho A4, `--them` để train thêm dữ liệu ngoài như MSRS); so sánh E17 |
| `cai_tien/` | M11 (`m11_hoc_chung.py`, dùng hàm trong `train_student_seg.py`), nền B (`tb_sua.py`), chỉ số ghép (`chi_so_ghep.py`, `cong2_ci.py`), độ lệch ảnh (`lech_y.py`), xuất ảnh TDFusion (`xuat_tdfusion.py`, cần clone `HaowenBai/TDFusion` kèm checkpoint). `danh_doi/` là script phân tích, đường dẫn còn cứng, chỉ để đọc |
| `scene_v1/` | Script chạy từng thí nghiệm; phân tích (`phan_tich_nosam.py`, `phan_tich_a2.py`, `do_tuong_phan.py`, `miou_theo_sang.py`, `kiem_proxy.py`, `oracle/oracle.py`, `kiem_a3/` đo cạnh và độ tách vùng mục 4.15); `kham_pha/` (train 841 / val 151); `goc/` (split scene_v1 992/228 của nhóm, kèm `scene_decision.md`) |
| `chi_phi/` | Đo chi phí (E18) |
| `pr6_eval/` | Code chỉ số ghép của Lộc lớn (SCD, MS-SSIM, Q_ABF…), chép nguyên `metrics.py` |
| `QUY_TAC_CHAM.md`, `moi_truong.txt` | Quy tắc so sánh của nhóm; phiên bản thư viện |

### Cần chuẩn bị

1. Dataset **FMB** (repo không kèm). Cần có `train/Visible`, `train/Infrared`, `train/Label`, và tương tự cho `test`.
2. Checkpoint Student SAGE gốc `checkpoints.pt` (từ repo `RollingPlain/SAGE_IVIF`), đặt vào `code/b3/sage_student/`. Chỉ cần khi xuất ảnh `sage`.
3. Biến môi trường:
   ```bash
   export SAGE_FMB=<thư mục FMB>          # nơi có train/Visible, test/Visible
   export SAGE_ANH=<nơi ghi ảnh ghép>     # mặc định code/b3/anh
   export PY=python                        # hoặc đường dẫn python có torch
   ```

### Lệnh (chạy bằng bash, từ thư mục `code/`)

| Việc | Lệnh | Thời gian (RTX 3050) |
|---|---|---|
| Xuất ảnh ghép từ checkpoint có sẵn | `cd b3 && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY xuat_anh.py --bien_the tbsua_w150` | vài phút |
| Train Student bỏ SAM + M11 (mục 4.3) | `bash scene_v1/chay_p1.sh` | khoảng 1,5 giờ |
| Đặt sàn từ 5 seed (mục 4.2) | `bash scene_v1/chay_san.sh` | khoảng 1 giờ |
| Nền B, w_tb = 150 (mục 4.4) | `bash scene_v1/chay_tbsua150.sh` | khoảng 15 phút |
| M11 bản sửa λ = 5 (mục 4.5) | `bash scene_v1/chay_m11_l5.sh` | khoảng 20 phút |
| Chẩn đoán lực kéo M11 | `cd cai_tien && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY m11_hoc_chung.py --ten m11_kp_dograd --init nosam_kp841 --epochs 8 --do_grad 10` | khoảng 12 phút |
| Sàng lọc mIoU bằng B0 | `bash scene_v1/chay_b0_sangloc.sh` | khoảng 1,2 giờ mỗi lượt |
| Kiểm lỗi BatchNorm (mục 4.6) | `cd cai_tien && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY ../scene_v1/kiem_bn.py tbsua_w150` | vài phút |
| Nền A + CE, sửa BN (mục 4.6) | `bash scene_v1/chay_nena_ce.sh` | khoảng 50 phút |
| Nền B + CE, sửa BN (mục 4.6) | `TEN=m11b_l15_e30_bn LAM=15 EPOCHS=30 BN=1 bash scene_v1/chay_nenb_ce.sh` | khoảng 1 giờ |
| mIoU B0 cho một bản (mục 4.7) | `TEN=m11b_l15_e30_bn BO_LOC=1 bash scene_v1/chay_nenb_miou.sh` | khoảng 1,2 giờ |
| Đối chứng λ = 0 (mục 4.9) | `TEN=m11a_l0_e30_bn GOC=nosam_kp841 BO_LOC=1 bash scene_v1/chay_nenb_miou.sh` (sau khi train Student với λ = 0) | khoảng 2 giờ |
| Router lý tưởng (mục 4.10) | `python scene_v1/oracle/oracle.py` | vài giây |
| TDFusion (mục 4.10) | `TDFUSION=<thư mục clone TDFusion> bash scene_v1/chay_b1_tdfusion.sh` | khoảng 4 giờ |
| A1 tăng trọng số IR (mục 4.10) | `bash scene_v1/chay_a1_wir.sh` | khoảng 4,5 giờ |
| Phân tích person, tương phản (mục 4.11) | `cd b3 && SAGE_MAN=../scene_v1/kham_pha SAGE_RUNS=../../ket_qua/kham_pha/sang_loc_b0/runs python ../scene_v1/phan_tich_nosam.py` (cần ảnh ghép trong `SAGE_ANH`) | vài phút |
| A2 loss theo vùng (mục 4.12) | `LAM=10 bash scene_v1/chay_a2_vung.sh` (`LAM="30 10mr"`: λ 30 và λ 10 mở rộng lớp) | khoảng 1,5 giờ mỗi λ |
| Trần 4 kênh (mục 4.13) | `bash scene_v1/chay_tran_4kenh.sh` | khoảng 2,5 giờ |
| Cận trên v2: `vis_ir4_lr10`, `hai_nhanh` (mục 4.13) | `CHO=0 bash scene_v1/chay_tran_v2.sh`; thêm seed: `bash scene_v1/chay_hai_nhanh_seed.sh` | khoảng 1,5 giờ mỗi lượt |
| A2 λ = 100 (mục 4.14) | `LAM="10 30 100" bash scene_v1/chay_a2_vung.sh` | khoảng 1,5 giờ |
| A2 λ = 10, seed 1 và 2 (mục 4.14) | `bash scene_v1/chay_a2_seed.sh` | khoảng 35 phút (chỉ Student; SegFormer chấm trên Kaggle) |
| Đo cạnh, độ tách vùng (mục 4.15) | `cd b3 && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY ../scene_v1/kiem_a3/do_tuong_phan_lop.py` (và `do_canh_bien.py`) | vài phút |
| A4 (mục 4.16) | `bash scene_v1/chay_a4.sh` | khoảng 25 phút (chỉ Student) |

Đã thử: xuất 3 ảnh `tbsua_w150` bằng code trong repo này, kết quả **giống hệt từng điểm ảnh** với bản gốc. Các script khác đã chạy thử `--help` và đều nạp được.
