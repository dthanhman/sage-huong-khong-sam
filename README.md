# Ghép ảnh IR–VIS không cần SAM: tổng kết hướng thử nghiệm

Người làm: Đào Thanh Mân. Cập nhật: **26/11/2026** (mục mới 4.26–4.33: `vhn`, oracle và cổng VIS trên SegFormer-B3; đổi nền sang TB (`tbsua`, `a04`); kiểm chéo theo cảnh; các hướng rớt 24–26/11; khóa phương án `a04` và giao thức Test; mục 5–7 viết lại).

**Tóm tắt kết quả đến 26/11:**
- Bỏ SAM rẻ hơn khoảng 250 lần lúc train, và mIoU **không thua** SAGE, cả trên tầng sàng lọc (B0, val 151) lẫn tầng chọn (SegFormer-B3, Dev 228, 3 seed) (mục 2.1, 4.8, 4.24).
- Trên SegFormer-B3, **vật cản là TB, không phải SAGE**. TB mạnh ở 8 lớp phổ biến. Các bản dựa trên nền Student bỏ SAM (`vhn`, `vnc`) hơn SAGE nhưng không hơn TB nhất quán (mục 4.26).
- **Đổi nền sang TB** cho kết quả tốt nhất. Phương án đã **khóa** là `a04` = TB + 0,4 × (phần sửa học được, không SAM, không nhãn). Trên Dev 228, 3 seed, 13 lớp: `a04` **60,77**, TB 60,15, SAGE 59,41 (mục 4.27).
  - `a04` − SAGE: **+1,36**, khoảng tin cậy 95% [+0,06; +3,09] → **hơn nhất quán**.
  - `a04` − TB: **+0,63**, khoảng tin cậy [−0,04; +1,26] → đạt biên "không kém TB" (chốt trước: cận dưới > −0,5), **chưa hơn chắc**.
  - 8 lớp phổ biến: `a04` 77,48 = TB 77,49. Luật đặt trước (≥ 77,62) **rớt**, phải ghi khi báo cáo.
- Kiểm chéo theo cảnh trên 411 ảnh Train (3 seed): `a04` − TB +0,47, khoảng tin cậy chứa 0. Lợi dồn vào vài ảnh bus (mục 4.28).
- Thử thêm cổng VIS lên `a04` (`a04_cvis`) để lấy lại đèn, xe tải, bus: **rớt**, lớp hiếm còn giảm (−2,55 so `a04`) (mục 4.30).
- Nhận định chính: **một ảnh ghép chưa giữ được cùng lúc** lợi của IR (person +19 so VIS) và lợi của VIS (truck +10,8, đèn +4,9, bus +2,0 so `a04`) (mục 4.30, 4.33).
- **Đang chạy (26/11):** Test 280 ảnh (chính: 273 ảnh sạch), SegFormer-B3 train 1.220 ảnh, `a04` 5 seed, mốc TB/SAGE thêm seed 3, 4. Chưa có số (mục 4.31).

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
| 21/11 chiều | A2 3 seed và A4 rớt; cận trên 3 seed đạt; phân tích vì sao rớt | 4.13, 4.14, 4.16, 4.17 |
| 21–22/11 | Đổi kiến trúc Student (nhánh ngữ cảnh, rộng hơn), loss giữ đèn; mốc B3 Dev 228 đủ 3 seed | 4.18, 4.24 |
| 22/11 | Đối chứng cận trên; kéo cường độ theo lớp; chỉnh màu; thêm MSRS; oracle theo vùng; bỏ SAM trên B3 Dev 228 | 4.13, 4.19–4.21, 4.24 |
| 22–23/11 | `vhn` 3 seed đạt trên val 151; chưng cất vào Student (4 kiểu), S-vùng | 4.22, 4.23 |
| 23/11 tối | `vhn` trên B3 Dev 228 (2 seed): hơn SAGE, chưa hơn TB; oracle B3 chỉ +1,88 so TB | 4.26 |
| 24/11 | Chỗ trống B3 nằm ở lớp hiếm; `vnc` (VIS + cổng nhiệt) rớt; `vhn3` (thầy B3) rớt; đổi nền sang TB: `tbsua_992` | 4.26, 4.27 |
| 24/11 tối | `tbsua_992` 3 seed: hơn SAGE nhất quán, chưa hơn TB; chọn α = 0,4 trên Train → `a04` | 4.27 |
| 25/11 | `a04` seed 0, 1; kiểm chéo theo cảnh `kc1` (3 seed); nhiều hướng sửa bằng nhãn rớt | 4.27–4.29 |
| 26/11 | `a04` seed 2 → **khóa `a04`**; `a04_cvis` rớt; đẩy Test | 4.27, 4.30, 4.31 |

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

Cập nhật 21/11: TB chạy thêm seed 2. TB đủ 3 seed: 14 lớp 47,72, **10 lớp 57,11**, person 46,62. Các mục sau dùng số 3 seed này.

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

Bus chỉ có 14 ảnh và traffic_light 13 ảnh, nên hai số âm cuối dao động mạnh.

**Đủ 3 seed (21/11) và đối chứng (22/11).** `hai_nhanh` to gấp đôi B0. Để biết phần hơn đến từ IR hay chỉ do mạng to, mình chạy đối chứng `hai_nhanh_visvis`: y hệt `hai_nhanh`, nhưng nhánh thứ hai nhận ảnh xám của VIS thay cho IR.

| | Seed 0 / 1 / 2 | **10 lớp (TB 3 seed)** |
|---|---|---|
| `hai_nhanh` (VIS + IR) | 59,27 / 59,40 / 59,58 | **59,42** |
| `hai_nhanh_visvis` (VIS + VIS) | 55,63 / 57,22 / 58,45 | 57,10 |
| Bỏ SAM / SAGE / TB (mốc) | | 57,50 / 57,25 / 57,11 |

| Cặp (3 seed) | Hiệu | Khoảng tin cậy 95% | Kết luận |
|---|---|---|---|
| `hai_nhanh` − SAGE | +2,17 | [+0,95; +3,10] | hơn chắc chắn |
| `hai_nhanh` − TB | +2,31 | [+0,85; +3,37] | hơn chắc chắn |
| `hai_nhanh` − `hai_nhanh_visvis` | +2,32 | [+0,70; +4,36] | hơn chắc chắn |
| `hai_nhanh_visvis` − bỏ SAM | −0,40 | [−2,57; +1,10] | chưa phân biệt |

- Mạng to gấp đôi mà không có IR thì **không hơn** bỏ SAM. Vậy phần hơn của `hai_nhanh` là do **thông tin IR**. Ảnh ghép một kênh đang làm rơi một phần thông tin này.
- IoU trung bình 3 seed, `hai_nhanh` so với SAGE: person 53,3 / 46,8; sidewalk 26,4 / 23,6; truck 11,2 / 3,7.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs_may_nha/{vis_ir4,vis_ir4_lr10,hai_nhanh,hai_nhanh_visvis,vis}_b0_s*/`, `ket_qua/kham_pha/sang_loc_b0/so_sanh_lop_du_dev_b0_tran_v2.csv`, `so_sanh_lop_du_dev_b0_visvis.csv`. Code: `code/b3/train_seg.py` (biến thể `vis_ir4`, `vis_ir4_lr10`, `hai_nhanh`, `hai_nhanh_visvis`), script `code/scene_v1/chay_tran_v2.sh`, `chay_hai_nhanh_seed.sh`, `chay_visvis.sh`.

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

**Kết quả 3 seed (Kaggle, 21/11):** SegFormer-B0 cho A2 seed 0, 1, 2 (Student seed s đi cặp SegFormer seed s), chạy trên Kaggle để **cùng máy** với mốc SAGE và TB.

| | Seed 0 / 1 / 2 | 10 lớp (TB 3 seed) |
|---|---|---|
| A2 λ = 10 | 57,18 / 56,87 / 57,22 | 57,09 |
| Bỏ SAM / SAGE / TB | | 57,50 / 57,25 / 57,11 |

| Cặp (3 seed) | Hiệu | Khoảng tin cậy 95% | Kết luận |
|---|---|---|---|
| A2 − SAGE | −0,16 | [−1,04; +0,94] | chưa phân biệt |
| A2 − TB | −0,02 | [−0,50; +0,71] | chưa phân biệt |
| A2 − bỏ SAM | −0,41 | [−0,94; +0,60] | chưa phân biệt |

- **A2 rớt.** Seed 0 chấm trên Kaggle được 57,18, còn trên máy nhà là 57,44. Tức là chạy lại cùng cấu hình đã lệch khoảng 0,3 điểm, bằng cỡ phần hơn đo được ở mục 4.12.
- Lý do rớt ở mục 4.17.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs_may_nha/nosam_kp841_vung{30,100,10mr}_b0_s0/`, `runs/a2l10_b0_s{0,1,2}/` (A2 λ = 10 chấm trên Kaggle), `so_sanh_lop_du_dev_b0_a2_3seed.csv`, `so_sanh_lop_du_a2_lam.csv`, `ket_qua/kham_pha/dev_kp_vung_*.csv`, `dev_a2_seed_*.csv`. Checkpoint: `code/b3/sage_student/nosam_kp841_vung10{,_s1,_s2}.pt`.

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
- **mIoU (Kaggle, seed 0):** 10 lớp 57,18, bằng A2 (57,18). IoU sidewalk 18,2, **thấp hơn** A2 (20,1). Biển báo tăng (47,6 so với 42,4) nhưng person giảm (46,6 so với 48,0).
- **A4 rớt** tiêu chí (2) và (3), nên không chạy seed 1, 2.

Checkpoint: `code/b3/sage_student/nosam_kp841_vung10_tach1.pt`. Code: `code/e17/train_student_nosam.py` (tùy chọn `--lam_tach`), script `code/scene_v1/chay_a4.sh`. Chỉ số ghép: `ket_qua/kham_pha/dev_a4_*.csv`. mIoU: `ket_qua/kham_pha/sang_loc_b0/runs/a4_b0_s0/`, `so_sanh_lop_du_dev_b0_a4_s0.csv`.

### 4.17. Vì sao A2, A4 rớt: ảnh ghép đánh đổi giữa các lớp (21–22/11)

**(a) Được lớp này, mất lớp kia.** So A2 với bỏ SAM (3 seed): person +2,4, nhưng traffic_sign −3,6. Cộng lại gần bằng 0. A4 so với A2 (seed 0): biển báo +5,2, nhưng sidewalk −1,8 và person −1,4.

**(b) Mọi ảnh ghép đều như vậy.** IoU trên B0 val 151 (trung bình 3 seed; VIS chỉ 1 seed):

| Lớp | VIS | SAGE | TB | Bỏ SAM | `hai_nhanh` |
|---|---|---|---|---|---|
| person | 28,0 | 46,8 | 46,6 | 44,6 | **53,3** |
| truck | **13,0** | 3,8 | 2,6 | 4,7 | 11,2 |
| traffic_light | **54,8** | 45,2 | 28,4 | 46,3 | 44,5 |
| sidewalk | 24,1 | 23,6 | 19,0 | 19,8 | **26,4** |
| traffic_sign | **47,5** | 44,3 | 43,5 | 46,3 | 45,1 |

- Ảnh ghép đưa person từ 28 lên khoảng 46 nhờ IR, nhưng **thua VIS** ở xe tải, đèn, biển báo, vỉa hè.
- `hai_nhanh` (thấy riêng cả hai nguồn) giữ được cả hai phía.
- SegFormer-B3 trên Dev 228 cũng cùng kiểu (mục 4.24): person VIS 51,6 so với ảnh ghép khoảng 70; truck VIS 36,9 so với SAGE 30,5 và TB 24,2.

**(c) Tăng tương phản người đã bão hòa.** Tỉ lệ tương phản người: bỏ SAM 0,97 / SAGE 1,32 / A2 1,79, nhưng IoU person chỉ 44,6 / 46,8 / 47,0. Làm người nổi hơn nữa không kéo IoU lên thêm.

**(d) Loss A2 làm đèn bị nhầm thành biển báo.** Loss A2 không có lớp đèn, nhưng Student học luật chung "IR nổi thì tô đậm", nên đèn cũng bị tô đậm và giống biển báo hơn. Số điểm ảnh đèn bị đoán thành biển báo (seed 0): bỏ SAM 2,4 nghìn, SAGE 2,8 nghìn, A2 6,2 nghìn.

Số liệu gốc: `ket_qua/kham_pha/phan_tich/nut_that_1_kenh.csv`. Code: `code/scene_v1/phan_tich_nut_that.py`.

### 4.18. Đổi kiến trúc Student, thêm loss giữ đèn (21–22/11)

Ý tưởng: Student gốc chỉ nhìn một vùng nhỏ quanh mỗi điểm ảnh, nên có thể không đủ ngữ cảnh để biết đâu là người. Thử cho Student nhìn rộng hơn hoặc to hơn.

Ngưỡng đặt trước: 10 lớp ≥ 57,99, person hơn bỏ SAM, biển báo không giảm quá 1 điểm, qua sàn ảnh. Chấm B0 val 151, seed 0. Mốc cùng máy (Kaggle): bỏ SAM 57,19, SAGE 56,90, TB 56,98.

| Bản | Thay đổi | Tham số | 10 lớp |
|---|---|---|---|
| `sxa0` | Thêm nhánh ngữ cảnh rộng (độ phân giải 1/16), cộng vào giữa mạng | 0,252M | 56,99 |
| `sxa10` | `sxa0` + loss A2 λ = 10 | 0,252M | 56,85 |
| `sxa3_10` | Nhánh ngữ cảnh 3 tầng + A2 λ = 10 | 0,310M | 57,48 |
| `sxa30` | `sxa0` + A2 λ = 30 | 0,252M | 56,96 |
| `srong10` | Student rộng gấp đôi (dim 64) + A2 λ = 10 | 0,537M | 56,40 |
| `sden10` | `sxa10` + loss giữ đèn không sáng hơn VIS (máy nhà; mốc bỏ SAM máy nhà 56,99) | 0,252M | 57,41 |
| `nosam_du` | Bỏ SAM, nhưng tốc độ học giảm hết chu kỳ (bản gốc chỉ giảm nửa) | 0,136M | 55,94 |

- **Cả bảy đều rớt.** Không bản nào tới 57,99.
- **Tăng tham số không tăng mIoU:** `srong10` to gấp 4 lần mà thấp nhất, kém `sxa3_10` một cách chắc chắn (−1,08).
- `sxa3_10` có person 50,0, cao nhất trong các ảnh ghép, nhưng lại mất đèn và biển báo (đèn bị nhầm thành biển 6,0 nghìn điểm ảnh). Lại đúng kiểu đánh đổi ở mục 4.17.
- `sxa10` nhầm đèn thành biển nhiều nhất (9,4 nghìn). Loss giữ đèn (`sden10`) kìm được đèn (tương phản đèn 14,2 → 8,3), nhưng **kìm luôn cả người** (tỉ lệ tương phản người 2,34 → 1,76). Student không tách được hai luật.
- `nosam_du` kém bỏ SAM một cách chắc chắn (−1,25), nên giữ lịch tốc độ học như bản gốc.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs/{sxa0,sxa10,sxa3_10,sxa30,srong10,nosam_du}_b0_s0/`, `runs_may_nha/sden10_b0_s0/`, `so_sanh_lop_du_dev_b0_sxa_s0.csv`, `so_sanh_lop_du_dev_b0_moi_s0.csv`, `so_sanh_lop_du_dev_b0_lint_s0.csv`; log `ket_qua/kham_pha/log_2211_2311/chay_sxa.out.txt`, `chay_sden.out.txt`. Code: `code/b3/sage_student/model_xa.py`, `train_student_nosam.py` (`--kien_truc xa`, `--xa_them`, `--dim`, `--lam_den`, `--lr_du`). Các bản `sxa3_10`, `sxa30`, `srong10` train thẳng trên Kaggle, checkpoint không tải về.

### 4.19. Kéo cường độ theo lớp (22/11)

**Ý tưởng:** sửa thẳng đánh đổi ở mục 4.17. Trong loss nội dung, mỗi điểm ảnh được kéo về VIS hay IR tùy theo **lớp của nó trong nhãn**: lớp mà VIS thắng ảnh ghép (vỉa hè, đèn, biển báo, xe tải, xe buýt) kéo về VIS (trọng số IR 0,1); person kéo về IR (0,7); lớp khác giữ 0,4 như bản gốc. Bảng trọng số đặt theo luật viết trước (`code/scene_v1/tinh_w_lop.py`). Nhãn chỉ dùng lúc train.

| Bản (seed 0, Kaggle) | 10 lớp |
|---|---|
| Kéo theo lớp (`lint`) | 56,92 |
| `lint` + A2 | 56,49 |
| `lint` + A2 + nhánh ngữ cảnh 3 tầng | 54,73 (kém chắc chắn mọi mốc) |
| Mốc: bỏ SAM / SAGE / TB | 57,19 / 56,90 / 56,98 |

**Cả ba rớt.** Để hiểu vì sao, mình đo **hệ số kéo về IR** của từng lớp trên ảnh ghép (0 = giống VIS, 1 = giống IR), trên val 151:

| Lớp | Đích của `lint` | `lint` | Bỏ SAM | SAGE |
|---|---|---|---|---|
| person | 0,7 | 0,41 | 0,48 | **1,06** |
| sidewalk | 0,1 | 0,38 | 0,42 | 0,96 |
| traffic_sign | 0,1 | 0,20 | 0,21 | 0,07 |
| traffic_light | 0,1 | **0,09** | 0,22 | 0,09 |
| truck | 0,1 | 0,23 | 0,25 | 0,28 |

- `lint` **không làm theo luật**: person còn đi ngược chiều (0,41 < 0,48). Chỉ đèn đạt.
- SAGE thì **có chọn lọc** rõ (person 1,06, biển báo 0,07), nhờ SAM. Nhưng mIoU của SAGE vẫn ngang bỏ SAM.
- Kết luận: Student 0,136M **không biết điểm ảnh nào thuộc lớp nào**, nên không áp được luật theo lớp, dù luật đó được dạy bằng nhãn.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs/lint*_b0_s0/`, `so_sanh_lop_du_dev_b0_lint_s0.csv`, `ket_qua/kham_pha/phan_tich/he_so_ir/he_so_ir_mat.csv`. Code: `train_student_nosam.py --w_lop`, `code/scene_v1/do_he_so_ir.py`, `w_lop.json`.

### 4.20. Chỉnh màu và thêm dữ liệu không nhãn (22/11)

**Chỉnh màu.** Ảnh bỏ SAM bạc màu ở những vùng IR làm sáng lên (person còn 77% độ bão hòa của VIS). Hai bản sửa chỉ đổi màu, giữ nguyên kênh độ sáng:

| Bản (3 seed, Kaggle) | 10 lớp | − bỏ SAM | − SAGE | − TB |
|---|---|---|---|---|
| Giữ màu (`mau`) | 57,25 | −0,25 | +0,00 | +0,14 |
| Giữ màu, không làm nhạt (`mau2`) | 57,43 | −0,07 | +0,18 | +0,32 |

Mọi cặp đều "chưa phân biệt". **Chỉnh màu không giúp mIoU.**

**Thêm 1.083 cặp MSRS không nhãn** khi train Student bỏ SAM (seed 0): 10 lớp 57,29 so với 57,19, chỉ +0,10. Ý định là giúp ảnh tối, nhưng nhóm ảnh tối không tăng (43,6 so với 44,1).

**Sửa một nhận định cũ.** Trước đây mình thấy SAGE và TB hơn bỏ SAM ở ảnh tối. Kiểm lại thì phần hơn đó **gần như toàn bộ do lớp bus** (vài khung đêm: bỏ SAM 24, SAGE 58, TB 51), mà bus không thuộc 10 lớp chấm. Các lớp khác ở nhóm tối chênh không quá ±5.

**Đổi quyết định 22/11:** vì đã dùng MSRS để train, MSRS không còn là dữ liệu "lạ". Bộ kiểm tổng quát chuyển sang **SemanticRT** (mục 5).

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs/{nosam_kp841_mau,nosam_kp841_mau2}_b0_s*/`, `runs/nosam_msrs_b0_s0/`, `so_sanh_lop_du_dev_b0_mau*.csv`, `so_sanh_lop_du_dev_b0_msrs_s0.csv`. Code: `code/b3/xuat_anh.py` (hậu tố `_mau`, `_mau2`), `train_student_nosam.py --them`.

### 4.21. Ghép theo vùng ngữ nghĩa thì đổi được mIoU: thử bằng oracle (22/11)

**Câu hỏi:** nếu ảnh ghép biết đúng lớp của từng điểm ảnh và trộn VIS/IR theo lớp đó, mIoU có tăng không? Đây là phép thử trần, **không phải phương pháp**.

- `orc_dung`: trộn theo bảng trọng số ở mục 4.19, lấy lớp từ **nhãn thật**, làm mờ biên 4 px.
- `orc_nguoc`: đảo vai (trọng số 0,8 − w). Dùng để tách phần lợi do "trộn đúng nguồn" khỏi phần lợi do ảnh để lộ hình dạng nhãn.

| B0 val 151, seed 0, máy nhà | 10 lớp |
|---|---|
| `orc_dung` (nhãn thật) | **64,83** |
| `orc_nguoc` (nhãn thật, đảo vai) | 59,03 |
| Nhãn **đoán** bởi `hai_nhanh` | 58,18 |
| Nhãn đoán bởi `hai_nhanh`, đảo vai | 55,56 |
| Nhãn đoán bởi B0 trên ảnh bỏ SAM | 57,56 |
| Bỏ SAM / VIS | 56,99 / 56,27 |

- Trộn đúng nguồn hơn đảo vai **+5,80** [+2,95; +8,13]. Ảnh ghép **có thể** đổi mIoU nhiều, nếu trộn theo vùng.
- Một phần lợi là do lộ hình nhãn: chính `orc_nguoc` cũng hơn bỏ SAM +2,04.
- Dùng nhãn **đoán** thay nhãn thật: nhãn từ `hai_nhanh` cho +1,19 so với bỏ SAM [+0,07; +2,26], "hơn chắc chắn". Nhãn đoán từ B0 bỏ SAM chỉ +0,57, chưa phân biệt (mô hình không tự sửa được lỗi của chính nó).
- Nhưng trên đủ 14 lớp, bản nhãn đoán `hai_nhanh` lại thấp hơn bỏ SAM (49,20 so với 50,33), do đèn và xe buýt.

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs_may_nha/orc_*_b0_s0/`, `runs_may_nha/so_sanh_lop_du_dev_b0_oracle{,_doan,_hn}.csv`. Code: `code/b3/xuat_anh.py` (`orc_dung`, `orc_nguoc`), `code/scene_v1/oracle_doan.py`, `chay_oracle.sh`.

### 4.22. `vhn`: ghép theo vùng do mô hình đoán, chia chéo dữ liệu (22–23/11)

**Cách làm.** Bản kiểm sạch của oracle nhãn đoán:
- Nền là ảnh bỏ SAM (`nosam_kp841`), kênh độ sáng Y_n.
- Train **hai** `hai_nhanh` B0, mỗi cái trên **một nửa** 841 ảnh train (chia theo khối 50 ảnh liền số xen kẽ, bỏ 2 ảnh đệm giữa các khối). Mỗi mô hình đoán cho nửa nó **chưa thấy**. Val 151 lấy trung bình hai mô hình. Như vậy không ảnh nào được đoán bởi mô hình đã học nó.
- Từ xác suất đoán (mờ 4 px): p_người, p_VIS (VIS = vỉa hè, đèn, xe tải).
- Ảnh ghép: Y = Y_n + p_người · (0,3·VIS + 0,7·IR − Y_n) + p_VIS · (0,9·VIS + 0,1·IR − Y_n). Màu giữ từ VIS.
- `vhn_nguoc`: đổi vai hai nhóm, làm đối chứng.

**Kết quả B0 val 151 (Kaggle):**

| | Seed 0 / 1 / 2 | **10 lớp (TB)** | 14 lớp (TB) |
|---|---|---|---|
| `vhn` | 57,77 / 57,96 / 58,33 | **58,02** | 50,19 |
| `vhn_nguoc` | 55,47 / – / – | 55,47 | 48,73 |
| Bỏ SAM | | 57,50 | 49,91 |
| SAGE | | 57,25 | 47,85 |
| TB | | 57,11 | 47,72 |

| Cặp (10 lớp, 3 seed) | Hiệu | Khoảng tin cậy 95% | Kết luận |
|---|---|---|---|
| `vhn` − SAGE | +0,77 | [−0,01; +1,64] | **hơn nhất quán** |
| `vhn` − TB | +0,91 | [+0,33; +1,55] | **hơn chắc chắn** |
| `vhn` − bỏ SAM | +0,51 | [+0,26; +0,89] | hơn chắc chắn |

- **Đây là cách đầu tiên đạt đích ở mục 2.4** ("hơn nhất quán cả SAGE và TB") trên val 151. Trên 14 lớp, mọi cặp vẫn "chưa phân biệt".
- Lợi chủ yếu ở person (trung bình 3 seed: 47,7 so với bỏ SAM 44,6, khoảng 60% phần hơn), rồi vỉa hè và biển báo.
- `vhn_nguoc` thấp hơn hẳn (55,47), nên phần hơn đến từ việc trộn đúng nguồn, không phải do đổi ảnh bất kỳ.
- **Ghi rõ:** luật ngưỡng nội bộ đặt trước cho bước này là `vhn` − bỏ SAM ≥ +1,0. Theo luật đó, `vhn` **chưa đạt** (+0,51). Nó chỉ đạt theo đích chính.

**Vì sao chưa phải phương pháp nộp:**
- Lúc chạy cần hai mạng `hai_nhanh` (mỗi mạng khoảng 7,2M tham số) để đoán vùng. Không còn là "Student nhỏ".
- Luật trộn (nhóm lớp VIS, hệ số 0,3/0,7, 0,9/0,1) được chọn **sau khi** đã xem số trên val 151. Phải kiểm trên Dev 228 (đang chạy, mục 5).

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs/{vhn,vhn_nguoc}_b0_s*/`, `so_sanh_lop_du_dev_b0_vhn3seed.csv`, `so_sanh_lop_du_dev_b0_vhn.csv`. Code: `code/b3/vung_hai_nhanh.py` (bước `chia`, `dung`, `luu_p`); cách chia hai nửa ở `code/scene_v1/_vhn/` (`nua.csv`, `man_A/`, `man_B/`). Checkpoint hai `hai_nhanh` nửa (khoảng 28 MB mỗi cái) không kèm repo, tạo lại bằng `vung_hai_nhanh.py chia` rồi `train_seg.py --bien_the hai_nhanh` trên từng nửa.

### 4.23. Chưng cất `vhn` vào một Student chạy độc lập (23/11)

**Mục tiêu:** có ảnh ghép giống `vhn` mà lúc chạy không cần mạng phân đoạn. `vhn` làm "thầy" (giống vai Teacher của SAGE, nhưng không có SAM), Student học bắt chước ảnh của thầy.

Mọi bản dưới đây chấm B0 val 151, seed 0, **máy nhà**. Mốc cùng máy: bỏ SAM 56,99.

| Bản | Cách học | 10 lớp | − bỏ SAM [KTC 95%] |
|---|---|---|---|
| `cc_vhn` | Loss gốc + thêm MSE với ảnh thầy | 56,42 | −0,56 [−1,18; +0,13] |
| `cc_vhn_thuan` | Chỉ học theo ảnh thầy (bỏ loss gốc) | 56,72 | −0,27 [−1,11; +0,42] |
| `du2_mat_ft` | Khởi tạo từ bỏ SAM, thêm ResNet18 ImageNet đóng băng làm "mắt", nhấn mạnh chỗ thầy sửa | 57,06 | +0,08 [−0,43; +0,49] |
| **S-vùng** (`svung`) | Student có cấu trúc giống công thức của thầy (bên dưới) | **57,12** | +0,14 [−0,14; +0,41] |

**Chẩn đoán.** Đo Student chép được bao nhiêu phần chỉnh của thầy, theo lớp: tỉ lệ chép = Σ(D_s·D_t) / Σ(D_t²), với D_t = ảnh thầy − ảnh bỏ SAM, D_s = ảnh Student − ảnh bỏ SAM (1 = chép đủ, 0 = không chép).
- Thầy chỉnh mạnh nhất ở person (độ lệch RMS 14,9 mức xám trên val 151), các lớp khác dưới 4.
- Student train từ đầu (`cc_vhn`, `cc_vhn_thuan`) gần như không chép được person. Lý do chính: hai lần train **y hệt** của chính Student bỏ SAM đã lệch nhau nhiều hơn phần thầy chỉnh, nên tín hiệu của thầy bị nhiễu train lại nhấn chìm. Vì vậy các bản sau đều **khởi tạo từ Student bỏ SAM**.
- `du2_mat_ft`: chép person 0,54 (val 151) và 0,67 (ảnh train). IoU person +2,7, vỉa hè +2,5, đúng chỗ thầy sửa. Nhưng các lớp nền bị **trôi** (lệch không do thầy, RMS: sky 9,6, vegetation 6,3, building 5,4), làm biển báo −2,9, building −0,9. Cộng lại gần hòa.

**S-vùng.** Thầy có dạng công thức ở mục 4.22. Vì vậy Student được dựng **đúng theo công thức đó**:
- Student bỏ SAM **đóng băng** làm nền.
- ResNet18 ImageNet **đóng băng** (tầng 1–3, nhận Y và IR) làm đặc trưng.
- Một **đầu vùng** nhỏ (0,095M tham số, phần duy nhất được học) đoán p_người và p_VIS ở 1/4 độ phân giải, học theo bản đồ xác suất của thầy (BCE). Tổng 3,0M tham số.
- Ghép đúng công thức của thầy. Chỗ p = 0 thì ảnh **đúng bằng** ảnh bỏ SAM, nên **không thể trôi**.

Kết quả S-vùng:
- Train 6,5 phút. Lớp nền hết trôi (lệch không do thầy chỉ 0,2–1,0 mức xám ở các lớp nền, so với 3–10 của `du2_mat_ft`). Chép person 0,75 trên val 151 và 0,87 trên ảnh train (`du2_mat_ft` 0,54 và 0,67).
- IoU person 47,2 (+2,1). Building +0,5, biển báo +0,4, vegetation 0. Nhưng vỉa hè −0,5: đầu vùng gần như không học được bản đồ "lớp VIS" (IoU mềm trên val 151 chỉ 0,07, so với 0,30 của bản đồ người).
- 14 lớp: 49,92, thấp hơn bỏ SAM 0,41 (motorcycle 1 ảnh, bus, đèn).

**Đọc:** S-vùng sửa đúng lỗi trôi, nhưng phần lợi nhỏ. Lý do: chính thầy `vhn` chỉ hơn bỏ SAM khoảng +0,5 (3 seed). **Student tốt cỡ nào cũng khó vượt thầy.** Muốn hơn rõ thì cần thầy mạnh hơn, không chỉ chưng cất tốt hơn.

Chưa đo: thời gian chạy một ảnh của S-vùng (có thêm ResNet18).

Số liệu gốc: `ket_qua/kham_pha/sang_loc_b0/runs_may_nha/{cc_vhn,cc_vhn_thuan,du2_mat_ft,svung}_b0_s0/`, `runs_may_nha/so_sanh_lop_du_dev_b0_{cc,thuan,du2,svung}.csv`; log và bảng tỉ lệ chép ở `ket_qua/kham_pha/log_2211_2311/` (`chay_du.out.txt`, `chay_du2.out.txt`, `svung.log.txt`, `chep_svung.txt`). Code: `code/scene_v1/chep_delta.py`; `code/b3/sage_student/model_mat.py`, `model_vung.py`; `code/e17/train_student_nosam.py` (`--ref`, `--chi_ref`, `--w_khac`, `--khoi_tao`, `--dong_bang_goc`, `--kien_truc mat`), `code/e17/train_student_vung.py` (cần `code/scene_v1/_vhn/p_thay.npz`, bản đồ của thầy đã kèm). Checkpoint: `code/b3/sage_student/{cc_vhn,cc_vhn_thuan,cc_vhn_thuan_w10,du_ft,du2_mat_ft,svung}.pt`.

### 4.24. Tầng chọn: SegFormer-B3 trên Dev 228, 3 seed (21–22/11)

Thiết lập: SegFormer-B3 train trên 992 ảnh, chấm Dev 228 bằng checkpoint cuối, Kaggle T4, 3 seed. Student bỏ SAM được **train lại trên 992 ảnh** (`nosam_992`), cũng trên Kaggle. Dev 228 có 12 lớp đủ ảnh (≥ 20 ảnh; motorcycle chỉ 7 ảnh, bicycle 0 ảnh).

| Ảnh đưa vào | Seed 0 / 1 / 2 (12 lớp) | **12 lớp (TB)** | 13 lớp có mặt (TB) |
|---|---|---|---|
| Bỏ SAM (`nosam_992`) | 64,00 / 65,48 / 64,49 | **64,66** | **60,44** |
| SAGE | 63,99 / 64,63 / 64,19 | 64,27 | 59,41 |
| TB | 64,15 / 64,39 / 63,78 | 64,11 | 60,15 |
| VIS | 63,88 / 64,10 / 63,32 | 63,77 | 59,30 |

| Cặp (12 lớp, 3 seed) | Hiệu | Khoảng tin cậy 95% | Kết luận |
|---|---|---|---|
| Bỏ SAM − SAGE | +0,38 | [−0,98; +1,60] | chưa phân biệt |
| Bỏ SAM − TB | +0,55 | [−0,95; +1,89] | chưa phân biệt |
| SAGE − TB | +0,17 | [−1,06; +1,30] | chưa phân biệt |
| SAGE − VIS | +0,51 | [−1,43; +2,73] | chưa phân biệt |

- Trên tầng chọn, bỏ SAM **cũng không thua SAGE**, giống tầng sàng lọc. Mọi cặp đều chưa phân biệt.
- Kiểm tái lập: seed 0 khớp số đã có của nhóm (13 lớp: SAGE 59,16 so với 59,05 của bạn Quân; VIS 59,46 so với 59,52 và TB 60,29 so với 60,04 của bạn Lộc bé).
- IoU trung bình 3 seed, bỏ SAM / SAGE / TB / VIS: person 69,4 / 70,0 / 70,2 / 51,6; truck 27,5 / 30,5 / 24,2 / **36,9**; traffic_light 29,4 / 25,0 / 25,3 / **31,1**. Cùng kiểu đánh đổi theo lớp ở mục 4.17.
- Đích cho phương pháp trên tầng này: hơn nhất quán cả SAGE (64,27) và TB (64,11) trên 12 lớp.

Số liệu gốc: `ket_qua/dev228_b3/runs/` (12 lần chạy: ma trận nhầm lẫn `dev_nham_last.npz`, IoU từng lớp, cấu hình, log), `so_sanh_lop_du_dev_b3_vis4.csv` (đủ 4 loại ảnh). Tính lại: `cd code/b3 && SAGE_RUNS=../../ket_qua/dev228_b3/runs python so_sanh_lop_du.py --bien_the nosam_992 sage tb vis --model b3 --tap dev --hau _15lop --seeds 0 1 2 --min_anh 20 1`.

### 4.25. Nhận định chung đến 23/11 (trên B0; xem thêm 4.33)

- Trên B0 val 151, các ảnh ghép **không dùng nhãn** (khoảng 30 bản) cho 10 lớp trong khoảng 54,7–58,0, phần lớn 56,3–57,8. SAGE, TB, bỏ SAM chênh nhau dưới mức dao động do seed (riêng bỏ SAM đã dao động 57,19–57,88 giữa 3 seed).
- Câu đúng là: **các cách ghép toàn cục** (SAGE, TB, VIS, các biến thể Student) cho mIoU gần như nhau; **ghép theo vùng ngữ nghĩa** thì đổi được mIoU, cả lên (`vhn` +0,5 đến +0,9, oracle nhãn thật +7,8) và xuống (`vhn_nguoc` −1,7, seed 0).
- "SegFormer train lại tự bù phần khác biệt của ảnh ghép toàn cục" mới là **giả thuyết**. Bằng chứng gián tiếp: chỉnh màu, tăng tương phản, đổi kiến trúc đều làm ảnh đổi rõ mà mIoU không đổi.
- Các số này chủ yếu trên B0 và val 151. Chưa đối chiếu với các phương pháp khác trong bảng của bài SAGE.

### 4.26. Ghép theo vùng trên SegFormer-B3: hơn SAGE, chưa hơn TB (23–24/11)

Mọi số trong mục này: SegFormer-B3 train 992 ảnh, chấm Dev 228, checkpoint cuối, mIoU 13 lớp có mặt. Mốc cùng seed lấy từ mục 4.24.

**`vhn` (mục 4.22) trên Dev 228, 2 seed.** Dùng lại hai mạng `hai_nhanh` nửa dữ liệu. Dev 228 không trùng hai nửa train.

| | Seed 0 | Seed 1 | TB 2 seed |
|---|---|---|---|
| `vhn` | 59,92 | 61,71 | 60,82 |
| SAGE | 59,16 | 59,78 | 59,47 |
| TB | 60,29 | 60,08 | 60,19 |
| Bỏ SAM (`nosam_992`) | 60,21 | 61,12 | 60,67 |

- `vhn` − SAGE +1,34 (khoảng tin cậy theo ảnh [+0,03; +3,01]). `vhn` − TB +0,63, nhưng seed 0 thua (−0,37). `vhn` − bỏ SAM +0,15.
- Luật đặt trước yêu cầu **mỗi seed** hơn cả SAGE và TB → **không đạt**, không chạy seed 2. Trên B3, lợi của `vhn` gần bằng lợi của nền bỏ SAM. Phần "vùng" không thêm gì rõ.

**Oracle trên B3 (seed 0).** Ghép theo vùng bằng nhãn thật (mục 4.21) là trần của mọi cách ghép theo vùng: `orc_dung` 62,17, `orc_nguoc` 60,55, TB 60,29. Oracle chỉ hơn TB **+1,88**. Luật đặt trước: dưới +2 thì chuyển hướng. Trên B0, oracle hơn tới +7,8, nên phần lớn lợi của oracle trên B0 là do B0 yếu.

**Chỗ trống của B3.** Nếu chọn bản tốt nhất cho từng lớp trong {SAGE, TB, bỏ SAM, VIS} (3 seed, hậu nghiệm), mIoU lên 62,27 (+2,86 so SAGE, +2,1 so TB). Phần lợi nằm ở **lớp nhỏ, hiếm**: đèn +6,1 (VIS 31,1), truck +6,4 (VIS 36,9), biển +4,0 (TB 71,5), bus +4,0 (VIS 76,1), motorcycle (7 ảnh, nhiễu). **Không ở person** (+0,2): trên B3, mọi ảnh ghép có IR đều cho person khoảng 69–71. Vì vậy từ 24/11 thôi nhắm vào person.

**`vnc` = VIS làm nền + cổng nhiệt (24/11).** Y = Y_vis + g·(Y_nosam − Y_vis). Cổng g mở ở vật nhỏ ấm (top-hat trắng trên IR) và ở ảnh tối. Không nhãn, không mạng phụ, tham số chọn trên ảnh Train. Rào: MS-SSIM 99,4%, Q_ABF 114,5%, SCD 83,6% SAGE.

| | Seed 0 | Seed 1 | TB |
|---|---|---|---|
| `vnc` | 61,22 | 60,76 | 60,99 |

- Luật: mỗi seed hơn mốc cao nhất cùng seed (s0 > 60,29, s1 > 61,12) → seed 1 **rớt**.
- So TB +0,81, so SAGE +1,52, so bỏ SAM +0,32: mọi khoảng tin cậy chứa 0.
- Lợi dồn vào **một đoạn video** (ảnh 01383–01419, 37 ảnh sáng, nhiều truck và bus). Bỏ đoạn này: `vnc` 59,40 < bỏ SAM 59,65 < TB 60,59. Đèn còn thấp hơn bỏ SAM (−2,9), trái giả thuyết.

**`vhn3`, `vhn4`, `vhn5` (thầy mạnh hơn).** Thay `hai_nhanh` B0 bằng B3: mIoU mỗi nửa lên 51,03 / 55,96 (B0: 44,9 / 45,3). Nhưng trên Dev 228, bản đồ vùng "lớp VIS" chỉ bắt được 23% (recall). `tbsua_vhn3` (bản đồ `vhn3` đặt trên nền `tbsua`, mục 4.27) seed 0 được 60,16, thấp hơn chính nền `tbsua` seed 0 (61,49) → rớt. `vhn4` (bản đồ chỉ vỉa hè + đèn) và `vhn5` (cổng không nhãn) rớt luật đặt trên Train trước khi lên GPU. **Dừng họ `vhn`.**

Số liệu: `ket_qua/dev228_b3/runs/` (`vhn_*`, `vnc_*`, `orc_*`, `tbsua_vhn3_*`); phân tích trong `ket_qua/dev228_b3/phan_tich/` (`soi_vhn3b.txt`, `ban_do_vhn4.txt`, `chon_cong_vhn5.txt`).

### 4.27. Đổi nền sang TB: `tbsua_992` và `a04` (24–26/11)

Rà lại khoảng 48 bản B0 theo 8 lớp phổ biến, mình thấy họ **nền TB + phần sửa học được** (`tbsua`, mục 4.3) từng bị bỏ oan ngày 19/11. Khi đó họ này được chấm bằng 14 lớp, mà điểm 14 lớp bị motorcycle (1 ảnh) kéo lệch. Trên SegFormer-B3, TB lại là mốc khó vượt nhất. Vì vậy mình chuyển sang nền TB.

**`tbsua_992`:** Student SAGE (giữ kiến trúc), không SAM, không nhãn, học phần sửa trên nền TB. Train 992 ảnh, cùng tham số `tbsua_w150` (mục 4.4). Rào Dev 228: MS-SSIM 101,5% / Q_ABF 115,5% / SCD 89,7% SAGE → qua.

**`a04`:** Y = TB + α·(Y_tbsua − TB), với **α = 0,4**. α được chọn trên **124 ảnh Train** theo luật đặt trước: α nhỏ nhất qua rào MS-SSIM ≥ 99% và Q_ABF ≥ 98% SAGE (SCD 82,5%, dưới mức cảnh báo 85%). Chỉ một α được đưa lên B3, không dò α theo Dev. Rào Dev 228: MS-SSIM 103,8% / Q_ABF 105,9% / SCD 87,0%. Không thêm tham số học, chi phí chạy bằng Student SAGE.

| Ảnh (B3, Dev 228) | 13 lớp, seed 0 / 1 / 2 | **13 lớp TB** | 8 lớp phổ biến TB | 5 lớp khó TB |
|---|---|---|---|---|
| **`a04`** | 60,89 / 60,49 / 60,93 | **60,77** | 77,48 | 34,04 |
| `tbsua_992` | 61,49 / 60,41 / 59,67 | 60,52 | 77,22 | 33,81 |
| Bỏ SAM (`nosam_992`) | 60,21 / 61,12 / 59,98 | 60,44 | 76,58 | 34,60 |
| TB | 60,29 / 60,08 / 60,07 | 60,15 | 77,49 | 32,40 |
| SAGE | 59,16 / 59,78 / 59,29 | 59,41 | 76,65 | 31,83 |
| VIS | 59,46 / 59,72 / 58,73 | 59,30 | 74,00 | 35,79 |

8 lớp phổ biến: road, building, vegetation, sky, person, car, pole, traffic_sign. 5 lớp khó: sidewalk, traffic_light, truck, bus, motorcycle.

| Cặp (3 seed, so theo cặp seed) | Hiệu | Khoảng tin cậy 95% (khối 10 ảnh) | Đọc |
|---|---|---|---|
| `a04` − SAGE, 13 lớp | +1,36 | [+0,06; +3,09] | hơn nhất quán |
| `a04` − SAGE, 8 lớp | +0,83 | [+0,35; +1,14] | hơn nhất quán |
| `a04` − TB, 13 lớp | +0,63 (+0,60 / +0,41 / +0,86) | [−0,04; +1,26] | **không kém TB** (biên −0,5), chưa hơn chắc |
| `a04` − TB, 8 lớp | −0,01 | | ngang |
| `tbsua_992` − SAGE, 13 lớp | +1,11 | chứa 0 theo khối cảnh | hơn ở cả 3 seed |
| `tbsua_992` − TB, 13 lớp | +0,38 (seed 2: −0,40) | chứa 0 | chưa phân biệt |

Quá trình và luật (ghi đủ để minh bạch):
- `tbsua_992` (seed 0, 1): luật là 13 lớp ≥ 60,90 **và** 8 lớp ≥ 77,62 (mốc TB). 13 lớp đạt sát (60,95), 8 lớp rớt (76,99), chủ yếu do biển báo (2 ảnh liền 00174–00175 và seed 1).
- `a04` (seed 0, 1): luật là 8 lớp ≥ 77,62 và 5 lớp không thua TB. 5 lớp đạt (+1,81), **8 lớp rớt** (77,31). Theo luật thì không chạy seed 2.
- Mân quyết chạy thêm seed 2 (ghi rõ là ngoài luật). **Trước khi mở số seed 2**, Mân chốt biên "không kém TB": cận dưới khoảng tin cậy 13 lớp của `a04` − TB phải > −0,5. Seed 2 cho `a04` 60,93, hơn cả TB (60,07) và SAGE (59,29). Theo luật đọc đã ghi trước → **khóa `a04`**.
- Lợi so TB phụ thuộc đoạn 01383–01419: bỏ đoạn này, `a04` − TB (2 seed) từ +0,51 còn +0,01. Theo độ sáng: ảnh sáng +1,32, vừa +0,14, tối +0,13.
- Ở 8 lớp, phần thua TB chủ yếu là biển báo (−1,7) và building (−0,45, chỉ ở seed 0).

Số liệu: `ket_qua/dev228_b3/runs/` (`tbsua_992_*`, `tbsua_a04_*`); `phan_tich/cham_tbsua3.txt`, `cham_a04.txt`, `cham_a04_3s.txt`, `phan_tich_ho_tb.txt`, `chay_tbsua992.out.txt` (rào). Tính lại: `cd code && python scene_v1/cham_a04_3s.py`.

### 4.28. Kiểm chéo theo cảnh trên Train (`kc1`, 25/11)

Dev 228 chỉ có khoảng 10 đoạn video, nên lợi dễ dồn vào một đoạn. Để có thêm cảnh, mình chia lại 992 ảnh Train: **phần kiểm 411 ảnh** (5 cụm id liền: 1–39, 136–156, 519–586, 649–700, 1059–1300), **train 782 ảnh**, bỏ 27 ảnh đệm. Trùng cảnh được kiểm bằng so khớp đặc trưng ORB (có 1 báo động giả, nhìn bằng mắt thì khác cảnh). Chạy SegFormer-B3 trên TB và `a04`, 3 seed. Lưu ý: `tbsua_992` đã học (không nhãn) trên cả 992 ảnh này, nên phần kiểm không hoàn toàn "chưa thấy" với phần ghép.

| `a04` − TB (411 ảnh) | Seed 0 / 1 / 2 | TB | Khoảng tin cậy |
|---|---|---|---|
| 13 lớp | +0,60 / +0,51 / +0,31 | **+0,47** (59,95 → 60,43) | khối [−0,18; +0,92]; cụm [−0,98; +0,77] |
| 8 lớp | −0,34 / −0,05 / −0,14 | −0,18 | [−0,45; +0,01] |
| 5 lớp | | +1,51 | |

- Theo luật đặt trước: dương cả 3 seed nhưng khoảng tin cậy chứa 0 → **"tín hiệu, chưa kết luận"**.
- Lợi dồn vào **2 ảnh bus liền nhau** (01271–01272). Bỏ 2 ảnh này thì 13 lớp còn +0,16. 4/5 cụm âm, chỉ cụm 1059–1300 dương.
- person −0,89 trên `kc1` (cả 3 seed), nhưng +0,15 trên Dev 228 → không lặp lại. Biển báo là lớp duy nhất thua TB trên cả hai tập.
- Gộp `kc1` (3 seed) và Dev 228 (2 seed), phân tích **sau khi thấy số**: 13 lớp +0,49, khoảng tin cậy theo khối [+0,03; +0,86], theo cụm [−0,40; +0,95]. Bỏ cả hai đoạn nhiều bus/truck thì còn +0,06.

Số liệu: `ket_qua/kiem_cheo_kc1/` (6 lần chạy, ma trận nhầm lẫn 411 ảnh). Cách chia: `code/scene_v1/chia_kiem_cheo.py`, `kiem_cheo_p1.csv`. Phân tích: `phan_tich/phan_tich_kc1.txt`, `gop_kc1_dev.txt`.

### 4.29. Các hướng rớt hoặc bị loại (24–26/11)

Mọi hướng đều có luật đặt trước. Phần lớn dừng **trước khi tốn GPU**, ngay ở bước kiểm trên Train.

| Hướng | Ý | Kết quả | File trong `phan_tich/` |
|---|---|---|---|
| Ảnh 3 kênh (R = IR, G = TB, B = Y_vis) | Đưa đủ thông tin cho SegFormer | B0 seed 0: 8 lớp −0,67 so TB (luật ≥ +1,0) | `chay_tri3_b0.out.txt` |
| Chuẩn hóa dải IR | IR có dải hẹp làm mờ người? | Người trong ảnh IR dải hẹp không mờ hơn; chỉ 1/3 ngưỡng đạt | (không lên GPU) |
| Ảnh tối dùng TB | TB thắng ở ảnh tối | Chênh lệch do motorcycle 1 ảnh; không lặp trên val 151 | |
| `m11b_992`, `m11c_l45_992` (`tbsua` + CE nhãn) | Học thêm từ nhãn | Ảnh đổi quá ít so `tbsua` (m11c: 2/8 lớp đổi ≥ 2 lần, luật 5/8) | `soi_m11b.txt`, `soi_m11c.txt` |
| Liều theo vùng phẳng (`lv`) | Giảm phần sửa ở vùng phẳng | Rớt rào Q_ABF trên phần kiểm (97,0 < 98) | `lieu_vung3.txt` |
| Nền 60% IR | Tăng tỉ lệ IR | Số B0 cũ: 8 lớp −0,54 so TB | `nen_ir.txt` |
| ResNet làm loss thay SAM | Giữ đặc trưng ImageNet | Tương quan với 8 lớp B3 chỉ 0,37–0,49 (luật ≥ 0,7) | `kiem_loss_resnet.txt` |
| Sửa ảnh theo giám khảo (CE) | Như M4 | Phần lớn là "lừa giám khảo": CE giảm −31,6% trên giám khảo dùng để tối ưu, chỉ −5,8% trên giám khảo khác | `kiem_m4.txt`, `nhieu_giam_khao.txt`, `tran_nhan.txt` |
| `a04-ct` | Giữ độ sáng, chỉ thêm chi tiết | Gần TB hơn `a04` chỉ 9,7% (luật ≥ 20%) | `a04_ct.txt` |

### 4.30. `a04_cvis`: thêm cổng VIS lên `a04` (26/11)

Ý tưởng: VIS thuần mạnh nhất ở truck, đèn và bus. Thêm một cổng **không nhãn** để trả VIS về ở những vùng đó: Y = Y_a04 + x·(Y_vis − Y_a04). Cổng hẹp, tham số chọn trên Train ngày 24/11 (s = 8, a = 0,2), không dò lại. Rào Dev 228: MS-SSIM 102,1% / Q_ABF 107,9% / SCD 85,6%. Luật đặt trước: giữ nếu TB 2 seed > 60,69 (mức của `a04` seed 0, 1) **và** cả 2 seed hơn TB cùng seed.

| B3, Dev 228 | Seed 0 / 1 | 13 lớp TB | 8 lớp | 5 lớp |
|---|---|---|---|---|
| `a04_cvis` | 59,55 / 59,28 | **59,41** | 76,83 | 31,56 |
| `a04` | 60,89 / 60,49 | 60,69 | 77,31 | 34,10 |
| TB | 60,29 / 60,08 | 60,19 | 77,62 | 32,29 |

- **Rớt** cả hai vế: cả 2 seed thua TB (−0,74 / −0,80).
- So `a04`: 13 lớp −1,28 [−2,40; −0,36]; 8 lớp −0,48 [−0,86; +0,19]; **5 lớp −2,55 [−5,80; −0,38]**. Cổng làm hại lớp hiếm rõ hơn lớp phổ biến.
- Kiểm nhanh nguyên nhân (chỉ có ma trận nhầm lẫn từng ảnh, không có dự đoán từng điểm ảnh):
  - Cổng mở **đúng chỗ**: phần điểm ảnh nằm trong vùng mở là bus 65%, truck 58%, biển 54%, nhưng person chỉ 4%, sky 2%.
  - Nhưng chính các lớp đó tụt: biển −2,1 / −4,9; bus −6,3 / −0,3; truck −8,1 / +1,9.
  - 76 ảnh cổng mở ít nhất: −0,08 (hòa). Bỏ 150 ảnh cổng mở nhiều nhất: −0,13. Tương quan theo ảnh yếu (r = −0,13, p = 0,06).
  - Kết luận: **nghiêng về lỗi do cổng**, chưa chắc. Giả thuyết (chưa kiểm): trộn nửa VIS tạo ra vùng không giống VIS mà cũng không giống ảnh ghép.

**VIS thuần so với `a04`** (IoU, TB 3 seed):
- VIS hơn ở truck (36,9 so 26,1, +10,8), đèn (31,1 so 26,2, +4,9), bus (76,1 so 74,1, +2,0), road và building (+0,4).
- VIS thua ở person (51,6 so 70,7, **−19,1**), motorcycle (−6,5), car, vỉa hè, vegetation, pole và biển.
- Trong 6 bản đã có 3 seed, VIS đứng đầu 5/13 lớp, `a04` 3/13. Nhưng `a04` cao nhất ở 13 lớp vì **đều**, lớp nào cũng gần đầu.

Số liệu: `ket_qua/dev228_b3/runs/a04_cvis_*`; `phan_tich/cham_a04cvis.txt`, `kiem_cong_cvis.txt`, `rao_a04_cvis.txt`. Code: `code/scene_v1/dung_a04_cvis.py`, `dung_2_nhanh.py`, `kiem_cong_cvis.py`.

### 4.31. Khóa phương án và giao thức Test (26/11)

- **Phương án khóa:** `a04`. Phần ghép `tbsua_992` **đóng băng** (train 992 ảnh, không nhãn), không train lại trên 1.220 ảnh.
- **Giao thức Test** (theo `QUY_TAC_CHAM.md`): SegFormer-B3 train đủ 1.220 ảnh train chính thức, 100 epoch, batch 8, chấm checkpoint cuối (không chọn checkpoint). mIoU 13 lớp trên **273 ảnh sạch** (chính) và trên 280 ảnh. So theo cặp seed.
- **Seed:** `a04` chạy seed 0–4. TB và SAGE đã có seed 0–2 từ mục 2.1, chạy thêm seed 3, 4. Không chạy bỏ SAM trên Test.
- **Ghi câu kết luận trước khi mở số**, cho 3 trường hợp: hơn cả TB và SAGE / chỉ hơn SAGE / thua TB. Biên "không kém TB": cận dưới khoảng tin cậy 13 lớp > −0,5.
- Chạy trên Kaggle (T4 × 2), mỗi notebook 4 lượt, khoảng 5,5–6 giờ. **Chưa có số.**

### 4.32. Ghi chú về cách làm (24–26/11)

- Mỗi ý tưởng có **luật đặt trước**, ghi trong notebook hoặc script trước khi chạy. Lần nào chạy thêm ngoài luật (seed 2 của `a04`) đều ghi rõ.
- Mọi phương án lên B3 trong mục 4.26–4.30 đều dùng rào chỉ số ghép 99% (MS-SSIM) / 98% (Q_ABF) so SAGE. `QUY_TAC_CHAM.md` có một rào tạm 95/90 (ghi ngày 23/11), nhưng không dùng để cứu phương án nào.
- Các số "gộp", "bỏ đoạn", "chọn tốt nhất từng lớp" là phân tích **sau khi thấy số**. Chúng chỉ để hiểu, không dùng để chọn phương án.

### 4.33. Nhận định chung đến 26/11

- Trên SegFormer-B3, các ảnh ghép **toàn cục** (SAGE, TB, bỏ SAM, `tbsua`, `a04`) cách nhau khoảng 1,4 điểm ở 13 lớp. `a04` cao nhất, nhưng phần hơn TB còn nằm trong mức nhiễu của Dev 228.
- **Đánh đổi theo lớp vẫn là nút thắt.** IR giúp person rất nhiều (+19 so VIS). Nhưng mọi ảnh ghép đều làm mất một phần tín hiệu màu ở truck, đèn và bus. Cả hai lần thử trả VIS về theo vùng (`vnc`, `a04_cvis`) đều không giữ được phần lợi này. Câu đúng là: **trong các bản đã thử, chưa có ảnh ghép nào giữ được cùng lúc lợi của IR và lợi của VIS.**
- Lợi ở lớp hiếm dồn vào vài đoạn video (01383–01419 trên Dev, 01271–01272 trên `kc1`). Mọi kết luận về lớp hiếm cần Test và dataset khác.
- Hướng có thể khảo sát sau: ghép ở bước phân đoạn (hai nhánh, hoặc gộp dự đoán của mạng VIS và mạng ảnh ghép). Nhưng đó là một phương pháp khác, chi phí gấp đôi, và phải so với TB + VIS cùng cách gộp.

---

## 5. Bước tiếp theo (26/11)

1. **Test (đang chạy trên Kaggle):** `a04` seed 0–4 và mốc TB, SAGE seed 3, 4. Trước khi mở số, ghi vào `QUY_TAC_CHAM.md` câu kết luận cho 3 trường hợp (hơn cả TB và SAGE / chỉ hơn SAGE / thua TB) và biên "không kém TB" (−0,5).
2. **Chấm Test:** so theo cặp seed trên 273 ảnh sạch (chính) và 280 ảnh, bằng `so_sanh_cap.py`. Báo cả 13 lớp, 8 lớp, 5 lớp và IoU từng lớp. Kiểm lại xem truck, đèn, bus có còn thua VIS không (mục 4.30).
3. **Kiểm tổng quát trên SemanticRT** (đã chia theo địa điểm, mục 5 bản 23/11): rào ảnh `a04` trên 300 ảnh Train SemanticRT đã đạt (MS-SSIM 106,7% / Q_ABF 116,3% SAGE). Chưa chạy SegFormer, đang chờ quota.
4. Nếu còn thời gian (số phụ, ghi rõ là hướng khác): gộp dự đoán mạng VIS và mạng `a04` theo một quy tắc chốt trước, so với TB + VIS cùng quy tắc.

## 6. Cần nhóm chốt

| Việc | Đề xuất |
|---|---|
| Số chính để báo cáo | mIoU 13 lớp có mặt trên Test 273 ảnh sạch (theo `QUY_TAC_CHAM.md`); 8 lớp và 5 lớp là số phụ |
| Biên "không kém TB" | Cận dưới khoảng tin cậy 13 lớp > −0,5 (Mân đã chốt 26/11; nhóm cần xác nhận) |
| Rào chỉ số ghép | 99/98 (đã dùng cho mọi phương án lên B3). Rào tạm 95/90 trong `QUY_TAC_CHAM.md` nên bỏ hoặc chốt |
| Ghi luật 8 lớp của `a04` | `a04` rớt luật 8 lớp đặt trước (77,31 < 77,62 ở seed 0, 1); seed 2 chạy ngoài luật. Phải ghi trong báo cáo |
| Dataset kiểm tổng quát | SemanticRT (chia theo địa điểm) |

## 7. Giới hạn

- **Dev 228 đã được nhìn rất nhiều lần** (khoảng 10 phương án, phân tích theo lớp, theo đoạn, theo độ sáng). Số Dev chỉ dùng để chọn. Kết quả báo cáo phải lấy từ Test (chạy một lần cho bản đã khóa) và dataset khác.
- Dev 228 chỉ có khoảng 10 đoạn video. Lợi ở lớp hiếm dồn vào 1–2 đoạn (mục 4.27, 4.28). Khoảng tin cậy theo khối 10 ảnh có thể hẹp hơn thực tế. Khoảng tin cậy theo cụm (rộng hơn) đều chứa 0.
- `a04` hơn TB chưa chắc (khoảng tin cậy 13 lớp chứa 0, 8 lớp ngang). Hơn SAGE thì nhất quán trên Dev.
- α = 0,4 và tham số cổng được chọn trên ảnh Train, không trên Dev. Nhưng **hướng** đổi nền sang TB và ý cổng VIS xuất phát từ phân tích Dev 228.
- `tbsua_992` học (không nhãn) trên 992 ảnh Train, gồm cả phần kiểm của `kc1`.
- Checkpoint SAGE được train trên đủ 1.220 ảnh train FMB (đã thấy Dev 228). Nếu có lợi thế thì đang nghiêng về SAGE.
- Một số lần chạy trên máy nhà (RTX 3050), các mốc B3 trên Kaggle (T4). Chạy lại cùng cấu hình đã lệch khoảng 0,3 điểm (mục 4.14). Mọi so sánh B3 trong mục 4.24–4.30 đều chạy trên Kaggle với cùng code.
- Các mục 4.18–4.23 nhiều bản mới có 1 seed trên B0.
- Chưa đối chiếu với các phương pháp khác trong bảng của bài SAGE.

---

## Phụ lục A: dữ liệu gốc đi kèm (không cần đọc để hiểu)

| Thư mục | Có gì |
|---|---|
| `ket_qua/e17_bo_sam/` | So SAGE với bỏ SAM (mục 2.1). `runs_b3/` chứa 17 lần chạy SegFormer-B3 (VIS, IR, TB, SAGE, bỏ SAM × 3 seed, thêm 2 Student bỏ SAM khác): ma trận nhầm lẫn từng ảnh (`test_nham_last.npz`), cấu hình, log. Tính lại bảng bằng: `cd code/b3 && SAGE_RUNS=../../ket_qua/e17_bo_sam/runs_b3 python so_sanh_cap.py --bien_the vis tb sage nosam --model b3 --hau _1220_15lop` |
| `ket_qua/e18_chi_phi/` | Bảng chi phí đo trên T4 (mục 2.2) |
| `ket_qua/danh_doi/` | Chỉ số ghép so với mIoU (mục 2.3) |
| `ket_qua/split_cu/` | Số của các phương án mục 3 |
| `ket_qua/kham_pha/` | Chỉ số ghép trên val 151, sàn, log train (mục 4.2, 4.4, 4.6). `dev_kp_td_*` (TDFusion), `dev_kp_wir_*` (A1), `dev_kp_vung_*` (A2) cho mục 4.10, 4.12 |
| `ket_qua/kham_pha/sang_loc_b0/` | mIoU sàng lọc B0 (mục 4.7–4.9, 4.13–4.23): `runs/` là các lần chạy trên Kaggle (ma trận nhầm lẫn `dev_nham_last.npz`, IoU từng lớp, log); `runs_may_nha/` có các lần chạy trên máy nhà (bỏ SAM seed 0, A1, A2 λ = 10/30/100/mở rộng lớp, TDFusion, chỉ VIS, cận trên `vis_ir4`/`vis_ir4_lr10`/`hai_nhanh`/`hai_nhanh_visvis`, S-đèn, oracle, chưng cất, S-vùng – mục 4.10, 4.12–4.14, 4.18, 4.21, 4.23), kèm các bảng so cặp `runs_may_nha/so_sanh_lop_du_dev_b0_*.csv`; các bảng so cặp chạy Kaggle `so_sanh_lop_du_dev_b0_*.csv` (tên theo thí nghiệm: `a2_3seed`, `a4_s0`, `visvis`, `sxa_s0`, `moi_s0`, `lint_s0`, `mau`, `mau2_*`, `msrs_s0`, `vhn`, `vhn3seed`, `moilop` = đủ 14 lớp); `so_sanh_cap_dev_b0.csv` (14 lớp), `so_sanh_lop_du_dev_b0.csv` (10 lớp đủ ảnh), `miou_theo_sang_b0.csv`. Lưu ý: cột `d_tb` là hiệu trung bình các seed, `CI_anh_seed` là khoảng tin cậy dùng trong README |
| `ket_qua/kham_pha/sang_loc_b3/` | SegFormer-B3 trên val 151 (mục 4.8): 4 lần chạy, so cặp 14 lớp và 10 lớp |
| `ket_qua/dev228_b3/` | Tầng chọn (mục 4.24, 4.26–4.30): `runs/` có 27 lần chạy SegFormer-B3 trên Dev 228 (bỏ SAM, SAGE, TB, VIS × 3 seed; `vhn`, `vnc`, `a04_cvis` × 2 seed; `tbsua_992`, `tbsua_a04` × 3 seed; `tbsua_vhn3`, `orc_dung`, `orc_nguoc` × 1 seed), mỗi lần có ma trận nhầm lẫn từng ảnh `dev_nham_last.npz`, IoU từng lớp, cấu hình, log; bảng so cặp `so_sanh_lop_du_dev_b3*.csv`; `phan_tich/` là output các script chấm và phân tích 24–26/11 (tên file trùng tên script) |
| `ket_qua/kiem_cheo_kc1/` | Kiểm chéo theo cảnh (mục 4.28): 6 lần chạy SegFormer-B3 (TB, `a04` × 3 seed) train 782 ảnh, chấm 411 ảnh phần kiểm; `kq_kc1.json` |
| `ket_qua/kham_pha/log_2211_2311/` | Log 21–23/11: chạy cả quy trình (`chay_*.out.txt`), train Student (S-xa, S-đèn, chưng cất, S-vùng), bảng tỉ lệ chép (mục 4.18, 4.23) |
| `ket_qua/kham_pha/phan_tich/` | Phân tích person, tương phản người, nhầm lớp (mục 4.11); `nut_that_1_kenh.csv` (mục 4.17); `he_so_ir/` hệ số kéo về IR theo lớp (mục 4.19, 4.23); `a2_nosam_kp841_vung10/` cho mục 4.12; `proxy/` là thử chấm nhanh bằng SegFormer đóng băng (không dùng, vì phạt oan ảnh khác kiểu bỏ SAM) |
| `ket_qua/kham_pha/log_train_a1_a2/` | Log train Student A1, A2 (mọi λ, 3 seed), A4 và log chạy cả quy trình (`*.out.txt`: cận trên, A2 λ, A2 seed, A4) |
| `ket_qua/kham_pha/kiem_vung/` | Độ mạnh cạnh tại ranh giới hai lớp trên ảnh train (mục 4.15) |
| `ket_qua/chan_doan_m11/` | Log đo lực kéo (mục 4.5), các lần chạy nền B trước khi sửa BN |
| `hinh/` | Hình minh họa |
| `code/` | Code chạy được, xem Phụ lục B |

## Phụ lục B: chạy lại

### Cấu trúc `code/`

| Thư mục | Có gì |
|---|---|
| `b3/` | `chung.py` (đường dẫn, danh sách lớp, mIoU), `xuat_anh.py` (xuất ảnh ghép; thêm biến thể `vnc` và hàm `cong_vnc` (mục 4.26); thêm hậu tố `_mau`, `_mau2`, ảnh oracle `orc_dung`, `orc_nguoc`), `vung_hai_nhanh.py` (`vhn`: chia hai nửa, dựng ảnh, lưu bản đồ thầy cho S-vùng), `train_seg.py` / `eval_seg.py` (train, chấm SegFormer B0/B3), `so_sanh_cap.py` (so theo cặp seed), `so_sanh_lop_du.py` (so cặp chỉ trên lớp đủ ảnh). `train_seg.py` có thêm biến thể `vis_ir4`, `vis_ir4_lr10`, `hai_nhanh`, `hai_nhanh_visvis` (SegFormer nhận thẳng VIS + IR, để đo cận trên, mục 4.13) |
| `b3/sage_student/` | `model.py` (kiến trúc Student của SAGE) và **checkpoint Student đã train** (mỗi file khoảng 0,6 MB): `nosam*` (E17, 5 seed), `nosam_kp841*` (tập khám phá, 5 seed), `m11_kp`, `tbsua_kp841`, `tbsua_w150`, `m11a_l0_e30_bn` (đối chứng λ = 0), `nosam_kp841_w05/w06/w07` (A1), `nosam_kp841_vung10{,_s1,_s2}` (A2 đã chốt, 3 seed), `nosam_kp841_vung30/vung100/vung10mr` (A2 các λ khác), `nosam_kp841_vung10_tach1` (A4). Thêm 21–23/11: `model_xa.py` (nhánh ngữ cảnh rộng) với `sxa0`, `sxa10{,_s1}`, `sxa10mr{,_s1}`, `sden10`, `sden30` (mục 4.18); `lint`, `lint_a2`, `lint_a2_xa3`, `lint_nha` (mục 4.19); `nosam_du`; `nosam_kp841_nha2` (train lại bỏ SAM để đo nhiễu); `cc_vhn`, `cc_vhn_thuan`, `cc_vhn_thuan_w10`, `du_ft` (mục 4.23); `model_mat.py` + `du2_mat_ft` và `model_vung.py` + `svung` (mỗi file khoảng 12 MB vì chứa cả ResNet18). Thêm 24/11: `nosam_992` (bỏ SAM, train 992 ảnh, nền mốc Dev 228), `tbsua_992` (nền của `a04`, mục 4.27) |
| `e17/` | Train Student bỏ SAM (`train_student_nosam.py`; thêm tùy chọn `--w_ir` cho A1, `--lam_vung` và `--lop_vat` cho A2, `--lam_tach` cho A4, `--them` để train thêm dữ liệu ngoài như MSRS; thêm 21–23/11: `--kien_truc {goc,xa,mat}`, `--dim`, `--xa_them`, `--lam_den`, `--lr_du`, `--w_lop`, chưng cất `--ref`, `--chi_ref`, `--w_khac`, `--khoi_tao`, `--dong_bang_goc`); `train_student_vung.py` (S-vùng, mục 4.23); so sánh E17 |
| `cai_tien/` | M11 (`m11_hoc_chung.py`, dùng hàm trong `train_student_seg.py`), nền B (`tb_sua.py`), chỉ số ghép (`chi_so_ghep.py`, `cong2_ci.py`), độ lệch ảnh (`lech_y.py`), xuất ảnh TDFusion (`xuat_tdfusion.py`, cần clone `HaowenBai/TDFusion` kèm checkpoint). `danh_doi/` là script phân tích, đường dẫn còn cứng, chỉ để đọc |
| `scene_v1/` | Script chạy từng thí nghiệm; phân tích (`phan_tich_nosam.py`, `phan_tich_a2.py`, `do_tuong_phan.py`, `miou_theo_sang.py`, `kiem_proxy.py`, `oracle/oracle.py`, `kiem_a3/` đo cạnh và độ tách vùng mục 4.15; `phan_tich_nut_that.py` mục 4.17; `tinh_w_lop.py`, `w_lop.json`, `do_he_so_ir.py` mục 4.19; `oracle_doan.py` mục 4.21; `chep_delta.py`, `he_so_anh.py` mục 4.23); `_vhn/` (chia hai nửa cho `vhn` và bản đồ thầy `p_thay.npz`); `nua_kp/`; `kham_pha/` (train 841 / val 151); `goc/` (split scene_v1 992/228 của nhóm, kèm `scene_decision.md`). Thêm 24–26/11: `chay_tbsua992.sh` (train `tbsua_992`, xuất ảnh, đo rào), `lieu_alpha.py` (chọn α trên Train), `dung_alpha.py` (dựng `a04`), `chon_cong_tbsua.py`, `chon_cong_tbsua2.py`, `dung_2_nhanh.py` (cổng VIS, `tbsua_vhn3`), `dung_a04_cvis.py`, `kiem_cong_cvis.py` (mục 4.30), `cham_tbsua3.py`, `cham_a04_3s.py`, `cham_a04cvis.py`, `phan_tich_ho_tb.py`, `dai_dien_dev.py` + `dai_dien_dev.csv` (độ sáng từng ảnh), `chia_kiem_cheo.py` + `kiem_cheo_p1.csv`, `phan_tich_kc1.py`, `gop_kc1_dev.py` (mục 4.28); `_vhn3/p_thay_vhn3.npz` (bản đồ thầy `vhn3`) |
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
| Đối chứng cận trên `hai_nhanh_visvis`, 3 seed (mục 4.13) | `bash scene_v1/chay_visvis.sh` | khoảng 1,4 giờ mỗi seed |
| Đổi kiến trúc Student, S-xa (mục 4.18) | `bash scene_v1/chay_sxa.sh` (mặc định chỉ train Student và đo sàn; `B0=1` để chấm B0 trên máy) | khoảng 1 giờ phần Student |
| Loss giữ đèn, S-đèn (mục 4.18) | `bash scene_v1/chay_sden.sh`; chấm B0: `B0=1 BT_B0=sden10 bash scene_v1/chay_sden.sh` | khoảng 1 giờ + 1,2 giờ |
| Kéo cường độ theo lớp (mục 4.19) | Bảng trọng số: `python scene_v1/tinh_w_lop.py`; train + chấm: `bash scene_v1/chay_lint.sh`; hệ số kéo về IR: `bash scene_v1/chay_mat.sh` | khoảng 1 giờ mỗi bản |
| Chỉnh màu (mục 4.20) | `cd b3 && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY xuat_anh.py --bien_the nosam_kp841_mau2` rồi chấm B0 như `chay_b0_sangloc.sh` | vài phút + 1,2 giờ |
| Oracle theo vùng (mục 4.21) | `bash scene_v1/chay_oracle.sh`; nhãn đoán: `python scene_v1/oracle_doan.py` (`--nguon hai_nhanh --hau _hn` cho nhãn từ `hai_nhanh`) | khoảng 2,3 giờ |
| `vhn` (mục 4.22) | `cd b3 && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY vung_hai_nhanh.py chia --ra ../scene_v1/_vhn`; train `hai_nhanh` B0 trên từng nửa: `SAGE_MAN=../scene_v1/_vhn/man_A SAGE_RUNS=../scene_v1/_vhn/runs_A $PY train_seg.py --bien_the hai_nhanh --model b0 --seed 0` (tương tự `man_B`, `runs_B`); dựng ảnh: `$PY vung_hai_nhanh.py dung --ra ../scene_v1/_vhn`; bản đồ cho S-vùng: `$PY vung_hai_nhanh.py luu_p --ra ../scene_v1/_vhn` | khoảng 1 giờ mỗi nửa (T4) |
| Chưng cất vào Student (mục 4.23) | `bash scene_v1/chay_chung_cat.sh`, `chay_thuan.sh`, `chay_du.sh`, `chay_du2.sh`, rồi chấm `chay_du2_b0.sh` | khoảng 20–60 phút mỗi script phần Student, 1,2 giờ mỗi lần chấm B0 |
| S-vùng (mục 4.23) | `cd e17 && SAGE_MAN=../scene_v1/kham_pha SO_TRAIN=841 $PY train_student_vung.py --ten svung`; chấm: `bash scene_v1/chay_svung_b0.sh`; tỉ lệ chép: `cd scene_v1 && $PY chep_delta.py dev vhn nosam_kp841 ck:svung` | 7 phút + 1,2 giờ |
| `tbsua_992` và rào (mục 4.27) | `bash scene_v1/chay_tbsua992.sh` | khoảng 1 giờ |
| Chọn α trên Train, dựng `a04` (mục 4.27) | `SAGE_MAN=scene_v1/goc python scene_v1/lieu_alpha.py`; `SAGE_MAN=scene_v1/goc python scene_v1/dung_alpha.py` (cần ảnh `tbsua_992` trong `SAGE_ANH`) | vài phút |
| Chấm `a04` 3 seed trên Dev 228 (mục 4.27) | `python scene_v1/cham_a04_3s.py` (đọc `ket_qua/dev228_b3/runs/`) | vài giây |
| Kiểm chéo `kc1` (mục 4.28) | `python scene_v1/phan_tich_kc1.py`; `python scene_v1/gop_kc1_dev.py` | vài phút |
| `a04_cvis` (mục 4.30) | `SAGE_MAN=scene_v1/goc python scene_v1/dung_a04_cvis.py`; chấm: `python scene_v1/cham_a04cvis.py`; kiểm cổng: `SAGE_MAN=scene_v1/goc python scene_v1/kiem_cong_cvis.py` (cần ảnh `a04`, `a04_cvis`) | vài phút |

Đã thử: xuất 3 ảnh `tbsua_w150` bằng code trong repo này, kết quả **giống hệt từng điểm ảnh** với bản gốc. Ngày 26/11: chạy lại trong repo `cham_a04_3s.py`, `cham_a04cvis.py`, `cham_tbsua3.py`, `phan_tich_kc1.py`, `gop_kc1_dev.py`, `phan_tich_ho_tb.py` ra **đúng số** trong `ket_qua/dev228_b3/phan_tich/`. Các script khác đã chạy thử `--help` và đều nạp được.
