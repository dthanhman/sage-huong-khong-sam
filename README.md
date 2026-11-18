# Ghép ảnh IR–VIS không cần SAM: tổng kết hướng thử nghiệm

Người làm: Đào Thanh Mân. Cập nhật: 18/11/2026 tối.

File này đọc riêng là đủ hiểu. Các thư mục đi kèm chỉ chứa số liệu gốc và code, dùng khi muốn kiểm lại.

---

## 1. Bối cảnh trong một phút

- **SAGE** (CVPR 2025) ghép ảnh hồng ngoại (IR) với ảnh thường (VIS) thành một ảnh. Mạng chạy thật là một **Student** nhỏ (0,136M tham số). Lúc train, Student được dạy bởi một Teacher dùng **SAM**, mô hình phân đoạn rất nặng.
- Nhóm đánh giá ảnh ghép bằng cách train **SegFormer** trên ảnh ghép, rồi đo **mIoU** phân đoạn trên dataset FMB. mIoU càng cao thì ảnh ghép càng giúp hiểu cảnh.
- **TB** là cách ghép đơn giản nhất: lấy trung bình (VIS + IR) / 2 trên kênh độ sáng, màu giữ từ VIS. Không cần học, không tốn gì.
- Hướng này **khác** hướng Q06 của nhóm. Q06 fine-tune Student SAGE có sẵn, nên vẫn dựa gián tiếp vào SAM. Hướng này **bỏ hẳn SAM**.

**Câu hỏi:** bỏ SAM thì có giữ được mIoU không? Có làm được phương pháp không SAM mà mIoU hơn cả SAGE lẫn TB không?

---

## 2. Vì sao chọn hướng này

### 2.1. Bỏ SAM không làm giảm mIoU

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

### 2.2. Bỏ SAM rẻ hơn rất nhiều

Đo trên GPU Kaggle T4:

| | SAGE có SAM | Bỏ SAM | TB |
|---|---|---|---|
| SAM tạo mask cho 1.220 cặp ảnh | 99,2 giờ | 0 | 0 |
| Train 100 epoch | 26,3 giờ | 0,5 giờ | 0 |
| **Tổng** | **≈ 125 giờ** | **0,5 giờ** | 0 |
| Bộ nhớ GPU lúc train | 11,9 GB | 1,1 GB | – |
| Thời gian chạy một ảnh | 54,3 ms | 54,3 ms | 0,05 ms |

Bỏ SAM rẻ hơn khoảng 250 lần lúc train. Lúc chạy thì như nhau, vì cùng một Student.

### 2.3. Chỉ số ghép không đoán được mIoU

Các chỉ số đo chất lượng ảnh ghép (SCD, Q_ABF, MS-SSIM) **xếp hạng các phương pháp khác với mIoU**. Ví dụ: SAGE có SCD cao nhất (1,71), nhưng mIoU thấp hơn TB (SCD chỉ 1,38).

Vì vậy mình dùng chỉ số ghép làm **sàn**, chỉ để bảo đảm ảnh không bị hỏng. **Chọn phương án thì dựa vào mIoU.**

### 2.4. Mục tiêu đặt ra

Phương pháp của mình phải:
1. **không cần SAM**, kể cả không khởi tạo từ Student SAGE;
2. có mIoU **hơn cả SAGE và TB**, cùng cách chấm, nhiều seed;
3. **qua sàn** chỉ số ghép.

TB là đối thủ khó nhất: không tốn gì mà mIoU đang cao nhất.

---

## 3. Đã thử gì trên split cũ (16–18/11), và vì sao chưa đạt

Thiết lập: train 1.045 ảnh, chấm 175 ảnh Dev, SegFormer-B0. Đạt khi hơn TB ít nhất +1,0 điểm ở cả 2 seed. TB đạt 53,13 (seed 0) và 53,27 (seed 1).

| Phương án | Ý tưởng | Kết quả | Vì sao dừng |
|---|---|---|---|
| M1 | Fine-tune Student SAGE bằng loss phân đoạn từ một SegFormer đóng băng | Loss phân đoạn gần như không giảm | Vẫn khởi tạo từ SAGE; SegFormer đóng băng đã học thuộc ảnh train nên không chỉ được hướng sửa |
| M4 | Giữ TB, học thêm một đầu trộn nhỏ | 52,80 / 53,07 (≈ TB −0,3) | Đầu trộn gần như không học |
| M5 | Trộn IR theo độ sáng ảnh (luật tay) | 52,10 / 51,27 | Thêm IR vào ảnh sáng làm hại |
| M8b | TB cộng thêm chi tiết cạnh | 52,63 / 53,13 | Không hơn TB |
| M8c | VIS + IR − trung bình IR (để đẩy SCD lên) | 52,19 / 51,46 | SCD rất cao mà mIoU thấp; nhà cửa bị cháy trắng |
| M11 (bản đầu) | Student bỏ SAM và SegFormer **học cùng nhau** | Ảnh nghiêng về VIS, rớt SCD | Chưa chấm mIoU |

**Rút ra:**
- Trộn VIS/IR bằng luật tay hay theo tỉ lệ cố định đều không hơn TB.
- Dùng SegFormer đóng băng làm "thầy" không hiệu quả. Phải cho SegFormer học cùng mạng ghép.
- Đẩy chỉ số ghép lên không kéo mIoU lên theo.

---

## 4. Đang làm: split mới của nhóm (scene_v1)

### 4.1. Chia dữ liệu

Nhóm chốt split **scene_v1**: 992 ảnh Train và 228 ảnh Dev, chia theo cảnh. Quy tắc:
- **Dev 228 chỉ chấm một lần ở cuối**, cho phương án đã chọn. Không dùng để thử đi thử lại.
- Không dùng Test 280 để chọn phương án.

Để thử nhiều lần mà không đụng Dev, mình cắt thêm một **tập khám phá** từ 992 ảnh Train:
- **val 151**: nguyên 3 đoạn ảnh liền số (00136–00156, 00649–00700, 01223–01300), có cả ảnh tối và ảnh ban ngày. Lấy nguyên đoạn để không cắt đôi một cảnh.
- **train 841**: phần còn lại.

Vì vậy có **hai tầng chấm**:

| | Tầng 1: sàng lọc | Tầng 2: chốt |
|---|---|---|
| Mục đích | Chọn phương án | Báo cáo kết quả |
| SegFormer train trên | 841 ảnh | 992 ảnh |
| Chấm trên | val 151 | Dev 228 |
| Loại SegFormer | B0 (nhỏ, chạy máy nhà) | B3 (như bài báo, chạy Kaggle) |
| Số lần chấm | Nhiều lần | Một lần |

Số của hai tầng **không so trực tiếp với nhau**, vì khác mạng, khác dữ liệu train, khác tập chấm.

### 4.2. Sàn ảnh ghép

Mình train Student bỏ SAM 5 lần với 5 seed, đo trên val 151, rồi đặt ngưỡng = SAGE − 2 × độ lệch chuẩn giữa các seed:
- **MS-SSIM ≥ 99%** của SAGE;
- **Q_ABF ≥ 98%** của SAGE;
- SCD chỉ báo kèm, vì chính Student bỏ SAM cũng chỉ đạt 86–92%.

### 4.3. Hai nền phương án

**Nền A – M11 (học cùng):** Student khởi tạo từ bản bỏ SAM. Ảnh ghép đi thẳng vào SegFormer-B0, và hai mạng học cùng nhau. Loss của Student = λ × (loss phân đoạn) + β × (loss nội dung). 5 epoch đầu chỉ SegFormer học, để nó đủ tốt trước khi "dạy" Student.

**Nền B – TB + phần sửa:** ảnh ghép = TB + một phần sửa nhỏ do Student học. Lúc đầu phần sửa bằng 0, nên ảnh đúng bằng TB. Loss = giữ gần TB + giữ cạnh. Lý do: TB có mIoU tốt nhất nhưng mất cạnh (Q_ABF chỉ 60% SAGE), nên giữ TB và chỉ học cách trả cạnh. Bước sau sẽ thêm loss phân đoạn.

### 4.4. Kết quả chỉ số ghép trên val 151 (% so với SAGE)

| Ảnh ghép | Loại | MS-SSIM | Q_ABF | SCD (kèm) | Qua sàn? |
|---|---|---|---|---|---|
| SAGE | mốc | 100 | 100 | 100 | – |
| TB | mốc | 105,0 | 60,3 | 82,7 | ❌ (Q_ABF) |
| TB + chi tiết (2 kiểu) | đối chứng không học | 100,5–100,9 | 99,9–100,8 | 88–91 | ✅ |
| Student bỏ SAM | nền | 101,4 | 108,2 | 88,9 | ✅ |
| M11, λ = 1 | nền A | 101,1 | 109,9 | 89,1 | ✅ |
| TB + phần sửa, w_tb = 15 | nền B | 98,9 | 110,1 | 91,0 | ❌ (MS-SSIM) |
| **TB + phần sửa, w_tb = 150** | nền B | **101,4** | **109,2** | 89,9 | ✅ |

- Với w_tb = 15, phần sửa quá tay (trung bình 11,3 mức xám), ảnh trôi về phía VIS. Tăng trọng số giữ TB lên 150 thì phần sửa còn 4,9 mức xám và qua sàn.
- **Chưa phương án nào có mIoU trên split mới.** Qua sàn chỉ có nghĩa là ảnh không bị hỏng.

### 4.5. Chẩn đoán M11: vì sao học phân đoạn mà ảnh gần như không đổi

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

---

## 5. Bước tiếp theo

1. Hoàn tất sửa M11 theo mục 4.5.
2. **Sàng lọc bằng mIoU (tầng 1):** train SegFormer-B0 trên 841 ảnh, chấm val 151, cho: TB, SAGE, bỏ SAM, M11 bản sửa, TB + phần sửa (và chỉ VIS nếu nhóm muốn). Chạy seed 0 trước (khoảng 6–7 giờ). Phương án thua rõ thì bỏ; có triển vọng thì thêm seed 1–2.
3. TB + phần sửa: thêm loss phân đoạn.
4. **Vòng chốt (tầng 2):** phương án qua lọc → train lại trên 992 ảnh → SegFormer-B3, 3 seed → chấm Dev 228 một lần, so với SAGE, VIS, TB cùng cấu hình. Thêm một dòng vào bảng chi phí.

## 6. Cần nhóm chốt

| Việc | Đề xuất |
|---|---|
| Mốc trên Dev 228 | Hiện mới có SAGE 59,05 và VIS 59,43, **chỉ 1 seed, thiếu TB**. Đề xuất chạy SAGE, VIS, TB, mỗi cái 3 seed (còn 7 lượt, mỗi lượt khoảng 2,6 giờ trên Kaggle) |
| Tiêu chí qua sàng lọc | Hơn **cả TB và SAGE**, dương ở ít nhất 2/3 seed |
| Thêm "chỉ VIS" vào sàng lọc | Có, làm mốc dưới cùng |
| Hạn nộp | Một tài liệu ghi 20/11, tài liệu khác ghi 28/11, cần hỏi lại |

## 7. Giới hạn

- Số trên tập khám phá chỉ để so các phương án với nhau. Không báo cáo như kết quả cuối.
- SegFormer-B0 yếu hơn B3. Trước đây, chênh lệch đo trên B0 thường lớn hơn trên B3.
- Chênh lệch khoảng 1 điểm mIoU giữa TB, bỏ SAM và SAGE xấp xỉ mức dao động do seed. Phải có nhiều seed và so theo cặp seed mới kết luận được.

---

## Phụ lục A: dữ liệu gốc đi kèm (không cần đọc để hiểu)

| Thư mục | Có gì |
|---|---|
| `ket_qua/e17_bo_sam/` | So SAGE với bỏ SAM (mục 2.1). `runs_b3/` chứa 17 lần chạy SegFormer-B3 (VIS, IR, TB, SAGE, bỏ SAM × 3 seed, thêm 2 Student bỏ SAM khác): ma trận nhầm lẫn từng ảnh (`test_nham_last.npz`), cấu hình, log. Tính lại bảng bằng: `cd code/b3 && SAGE_RUNS=../../ket_qua/e17_bo_sam/runs_b3 python so_sanh_cap.py --bien_the vis tb sage nosam --model b3 --hau _1220_15lop` |
| `ket_qua/e18_chi_phi/` | Bảng chi phí đo trên T4 (mục 2.2) |
| `ket_qua/danh_doi/` | Chỉ số ghép so với mIoU (mục 2.3) |
| `ket_qua/split_cu/` | Số của các phương án mục 3 |
| `ket_qua/kham_pha/` | Chỉ số ghép trên val 151, sàn, log train (mục 4.2, 4.4) |
| `ket_qua/chan_doan_m11/` | Log đo lực kéo (mục 4.5) |
| `hinh/` | Hình minh họa |
| `code/` | Code chạy được, xem Phụ lục B |

## Phụ lục B: chạy lại

### Cấu trúc `code/`

| Thư mục | Có gì |
|---|---|
| `b3/` | `chung.py` (đường dẫn, danh sách lớp, mIoU), `xuat_anh.py` (xuất ảnh ghép), `train_seg.py` / `eval_seg.py` (train, chấm SegFormer B0/B3), `so_sanh_cap.py` (so theo cặp seed) |
| `b3/sage_student/` | `model.py` (kiến trúc Student của SAGE) và **checkpoint Student đã train** (mỗi file khoảng 0,6 MB): `nosam*` (E17, 5 seed), `nosam_kp841*` (tập khám phá, 5 seed), `m11_kp`, `tbsua_kp841`, `tbsua_w150` |
| `e17/` | Train Student bỏ SAM; so sánh E17 |
| `cai_tien/` | M11 (`m11_hoc_chung.py`, dùng hàm trong `train_student_seg.py`), nền B (`tb_sua.py`), chỉ số ghép (`chi_so_ghep.py`, `cong2_ci.py`), độ lệch ảnh (`lech_y.py`). `danh_doi/` là script phân tích, đường dẫn còn cứng, chỉ để đọc |
| `scene_v1/` | Script chạy từng thí nghiệm; `kham_pha/` (train 841 / val 151); `goc/` (split scene_v1 992/228 của nhóm, kèm `scene_decision.md`) |
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
| Sàng lọc mIoU bằng B0 (chưa chạy) | `bash scene_v1/chay_b0_sangloc.sh` | khoảng 1,2 giờ mỗi lượt |

Đã thử: xuất 3 ảnh `tbsua_w150` bằng code trong repo này, kết quả **giống hệt từng điểm ảnh** với bản gốc. Các script khác đã chạy thử `--help` và đều nạp được.
