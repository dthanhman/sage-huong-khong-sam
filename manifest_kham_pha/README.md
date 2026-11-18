# scene_v1 – split của nhóm (Quân, 16/11/2026) và tập khám phá nội bộ

- `goc/`: bản sao manifest scene_v1 (992 Train / 228 Dev) + `scene_decision.md`, lấy từ `q02b_kaggle_bundle.zip` (Drive nhóm). Không sửa.
- `kham_pha/`: tách TỪ 992 Train để khám phá phương pháp, theo quy tắc Tab 4 (không lặp theo Dev 228, không dùng Test).
  - `dev.csv` (= val nội bộ, 151 ảnh): lấy NGUYÊN 3 đoạn id liền nhau của Train scene_v1 – 00136–00156 (21, có ảnh tối), 00649–00700 (52, ảnh mờ tối), 01223–01300 (78, ban ngày). Lấy cả đoạn nên không cắt đôi cảnh nào; các đoạn được ngăn với Train còn lại bởi khoảng id thuộc Dev/Test.
  - `train.csv` (841 ảnh): phần còn lại. `test.csv`: rỗng (cấm Test).
  - Giới hạn: chưa kiểm ORB như Quân, nên có thể còn cùng địa điểm ở id xa. Chỉ dùng để so tương đối giữa phương pháp.
- Dùng: đặt `SAGE_MAN=<đường dẫn>/scene_v1/kham_pha` và `SO_TRAIN=841` khi chạy script trong `b3/`, `e17/`, `cai_tien/`.

## Chốt sàn trên kham_pha (thiết lập 18/11, CHƯA CHẠY)

- Script: `scene_v1/chay_san.sh` (chạy từ `tai_lap/`: `bash scene_v1/chay_san.sh`, ~1 h máy nhà, phiên phải mở suốt).
- Làm gì: train `nosam_kp841_s1..s4` (cùng công thức E17, `--chi_train`, 841 ảnh, chỉ khác `--seed`) → xuất ảnh → chỉ số ghép trên val 151 cho `sage` + 5 seed nosam (seed 0 = `nosam_kp841` đã có) → `cong2_ci.py` in CI95 và SD do seed.
- Ra: `b3/sage_student/nosam_kp841_s*.pt`, log `scene_v1/nosam_kp841_s*.log`, bảng `cai_tien/chi_so_ghep/dev_kp_san_*.csv`, tóm tắt `scene_v1/san_ci.txt`.
- **Công thức đặt trước (giữ như 17/11 trên split cũ):** ngưỡng mỗi chỉ số = SAGE − 2 × SD(5 seed nosam, tính theo % SAGE), làm tròn tới 1%. Áp như nhau cho mọi phương án trên kham_pha.
- So với sàn cũ (split cũ: SCD 95 / MS-SSIM 98 / Q_ABF 93 % SAGE): nếu sàn mới lệch > 2 điểm % thì ghi rõ, dùng sàn mới cho số trên scene_v1.
- Lưu ý: theo kiểm 18/11 (`cai_tien/danh_doi/`), chỉ số ghép KHÔNG dự đoán được mIoU, và Tab 4 quy định chỉ số ghép chỉ để chẩn đoán. Sàn này dùng để nói "phương án có làm hỏng ảnh ghép không", không để chọn phương án. Độ dao động 5 seed nosam cũng cho biết M11 (m11_kp) khác nosam có vượt nhiễu seed không.

### Kết quả chốt sàn (18/11 16h30)
5 seed nosam (val 151), % SAGE: SCD 86,1–91,6 (SD 2,11) / MS-SSIM 101,4–102,1 (SD 0,34) / Q_ABF 105,7–108,2 (SD 1,23).
**Sàn scene_v1: SCD ≥ 96%, MS-SSIM ≥ 99%, Q_ABF ≥ 98% SAGE.** Q_ABF lệch sàn cũ (93%) 5 điểm > 2 → dùng sàn mới.
Lưu ý: chính nosam rớt SCD (86–92%), m11_kp 89,1% cũng rớt. Chờ Mân chọn: giữ SCD làm rào cứng hay chỉ báo kèm.
