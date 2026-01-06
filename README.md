# Brain Tumor Diagnostic

Hệ thống hỗ trợ chẩn đoán u não qua ảnh cộng hưởng từ (MRI) sử dụng kỹ thuật **Block-wise Fine-tuning**, **Ensemble Learning**, **Test-Time Augmentation** kết hợp giữa **VGG19** và **MobileNet**. Hệ thống tích hợp giải thích AI (XAI) thông qua **Grad-CAM**.

## Thành viên nhóm thực hiện

| Họ và Tên | MSSV |
| --- | --- |
| Hồ Phạm Quốc Bảo | 24520152 |
| Nguyễn Chí Tiến Thịnh | 24521688 |

## Tính năng nổi bật

* **Block-wise Fine-tuning:** Áp dụng chiến lược unfreeze mô hình theo từng block (từ $B_1$ đến $B_n$).
    * Giúp mô hình chuyển đổi từ việc nhận diện các đặc trưng tổng quát (cạnh, màu sắc) sang các đặc trưng y tế chuyên sâu (vân bề mặt khối u, ranh giới khối u).
    * Tối ưu hóa tốc độ hội tụ và tránh hiện tượng Overfitting trên tập dữ liệu CE-MRI nhỏ.
* **Ensemble Model:** Kết hợp chiến lược giữa VGG19 và MobileNetV2 để tận dụng tối đa khả năng trích xuất đặc trưng đa dạng.
* **Test Time Augmentation (TTA):** Thực hiện dự đoán trên nhiều biến thể ảnh (Flip, Brightness, Contrast) để đưa ra kết quả cuối cùng ổn định và khách quan nhất.
* **Explainable AI (Grad-CAM):** Trực quan hóa bản đồ nhiệt (Heatmap) tại lớp Convolution cuối cùng, giúp bác sĩ xác định chính xác vị trí khối u mà AI đang tập trung phân tích.
* **Modern Web UI:** Giao diện Dashboard chuyên nghiệp, hỗ trợ quy trình tải ảnh - phân tích - báo cáo chỉ trong một trang duy nhất.

## Dữ liệu sử dụng (Dataset)

Hệ thống được huấn luyện trên tập dữ liệu CE-MRI (Contrast-enhanced MRI).

* **[Dataset gốc (Chưa xử lý)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)**
* **[Dataset đã tiền xử lý](https://www.kaggle.com/datasets/hophamsailam/5-fold-brain-tumor-contrast-enhanced)**

## Hướng dẫn cài đặt và khởi chạy

### 1. Cài đặt môi trường

```bash
# Clone dự án
git clone https://github.com/24520152-ben/brain-tumor-classification.git

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo trên Windows
.venv\Scripts\activate.bat

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Khởi chạy hệ thống

```bash
python deployment/backend/main.py
```
Sau khi chạy, truy cập giao diện tại: **`http://localhost:8000`**