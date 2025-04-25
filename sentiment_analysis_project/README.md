# Dự án Thực tập: Web App Phân Tích Cảm Xúc Khách Hàng

Dự án này xây dựng một ứng dụng web đơn giản để phân tích cảm xúc (Tích cực, Tiêu cực, Trung tính) từ phản hồi của khách hàng bằng cách sử dụng mô hình Transformer (DistilBERT) được fine-tune trên PyTorch và Hugging Face. Giao diện được xây dựng bằng Streamlit và có một API tùy chọn bằng FastAPI.
Dùng Goole Colab để train model 

## Các Chức Năng Chính

*   **Phân tích văn bản đơn lẻ:** Nhập trực tiếp một phản hồi và xem nhãn cảm xúc dự đoán, điểm tin cậy, và biểu đồ phân bổ xác suất.
*   **Phân tích hàng loạt (File CSV):** Tải lên file CSV chứa các phản hồi, chọn cột văn bản, và xem thống kê tổng hợp (biểu đồ cột/tròn) cùng bảng kết quả chi tiết. Có thể tải xuống kết quả phân tích.
*   **(Tùy chọn) API Endpoint:** Endpoint `/predict/` (POST) nhận dữ liệu text và trả về kết quả phân tích dưới dạng JSON.
*   **Đánh giá Model:** Xem các chỉ số hiệu năng (Accuracy, Precision, Recall, F1), ma trận nhầm lẫn và biểu đồ quá trình huấn luyện trực quan trên giao diện web.

## Cấu trúc Thư mục