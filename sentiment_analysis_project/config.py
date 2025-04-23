# config.py (Đã Việt hóa chú thích và nhãn)

import torch
import os

# --- Cấu hình Dữ liệu ---
RAW_DATA_PATH = "data/raw/reviews.csv" # Đường dẫn đến file dữ liệu thô
PROCESSED_DATA_DIR = "data/processed" # Thư mục lưu dữ liệu đã xử lý
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train.csv") # File huấn luyện
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, "val.csv")   # File kiểm định
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test.csv")   # File kiểm thử

# --- Tên cột trong file CSV gốc (Đã cập nhật) ---
TEXT_COLUMN = "comment"      # Cột chứa văn bản phản hồi
RATING_COLUMN = "rate"       # Cột chứa điểm rating (ví dụ: 1-5)

# --- Ánh xạ Nhãn ---
# Định nghĩa ánh xạ từ nhãn số (0, 1, 2) sang tên nhãn tiếng Việt
LABEL_MAP = {
    0: "Tiêu cực", # Nhãn Tiêu cực
    1: "Trung tính",  # Nhãn Trung tính
    2: "Tích cực"  # Nhãn Tích cực
}
NUM_LABELS = len(LABEL_MAP) # Số lượng nhãn = 3

def map_rating_to_label(rating):
    """Ánh xạ điểm rating (từ cột RATING_COLUMN) sang nhãn số (0, 1, 2)."""
    try:
        rating = int(float(rating))
    except (ValueError, TypeError):
        return 1 # Mặc định là Trung tính nếu không phải số

    if rating in [1, 2]:
        return 0  # Tiêu cực
    elif rating == 3:
        return 1  # Trung tính
    elif rating in [4, 5]:
        return 2  # Tích cực
    else:
        return 1 # Mặc định là Trung tính cho các trường hợp khác

# --- Cấu hình Model ---
MODEL_NAME = "distilbert-base-uncased" # Tên model pre-trained từ Hugging Face
MODEL_SAVE_PATH = "saved_model"        # Thư mục lưu model đã fine-tune

# --- Cấu hình Huấn luyện ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Sử dụng GPU nếu có, nếu không dùng CPU
MAX_LENGTH = 128  # Độ dài tối đa của chuỗi token
BATCH_SIZE = 16   # Kích thước batch (giảm nếu gặp lỗi hết bộ nhớ GPU)
LEARNING_RATE = 2e-5 # Tốc độ học
NUM_EPOCHS = 3    # Số lượng epochs huấn luyện
TEST_SPLIT_SIZE = 0.15 # Tỷ lệ dữ liệu cho tập test
VALIDATION_SPLIT_SIZE = 0.1 # Tỷ lệ dữ liệu cho tập validation (tính trên phần còn lại)

# --- Cấu hình Trực quan hóa ---
VISUALIZATION_DIR = "visualizations" # Thư mục lưu các hình ảnh biểu đồ
CONFUSION_MATRIX_FILE = os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")
# Đường dẫn lưu các biểu đồ khác nếu cần (ví dụ: loss_curves.png)
TRAINING_CURVES_FILE = os.path.join(VISUALIZATION_DIR, "training_curves.png")
CLASSIFICATION_REPORT_FILE = os.path.join(VISUALIZATION_DIR, 'classification_report.txt')

# --- Cấu hình API (Tùy chọn) ---
API_HOST = "127.0.0.1" # Địa chỉ host cho API
API_PORT = 8000        # Cổng cho API