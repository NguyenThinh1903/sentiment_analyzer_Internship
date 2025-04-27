# config.py (Phiên bản Lai ghép - Thêm Ngưỡng)

import torch
import os
from dotenv import load_dotenv

load_dotenv()

# --- Đường dẫn Cơ sở trên Máy Local ---
PROJECT_DIR = '.'

# --- Cấu hình Nguồn Dữ liệu ---
TEXT_COLUMN = "comment" # Tên cột text mong đợi
# DATA_SOURCES, PROCESSED_DATA_DIR, và các file CSV giữ nguyên như trước
# ... (giữ nguyên phần DATA_SOURCES, PROCESSED_DATA_DIR, ...) ...
DATA_SOURCES = [
     {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file1_en.csv'), 'language': 'en',
        'text_col': 'Text', 'label_type': 'rating', 'label_col': 'Score', 'rating_scale': (1, 5)
    },
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file2_en.csv'), 'language': 'en',
        'text_col': 'review', 'label_type': 'text_label', 'label_col': 'sentiment',
        'label_values': {'positive': 2, 'negative': 0, 'neutral': 1}
    },
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file3_vi.csv'), 'language': 'vi',
        'text_col': 'comment', 'label_type': 'rating', 'label_col': 'rate', 'rating_scale': (1, 5)
     },
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file4_en.csv'), 'language': 'en',
        'text_col': 'Review', 'label_type': 'rating', 'label_col': 'Rating', 'rating_scale': (1, 5)
     },
]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed")
COMBINED_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "combined_processed_4files.csv")
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train_combined_4files.csv")
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, "val_combined_4files.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test_combined_4files.csv")


# --- Ánh xạ Nhãn Mục tiêu ---
TARGET_LABEL_MAP = {
    0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"
}
NUM_LABELS = len(TARGET_LABEL_MAP)

# --- Cấu hình Model ---
MODEL_NAME = "saved_model" # Hoặc XLM-R base
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "saved_model")

# --- Cấu hình Huấn luyện ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 160
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 1
TEST_SPLIT_SIZE = 0.15
VALIDATION_SPLIT_SIZE = 0.1

# --- Ngưỡng Tin Cậy và Điều kiện Gọi AI ---
# !!! THÊM CÁC THAM SỐ NÀY !!!
CONFIDENCE_THRESHOLD = 0.80 # Gọi Gemini nếu confidence < ngưỡng này (Thử nghiệm giá trị này)
ALWAYS_CHECK_NEGATIVE = True # Luôn gọi Gemini cho trường hợp "Tiêu cực" ?
# ALWAYS_CHECK_NEUTRAL = False # Có muốn luôn kiểm tra trường hợp "Trung tính" không?

# --- Cấu hình Trực quan hóa ---
VISUALIZATION_DIR = os.path.join(PROJECT_DIR, "visualizations")
CONFUSION_MATRIX_FILE = os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")
TRAINING_CURVES_FILE = os.path.join(VISUALIZATION_DIR, "training_curves.png")
CLASSIFICATION_REPORT_FILE = os.path.join(VISUALIZATION_DIR, 'classification_report.txt')
EVALUATION_SUMMARY_FILE = os.path.join(VISUALIZATION_DIR, 'evaluation_summary.json')
ERROR_ANALYSIS_FILE = os.path.join(VISUALIZATION_DIR, 'error_analysis.csv')

# --- API Configuration ---
API_HOST = "127.0.0.1"
API_PORT = 8000

# --- Cấu hình API Bên ngoài (Gemini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("-" * 50)
    print("CẢNH BÁO: GEMINI_API_KEY chưa được đặt.")
    print("Tạo file .env và thêm: GEMINI_API_KEY=YOUR_KEY")
    print("Chức năng Gemini sẽ bị vô hiệu hóa.")
    print("-" * 50)