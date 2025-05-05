# config.py (Phiên bản Final cho Local)

import torch
import os
from dotenv import load_dotenv

load_dotenv() # Đọc file .env ở thư mục gốc

# --- Đường dẫn Cơ sở ---
PROJECT_DIR = '.' # Giả định chạy từ thư mục gốc dự án

# --- Cấu hình Dữ liệu & Nhãn ---
TEXT_COLUMN = "comment" # Tên cột text mong đợi cho CSV upload
TARGET_LABEL_MAP = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
NUM_LABELS = len(TARGET_LABEL_MAP)

# --- Cấu hình Model Local ---
MODEL_NAME = "saved_model" # Tên thư mục chứa model đã tải về/đổi tên
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "saved_model")

# --- Cấu hình Thực thi ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 160
BATCH_SIZE = 16 # Dùng cho evaluate/predict trên local nếu cần

# --- Cấu hình API ---
API_HOST = os.getenv("API_HOST", "127.0.0.1") # Có thể ghi đè bằng .env
API_PORT = int(os.getenv("API_PORT", 8000)) # Có thể ghi đè bằng .env

# --- Cấu hình Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Không in cảnh báo ở đây, để api.py xử lý

# --- Cấu hình MySQL Knowledge Base ---
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD") # Đọc từ .env là quan trọng nhất
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sentiment_kb")
# Không in cảnh báo ở đây, để api.py xử lý

# --- Cấu hình Trực quan hóa (Cho Tab 3) ---
VISUALIZATION_DIR = os.path.join(PROJECT_DIR, "visualizations")
# Các tên file chuẩn sau khi đã đổi tên trên local
CONFUSION_MATRIX_FILE = os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")
TRAINING_CURVES_FILE = os.path.join(VISUALIZATION_DIR, "training_curves.png")
CLASSIFICATION_REPORT_FILE = os.path.join(VISUALIZATION_DIR, 'classification_report.txt')
EVALUATION_SUMMARY_FILE = os.path.join(VISUALIZATION_DIR, 'evaluation_summary.json')
ERROR_ANALYSIS_FILE = os.path.join(VISUALIZATION_DIR, 'error_analysis.csv')

# --- Các đường dẫn dữ liệu (Ít dùng trực tiếp, nhưng giữ lại) ---
DATA_SOURCES=[ # Giữ lại cấu trúc ví dụ
     {'path': os.path.join(PROJECT_DIR, 'data/raw/file1_en.csv'), 'language': 'en', 'text_col': 'Text', 'label_type': 'rating', 'label_col': 'Score', 'rating_scale': (1, 5)},
     {'path': os.path.join(PROJECT_DIR, 'data/raw/file2_en.csv'), 'language': 'en', 'text_col': 'review', 'label_type': 'text_label', 'label_col': 'sentiment', 'label_values': {'positive': 2, 'negative': 0, 'neutral': 1}},
     {'path': os.path.join(PROJECT_DIR, 'data/raw/file3_vi.csv'), 'language': 'vi', 'text_col': 'comment', 'label_type': 'rating', 'label_col': 'rate', 'rating_scale': (1, 5)},
     {'path': os.path.join(PROJECT_DIR, 'data/raw/file4_en.csv'), 'language': 'en', 'text_col': 'Review', 'label_type': 'rating', 'label_col': 'Rating', 'rating_scale': (1, 5)}, ]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed")
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train_combined_4files.csv")
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, "val_combined_4files.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test_combined_4files.csv")