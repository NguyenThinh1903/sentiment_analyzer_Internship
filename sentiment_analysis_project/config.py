import torch
import os

# --- Đường dẫn Cơ sở trên Google Drive ---
# !!! Thay đổi 'Colab_Sentiment_Project' nếu bạn đặt tên thư mục khác !!!
DRIVE_MOUNT_POINT = '/content/drive/MyDrive/'
PROJECT_DIR = os.path.join(DRIVE_MOUNT_POINT, 'Colab_Sentiment_Project')

# --- Cấu hình Nguồn Dữ liệu ---
DATA_SOURCES = [
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file1_en.csv'), # <-- Sửa đường dẫn
        'language': 'en', 'text_col': 'Text', 'label_type': 'rating',
        'label_col': 'Score', 'rating_scale': (1, 5)
    },
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file2_en.csv'), # <-- Sửa đường dẫn
        'language': 'en', 'text_col': 'review', 'label_type': 'text_label',
        'label_col': 'sentiment', 'label_values': {'positive': 2, 'negative': 0, 'neutral': 1}
    },
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file3_vi.csv'), # <-- Sửa đường dẫn
        'language': 'vi', 'text_col': 'comment', 'label_type': 'rating',
        'label_col': 'rate', 'rating_scale': (1, 5)
     },
    {
        'path': os.path.join(PROJECT_DIR, 'data/raw/file4_en.csv'), # <-- Sửa đường dẫn
        'language': 'en', 'text_col': 'Review', 'label_type': 'rating',
        'label_col': 'Rating', 'rating_scale': (1, 5)
     },
]

PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed") # <-- Sửa đường dẫn
COMBINED_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "combined_processed_4files.csv")
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train_combined_4files.csv")
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, "val_combined_4files.csv")
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test_combined_4files.csv")

# --- Ánh xạ Nhãn Mục tiêu ---
TARGET_LABEL_MAP = { 0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực" }
NUM_LABELS = len(TARGET_LABEL_MAP)

# --- Cấu hình Model ---
# MODEL_NAME = "xlm-roberta-base"
MODEL_NAME = os.path.join(PROJECT_DIR, "saved_model_xlmr_4files")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "saved_model_xlmr_4files") # <-- Sửa đường dẫn

# --- Cấu hình Huấn luyện ---
# DEVICE sẽ tự động là 'cuda' trên Colab nếu chọn GPU runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 160
# !!! Có thể tăng BATCH_SIZE trên Colab GPU mạnh hơn !!!
# Thử bắt đầu với 32 hoặc 64, nếu OOM thì giảm xuống
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1 # Giữ nguyên hoặc tăng nếu muốn huấn luyện kỹ hơn
GRADIENT_ACCUMULATION_STEPS = 2 # Giảm xuống vì batch size vật lý lớn hơn
TEST_SPLIT_SIZE = 0.15
VALIDATION_SPLIT_SIZE = 0.1

# --- Cấu hình Trực quan hóa ---
VISUALIZATION_DIR = os.path.join(PROJECT_DIR, "visualizations_xlmr_4files") # <-- Sửa đường dẫn
CONFUSION_MATRIX_FILE = os.path.join(VISUALIZATION_DIR, "confusion_matrix_xlmr.png")
TRAINING_CURVES_FILE = os.path.join(VISUALIZATION_DIR, "training_curves_xlmr.png")
CLASSIFICATION_REPORT_FILE = os.path.join(VISUALIZATION_DIR, 'classification_report_xlmr.txt')

# --- API Configuration (Không dùng trực tiếp trong Colab training) ---
API_HOST = "127.0.0.1"
API_PORT = 8000