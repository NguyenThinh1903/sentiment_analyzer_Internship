import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

import config # Import cấu hình

# --- Text Cleaning ---
def clean_text(text):
    """Basic text cleaning: lowercase, remove URLs, HTML tags, special chars."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    # Remove characters that are not alphanumeric or basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?'\"]", "", text)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(raw_path, text_col, rating_col, clean_func, map_func):
    """Loads data, applies cleaning and label mapping."""
    try:
        df = pd.read_csv(raw_path)
        print(f"Loaded {len(df)} rows from {raw_path}")
        # Lấy các cột cần thiết và loại bỏ NaN trong các cột đó
        df = df[[text_col, rating_col]].dropna()
        print(f"Processing {len(df)} rows after dropping NaN.")
        df['cleaned_text'] = df[text_col].apply(clean_func)
        df['label'] = df[rating_col].apply(map_func)
        # Chỉ giữ lại các cột cần thiết
        df = df[['cleaned_text', 'label']]
        df = df.dropna(subset=['cleaned_text']) # Đảm bảo text không rỗng sau khi clean
        df = df[df['cleaned_text'].str.len() > 0] # Đảm bảo text không rỗng
        print(f"Finished preprocessing. {len(df)} rows remaining.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {raw_path}")
        return None
    except KeyError as e:
        print(f"Error: Column '{e}' not found in the CSV. Check config.py (TEXT_COLUMN, RATING_COLUMN).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preprocessing: {e}")
        return None

# --- Data Splitting ---
def split_data(df, test_size, val_size, random_state=42):
    """Splits DataFrame into train, validation, and test sets."""
    if df is None or df.empty:
        print("Error: Cannot split empty or None DataFrame.")
        return None, None, None

    # Tách tập test trước
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label'] # Giữ tỷ lệ nhãn trong tập test
    )

    # Tính tỷ lệ validation trên phần còn lại (train_val_df)
    relative_val_size = val_size / (1.0 - test_size)

    # Tách tập train và validation từ train_val_df
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df['label'] # Giữ tỷ lệ nhãn
    )

    print(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# --- PyTorch Dataset Class ---
class SentimentDataset(Dataset):
    """Custom PyTorch Dataset for sentiment analysis."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text, # Giữ lại text gốc có thể hữu ích
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- DataLoader Creation ---
def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False):
    """Creates a DataLoader for a given DataFrame."""
    if df is None:
        print("Warning: DataFrame is None, cannot create DataLoader.")
        return None
    if 'cleaned_text' not in df.columns or 'label' not in df.columns:
        print("Error: DataFrame missing required columns 'cleaned_text' or 'label'.")
        return None

    dataset = SentimentDataset(
        texts=df.cleaned_text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=os.cpu_count() // 2 # Điều chỉnh nếu cần
    )

# --- Main Data Preparation Function ---
def prepare_data():
    """Loads, preprocesses, splits data, and saves processed files."""
    print("--- Starting Data Preparation ---")

    # Tạo thư mục processed nếu chưa có
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Load and preprocess
    df = load_and_preprocess_data(
        config.RAW_DATA_PATH,
        config.TEXT_COLUMN,
        config.RATING_COLUMN,
        clean_text,
        config.map_rating_to_label
    )

    if df is None:
        print("Exiting data preparation due to errors.")
        return False # Báo hiệu lỗi

    # 2. Split data
    train_df, val_df, test_df = split_data(
        df,
        config.TEST_SPLIT_SIZE,
        config.VALIDATION_SPLIT_SIZE
    )

    if train_df is None or val_df is None or test_df is None:
         print("Exiting data preparation due to splitting errors.")
         return False # Báo hiệu lỗi

    # 3. Save processed files
    try:
        train_df.to_csv(config.TRAIN_FILE, index=False)
        val_df.to_csv(config.VAL_FILE, index=False)
        test_df.to_csv(config.TEST_FILE, index=False)
        print(f"Processed data saved to {config.PROCESSED_DATA_DIR}")
        print("--- Data Preparation Finished ---")
        return True # Báo hiệu thành công
    except Exception as e:
        print(f"Error saving processed data files: {e}")
        return False # Báo hiệu lỗi

# --- Hàm tiện ích để load các tập dữ liệu đã xử lý ---
def load_processed_data():
    """Loads train, validation, and test DataFrames from processed files."""
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        val_df = pd.read_csv(config.VAL_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        print("Loaded processed train, validation, and test sets.")
        return train_df, val_df, test_df
    except FileNotFoundError:
        print(f"Error: Processed data files not found in {config.PROCESSED_DATA_DIR}.")
        print("Please run the data preparation step first (e.g., by running train.py).")
        return None, None, None
    except Exception as e:
        print(f"An error occurred loading processed data: {e}")
        return None, None, None


if __name__ == "__main__":
    # Chạy file này trực tiếp để chuẩn bị dữ liệu
    prepare_data()