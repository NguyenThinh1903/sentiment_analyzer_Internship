# data_processing.py (Phiên bản cập nhật cho nhiều dataset)
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import traceback 

import config 

# --- Text Cleaning ---
def clean_text(text):
    """Làm sạch văn bản cơ bản: chữ thường, xóa URL, HTML, ký tự đặc biệt không cần thiết."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s.,!?'\"-]", "", text, flags=re.UNICODE) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Hàm chuẩn hóa Nhãn ---
def standardize_label(label_value, label_type, rating_scale=None, label_values_map=None):
    """Chuyển đổi giá trị nhãn/rating gốc thành nhãn chuẩn 0, 1, 2."""
    if label_type == 'rating':
        try:
            rating = int(float(label_value))
            if rating_scale == (1, 5): 
                 if rating in [1, 2]: return 0 
                 elif rating == 3: return 1 
                 elif rating in [4, 5]: return 2 
                 else: return 1 
            else:
                 print(f"Cảnh báo: Thang điểm {rating_scale} chưa được xử lý, trả về Trung tính.")
                 return 1
        except (ValueError, TypeError, TypeError):
            return 1 

    elif label_type == 'text_label':
        label_str = str(label_value).lower().strip()
        return label_values_map.get(label_str, 1)

    else:
        print(f"Cảnh báo: label_type không xác định '{label_type}', trả về Trung tính.")
        return 1 # Mặc định Trung tính

# --- Hàm Tải và Gộp Dữ liệu ---
def load_and_combine_datasets():
    """Tải dữ liệu từ nhiều nguồn, làm sạch, chuẩn hóa nhãn và gộp lại."""
    all_dataframes = []
    print("--- Bắt đầu tải và gộp dữ liệu từ các nguồn ---")

    for source in config.DATA_SOURCES:
        print(f"Đang xử lý: {source['path']} ({source['language']})")
        try:
            try:
                df = pd.read_csv(source['path'], encoding='utf-8-sig', low_memory=False)
            except UnicodeDecodeError:
                 print("  Không đọc được bằng utf-8-sig, thử utf-8...")
                 try:
                    df = pd.read_csv(source['path'], encoding='utf-8', low_memory=False)
                 except UnicodeDecodeError:
                     print("  Không đọc được bằng utf-8, thử latin-1...")
                     df = pd.read_csv(source['path'], encoding='latin-1', low_memory=False)
                 except Exception as e:
                     print(f" Lỗi khi đọc bằng utf-8: {e}")
                     continue 
            except Exception as e:
                 print(f" Lỗi khi đọc bằng utf-8-sig: {e}")
                 continue 


            print(f"  Đã tải {len(df)} dòng.")
            # Kiểm tra sự tồn tại của các cột cần thiết
            if source['text_col'] not in df.columns:
                print(f"  Lỗi: Không tìm thấy cột text '{source['text_col']}'. Bỏ qua file này.")
                continue
            if source['label_col'] not in df.columns:
                 print(f"  Lỗi: Không tìm thấy cột label/rating '{source['label_col']}'. Bỏ qua file này.")
                 continue

            # 1. Chọn cột và đổi tên thành chuẩn ('text', 'original_label')
            df_processed = df[[source['text_col'], source['label_col']]].copy()
            df_processed.rename(columns={
                source['text_col']: 'text',
                source['label_col']: 'original_label'
            }, inplace=True)

            # 2. Loại bỏ NaN trước khi xử lý
            df_processed.dropna(subset=['text', 'original_label'], inplace=True)
            if df_processed.empty:
                 print("  Không còn dữ liệu sau khi loại bỏ NaN. Bỏ qua.")
                 continue

            # 3. Làm sạch text
            df_processed['cleaned_text'] = df_processed['text'].apply(clean_text)

            # 4. Chuẩn hóa nhãn
            rating_scale = source.get('rating_scale')
            label_map = source.get('label_values')
            df_processed['label'] = df_processed['original_label'].apply(
                lambda x: standardize_label(x, source['label_type'], rating_scale, label_map)
            )

            # 5. Thêm cột ngôn ngữ
            df_processed['language'] = source['language']

            # 6. Chỉ giữ các cột cần thiết và loại bỏ dòng có text rỗng sau khi clean
            df_processed = df_processed[['cleaned_text', 'label', 'language']].copy()
            df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]


            if not df_processed.empty:
                all_dataframes.append(df_processed)
                print(f"  Đã xử lý xong. Thêm {len(df_processed)} mẫu hợp lệ.")
            else:
                print("  Không có mẫu hợp lệ nào sau khi xử lý.")

        except FileNotFoundError:
            print(f"  Lỗi: Không tìm thấy file tại {source['path']}")
        except pd.errors.ParserError as pe:
             print(f"  Lỗi phân tích cú pháp CSV: {pe}. File có thể bị lỗi cấu trúc. Bỏ qua file.")
        except Exception as e:
            print(f"  Đã xảy ra lỗi không mong muốn khi xử lý file {source['path']}: {e}")
            traceback.print_exc()

    if not all_dataframes:
        print("Lỗi nghiêm trọng: Không có dữ liệu nào được tải và xử lý thành công.")
        return None

    # Gộp tất cả DataFrame lại
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nTổng số mẫu dữ liệu đã gộp (trước khi xóa trùng): {len(combined_df)}")

    # Xử lý trùng lặp (dựa trên cleaned_text) - Quan trọng sau khi gộp
    initial_len = len(combined_df)
    combined_df.drop_duplicates(subset=['cleaned_text'], inplace=True, keep='first')
    print(f"Tổng số mẫu dữ liệu sau khi loại bỏ trùng lặp: {len(combined_df)} (Đã loại bỏ {initial_len - len(combined_df)} mẫu)")

    # Kiểm tra phân phối nhãn sau khi gộp và xóa trùng
    print("\nPhân phối nhãn trong dữ liệu cuối cùng:")
    print(combined_df['label'].value_counts(normalize=True))

    # Kiểm tra lại xem còn NaN không (ít khả năng nhưng để chắc chắn)
    if combined_df['cleaned_text'].isnull().any() or combined_df['label'].isnull().any():
        print("Cảnh báo: Vẫn còn giá trị NaN trong dữ liệu cuối cùng!")
        combined_df.dropna(subset=['cleaned_text', 'label'], inplace=True)
        print(f"Số mẫu sau khi loại bỏ NaN cuối cùng: {len(combined_df)}")

    print("--- Hoàn thành tải và gộp dữ liệu ---")
    return combined_df

# --- Data Splitting (Giữ nguyên logic, chỉ thay đổi tên file input/output qua config) ---
def split_data(df, test_size, val_size, random_state=42):
    """Chia DataFrame đã gộp thành train, validation, test."""
    if df is None or df.empty:
        print("Lỗi: Không thể chia DataFrame rỗng hoặc None.")
        return None, None, None

    # Đảm bảo cột label là kiểu số nguyên để stratify hoạt động đúng
    df['label'] = df['label'].astype(int)

    try:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label'] 
        )
    except ValueError as e:
         print(f"Lỗi khi chia dữ liệu (stratify): {e}")
         print("Kiểm tra lại phân phối nhãn. Có thể một số lớp có quá ít mẫu.")
         print("Thử chia không dùng stratify (kết quả có thể bị lệch):")
         train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)


    # Kiểm tra lại stratify cho lần chia thứ 2
    try:
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=train_val_df['label']
        )
    except ValueError as e:
         print(f"Lỗi khi chia dữ liệu lần 2 (stratify): {e}")
         print("Thử chia không dùng stratify:")
         train_df, val_df = train_test_split(train_val_df, test_size=relative_val_size, random_state=random_state)


    print(f"Chia dữ liệu: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# --- PyTorch Dataset Class  ---
class SentimentDataset(Dataset):
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
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, 
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- DataLoader Creation  ---
def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False):
    if df is None or df.empty:
        print("Cảnh báo: DataFrame rỗng, không thể tạo DataLoader.")
        return None
    if 'cleaned_text' not in df.columns or 'label' not in df.columns:
        print("Lỗi: DataFrame thiếu cột 'cleaned_text' hoặc 'label'.")
        return None
    try:
        df['label'] = df['label'].astype(int)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi cột label sang int: {e}")

    dataset = SentimentDataset(
        texts=df.cleaned_text.values, 
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

# --- Hàm Chuẩn bị Dữ liệu Chính (Giữ nguyên logic, dùng config mới) ---
def prepare_combined_data():
    """Tải, gộp, xử lý, chia dữ liệu và lưu các file đã xử lý."""
    print("--- Bắt đầu Chuẩn bị Dữ liệu Gộp ---")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Tải và gộp
    combined_df = load_and_combine_datasets()
    if combined_df is None:
        print("Chuẩn bị dữ liệu thất bại do không thể tải/gộp dữ liệu.")
        return False

    # (Tùy chọn) Lưu file gộp đã xử lý lại
    try:
        combined_df.to_csv(config.COMBINED_PROCESSED_FILE, index=False, encoding='utf-8-sig')
        print(f"Đã lưu dữ liệu gộp đã xử lý vào: {config.COMBINED_PROCESSED_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu file dữ liệu gộp: {e}")

    # 2. Chia dữ liệu
    train_df, val_df, test_df = split_data(
        combined_df,
        config.TEST_SPLIT_SIZE,
        config.VALIDATION_SPLIT_SIZE
    )
    if train_df is None or val_df is None or test_df is None:
         print("Chuẩn bị dữ liệu thất bại do lỗi chia dữ liệu.")
         return False

    # 3. Lưu các file train/val/test đã chia
    try:
        train_df.to_csv(config.TRAIN_FILE, index=False, encoding='utf-8-sig')
        val_df.to_csv(config.VAL_FILE, index=False, encoding='utf-8-sig')
        test_df.to_csv(config.TEST_FILE, index=False, encoding='utf-8-sig')
        print(f"Đã lưu các file train/val/test đã xử lý vào {config.PROCESSED_DATA_DIR}")
        print("--- Chuẩn bị Dữ liệu Gộp Hoàn tất ---")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu các file train/val/test: {e}")
        return False

# --- Hàm tải dữ liệu đã xử lý (Cập nhật để dùng tên file mới) ---
def load_processed_data():
    """Tải train, validation, test DataFrames từ các file đã xử lý (gộp)."""
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        val_df = pd.read_csv(config.VAL_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        print("Đã tải các tập train, validation, test đã xử lý (từ dữ liệu gộp).")
        train_df['label'] = train_df['label'].astype(int)
        val_df['label'] = val_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)
        return train_df, val_df, test_df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy các file dữ liệu đã xử lý trong {config.PROCESSED_DATA_DIR}")
        print(f"Kiểm tra các file: {config.TRAIN_FILE}, {config.VAL_FILE}, {config.TEST_FILE}")
        print("Vui lòng chạy bước chuẩn bị dữ liệu trước (vd: chạy train.py hoặc data_processing.py).")
        return None, None, None
    except KeyError as e:
        print(f"Lỗi KeyError khi đọc file CSV đã xử lý: {e}. File có thể bị lỗi.")
        return None, None, None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải dữ liệu đã xử lý: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Chạy file này trực tiếp để chuẩn bị dữ liệu gộp
    prepare_combined_data()