# evaluate.py (Đã Việt hóa thông báo và report)

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import time

import config
from data_processing import load_processed_data, create_data_loader
from model import load_model_and_tokenizer
from visualization import plot_confusion_matrix, plot_training_history

# Sao chép hoặc import hàm evaluate_epoch từ train.py
# (Đảm bảo hàm này tồn tại và đúng)
def evaluate_epoch(model, data_loader, device):
    """Thực hiện đánh giá trên một tập dữ liệu."""
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total_samples == 0:
        print("Cảnh báo: Không tìm thấy mẫu nào trong dataloader để đánh giá.")
        epoch_accuracy = torch.tensor(0.0)
        epoch_loss = 0.0
    else:
        epoch_accuracy = correct_predictions.double() / total_samples
        epoch_loss = np.mean(losses) if losses else 0.0

    return epoch_accuracy, epoch_loss, all_preds, all_labels

def evaluate_model():
    """Tải model đã huấn luyện và đánh giá trên tập test."""
    print("--- Bắt đầu Quá trình Đánh giá Model ---")
    print(f"Sử dụng thiết bị: {config.DEVICE}")

    # --- 1. Tải Dữ liệu Test ---
    _, _, test_df = load_processed_data()
    if test_df is None:
        print("Không thể tải dữ liệu test. Kết thúc đánh giá.")
        return

    # --- 2. Tải Model và Tokenizer đã Huấn luyện ---
    print(f"Đang tải model từ {config.MODEL_SAVE_PATH}...")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_SAVE_PATH, config.NUM_LABELS)
    if model is None or tokenizer is None:
        print("Không thể tải model/tokenizer đã lưu. Hãy đảm bảo quá trình huấn luyện thành công.")
        print(f"Đường dẫn mong đợi: {config.MODEL_SAVE_PATH}")
        return
    model.to(config.DEVICE)

    # --- 3. Tạo Test DataLoader ---
    print("Đang tạo Test DataLoader...")
    test_data_loader = create_data_loader(test_df, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE, shuffle=False)
    if test_data_loader is None:
        print("Không thể tạo test DataLoader. Kết thúc.")
        return

    # --- 4. Thực hiện Đánh giá ---
    print("Đang đánh giá trên tập test...")
    start_time = time.time()
    test_acc_tensor, test_loss, y_pred, y_true = evaluate_epoch(model, test_data_loader, config.DEVICE)
    test_acc = test_acc_tensor.item()
    end_time = time.time()
    print(f"Thời gian đánh giá: {end_time - start_time:.2f} giây")

    print("\n--- Kết quả Đánh giá ---")
    print(f"Loss trên tập Test: {test_loss:.4f}")
    print(f"Độ chính xác trên tập Test: {test_acc:.4f}")

    # --- 5. Báo cáo Phân loại Chi tiết ---
    target_names = list(config.LABEL_MAP.values()) # Lấy tên nhãn tiếng Việt

    if not y_pred or not y_true:
         print("Cảnh báo: Không có dự đoán hoặc nhãn thực tế nào được tạo ra. Bỏ qua báo cáo phân loại và ma trận nhầm lẫn.")
    else:
        try:
            report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
            print("\nBáo cáo Phân loại:") # <--- Việt hóa
            print(report)

            # Lưu báo cáo vào file
            report_path = config.CLASSIFICATION_REPORT_FILE # Sử dụng biến từ config
            os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f: # Thêm encoding='utf-8'
                f.write("--- Kết quả Đánh giá Model ---\n") # <--- Việt hóa
                f.write(f"Loss trên tập Test: {test_loss:.4f}\n")
                f.write(f"Độ chính xác trên tập Test: {test_acc:.4f}\n\n") # <--- Việt hóa
                f.write("Báo cáo Phân loại:\n") # <--- Việt hóa
                f.write(report)
            print(f"Báo cáo phân loại đã được lưu vào {report_path}")

        except Exception as e:
            print(f"Lỗi khi tạo/lưu báo cáo phân loại: {e}")
            print("Hãy đảm bảo các dự đoán và nhãn thực tế có định dạng đúng.")

        # --- 6. Ma trận Nhầm lẫn ---
        print("\nĐang tạo Ma trận Nhầm lẫn...") # <--- Việt hóa
        try:
             cm = confusion_matrix(y_true, y_pred)
             # Hàm plot_confusion_matrix đã được Việt hóa title/labels
             plot_confusion_matrix(cm, class_names=target_names, save_path=config.CONFUSION_MATRIX_FILE)
        except Exception as e:
             print(f"Lỗi khi tạo/vẽ ma trận nhầm lẫn: {e}")

    # --- 7. Vẽ Biểu đồ Lịch sử Huấn luyện (Tùy chọn) ---
    history_path = os.path.join(config.MODEL_SAVE_PATH, 'training_history.json')
    history_plot_path = config.TRAINING_CURVES_FILE # Sử dụng biến từ config
    if os.path.exists(history_path):
        print("\nĐang vẽ Biểu đồ Lịch sử Huấn luyện...") # <--- Việt hóa
        # Hàm plot_training_history đã được Việt hóa title/labels
        plot_training_history(history_path, save_path=history_plot_path)
    else:
        print(f"\nKhông tìm thấy file lịch sử huấn luyện tại {history_path}, bỏ qua việc vẽ biểu đồ.")

    print("\n--- Đánh giá Hoàn tất ---") # <--- Việt hóa

if __name__ == '__main__':
    evaluate_model()