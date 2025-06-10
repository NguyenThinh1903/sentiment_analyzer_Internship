# evaluate.py (Bổ sung chi tiết đánh giá)
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import os
import time
import json
import traceback

import config
from data_processing import load_processed_data, create_data_loader
from model import load_model_and_tokenizer
from visualization import plot_confusion_matrix, plot_training_history

def evaluate_epoch(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            losses.append(loss.item())
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if total_samples == 0:
        epoch_accuracy = torch.tensor(0.0); epoch_loss = 0.0
    else:
        epoch_accuracy = correct_predictions.double() / total_samples
        epoch_loss = np.mean(losses) if losses else 0.0
    return epoch_accuracy, epoch_loss, all_preds, all_labels


def evaluate_model():
    """Tải model, đánh giá chi tiết và lưu kết quả."""
    print("--- Bắt đầu Quá trình Đánh giá Model Chi Tiết ---")
    print(f"Sử dụng thiết bị: {config.DEVICE}")

    # --- 1. Tải Dữ liệu Test ---
    train_df, val_df, test_df = load_processed_data()
    if test_df is None:
        print("Không thể tải dữ liệu test. Kết thúc đánh giá.")
        return

    # --- 2. Tải Model và Tokenizer ---
    print(f"Đang tải model từ {config.MODEL_SAVE_PATH}...")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_SAVE_PATH, config.NUM_LABELS)
    if model is None or tokenizer is None: return
    model.to(config.DEVICE)

    # --- 3. Tạo Test DataLoader ---
    print("Đang tạo Test DataLoader...")
    eval_batch_size = config.BATCH_SIZE 
    test_data_loader = create_data_loader(test_df, tokenizer, config.MAX_LENGTH, eval_batch_size, shuffle=False)
    if test_data_loader is None: return

    # --- 4. Thực hiện Đánh giá ---
    print("Đang đánh giá trên tập test...")
    start_time = time.time()
    results = {} 
    try:
         test_acc_tensor, test_loss, y_pred, y_true = evaluate_epoch(model, test_data_loader, config.DEVICE)
         test_acc = test_acc_tensor.item()
         results['test_accuracy'] = test_acc
         results['test_loss'] = test_loss
         results['predictions'] = y_pred
         results['true_labels'] = y_true
    except Exception as eval_err:
        print(f"Lỗi trong quá trình chạy evaluate_epoch: {eval_err}")
        traceback.print_exc()
        return
    end_time = time.time()
    results['evaluation_time_seconds'] = end_time - start_time
    print(f"Thời gian đánh giá: {results['evaluation_time_seconds']:.2f} giây")

    print("\n--- Kết quả Đánh giá Cơ bản ---")
    print(f"Loss trên tập Test: {results['test_loss']:.4f}")
    print(f"Độ chính xác trên tập Test: {results['test_accuracy']:.4f}")

    # --- 5. Tính toán và Lưu Metrics Chi Tiết ---
    try:
        target_names = list(config.TARGET_LABEL_MAP.values())
        labels_indices = list(config.TARGET_LABEL_MAP.keys())
    except AttributeError:
        print("Lỗi: Biến 'TARGET_LABEL_MAP' không được định nghĩa trong config.py!")
        return
    except Exception as e:
        print(f"Lỗi khi lấy target_names: {e}")
        return

    if not results['predictions'] or not results['true_labels']:
         print("Cảnh báo: Không có dự đoán/nhãn. Bỏ qua tính toán metrics chi tiết.")
    else:
        try:
            # Tính classification report dưới dạng dict để dễ truy cập
            report_dict = classification_report(
                results['true_labels'], results['predictions'],
                labels=labels_indices, target_names=target_names,
                output_dict=True, zero_division=0
            )
            results['classification_report_dict'] = report_dict
            results['classification_report_text'] = classification_report(
                 results['true_labels'], results['predictions'],
                 labels=labels_indices, target_names=target_names,
                 digits=4, zero_division=0
             )

            print("\nBáo cáo Phân loại:")
            print(results['classification_report_text'])

            # Trích xuất các metrics quan trọng khác
            precision, recall, f1, _ = precision_recall_fscore_support(
                results['true_labels'], results['predictions'],
                average='weighted', labels=labels_indices, zero_division=0
            )
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                results['true_labels'], results['predictions'],
                average='macro', labels=labels_indices, zero_division=0
            )
            results['weighted_precision'] = precision
            results['weighted_recall'] = recall
            results['weighted_f1'] = f1
            results['macro_precision'] = macro_precision
            results['macro_recall'] = macro_recall
            results['macro_f1'] = macro_f1

            print(f"Weighted Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            print(f"Macro Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1-Score: {macro_f1:.4f}")

            # Lưu classification report text
            report_path = config.CLASSIFICATION_REPORT_FILE
            os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("--- Kết quả Đánh giá Model ---\n")
                f.write(f"Loss trên tập Test: {results['test_loss']:.4f}\n")
                f.write(f"Độ chính xác trên tập Test: {results['test_accuracy']:.4f}\n")
                f.write(f"Weighted F1-Score: {results['weighted_f1']:.4f}\n")
                f.write(f"Macro F1-Score: {results['macro_f1']:.4f}\n\n")
                f.write("Báo cáo Phân loại Chi tiết:\n")
                f.write(results['classification_report_text'])
            print(f"Báo cáo phân loại chi tiết đã được lưu vào {report_path}")

        except Exception as e:
            print(f"Lỗi khi tính toán/lưu metrics chi tiết: {e}")
            traceback.print_exc()

        # --- 6. Phân tích Confusion Matrix và Lưu ---
        print("\nĐang tạo Ma trận Nhầm lẫn...")
        try:
             if results['true_labels'] and results['predictions']:
                 cm = confusion_matrix(results['true_labels'], results['predictions'], labels=labels_indices)
                 results['confusion_matrix'] = cm.tolist() 
                 plot_confusion_matrix(cm, class_names=target_names, save_path=config.CONFUSION_MATRIX_FILE)
                 print("Phân tích Ma trận Nhầm lẫn:")
                 for i, true_label_name in enumerate(target_names):
                      print(f"  Nhãn thực tế '{true_label_name}':")
                      for j, pred_label_name in enumerate(target_names):
                           if i == j:
                               print(f"    - Dự đoán đúng '{pred_label_name}': {cm[i, j]} mẫu")
                           elif cm[i, j] > 0:
                                print(f"    - Bị dự đoán nhầm thành '{pred_label_name}': {cm[i, j]} mẫu")
             else:
                 print("  Không đủ dữ liệu để tạo ma trận nhầm lẫn.")
        except Exception as e:
             print(f"Lỗi khi tạo/vẽ/phân tích ma trận nhầm lẫn: {e}")
             traceback.print_exc()

        # --- 7. Phân tích Lỗi (Error Analysis) - Lưu các mẫu bị sai ---
        print("\nĐang phân tích lỗi và lưu các mẫu dự đoán sai...")
        try:
            error_indices = [idx for idx, (true, pred) in enumerate(zip(results['true_labels'], results['predictions'])) if true != pred]
            if error_indices:
                 error_df = test_df.iloc[error_indices].copy()
                 error_df['predicted_label_index'] = [results['predictions'][i] for i in error_indices]
                 error_df['predicted_label_name'] = error_df['predicted_label_index'].map(config.TARGET_LABEL_MAP)
                 error_df['true_label_name'] = error_df['label'].map(config.TARGET_LABEL_MAP)
                 error_df_to_save = error_df[['cleaned_text', 'true_label_name', 'predicted_label_name']]

                 error_file_path = os.path.join(config.VISUALIZATION_DIR, 'error_analysis.csv')
                 error_df_to_save.to_csv(error_file_path, index=False, encoding='utf-8-sig')
                 print(f"Đã lưu {len(error_df_to_save)} mẫu dự đoán sai vào: {error_file_path}")
                 results['error_samples_count'] = len(error_df_to_save)
                 results['error_samples_examples'] = error_df_to_save.head(10).to_dict('records')
            else:
                 print("Không tìm thấy mẫu nào dự đoán sai.")
                 results['error_samples_count'] = 0
                 results['error_samples_examples'] = []

        except Exception as e:
             print(f"Lỗi trong quá trình phân tích lỗi: {e}")
             traceback.print_exc()


    # --- 8. Vẽ Biểu đồ Lịch sử Huấn luyện ---
    history_path = os.path.join(config.MODEL_SAVE_PATH, 'training_history.json')
    history_plot_path = config.TRAINING_CURVES_FILE
    if os.path.exists(history_path):
        print("\nĐang vẽ Biểu đồ Lịch sử Huấn luyện...")
        plot_training_history(history_path, save_path=history_plot_path)
    else:
        print(f"\nKhông tìm thấy file lịch sử huấn luyện tại {history_path}, bỏ qua.")

    # --- 9. Lưu tất cả kết quả đánh giá vào file JSON ---
    results_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'true_labels']}
    if 'confusion_matrix' in results_to_save:
        results_to_save['confusion_matrix'] = np.array(results_to_save['confusion_matrix']).tolist()

    evaluation_summary_path = os.path.join(config.VISUALIZATION_DIR, 'evaluation_summary.json')
    try:
        with open(evaluation_summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu tóm tắt kết quả đánh giá vào: {evaluation_summary_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file tóm tắt đánh giá: {e}")


    print("\n--- Đánh giá Chi Tiết Hoàn tất ---")


if __name__ == '__main__':
    evaluate_model()