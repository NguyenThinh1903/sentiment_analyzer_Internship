# train.py (Cập nhật train_epoch để dùng Gradient Accumulation)
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import time
import json

import config
from data_processing import prepare_combined_data, load_processed_data, create_data_loader
from model import load_model_and_tokenizer

def evaluate_epoch(model, data_loader, device):
    """Thực hiện đánh giá trên tập validation."""
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_samples = 0
    print("  Đang đánh giá trên tập Validation...")
    eval_start_time = time.time()
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
    eval_end_time = time.time()
    print(f"  Đánh giá Validation hoàn thành sau: {eval_end_time - eval_start_time:.2f} giây")
    epoch_accuracy = correct_predictions.double() / total_samples if total_samples > 0 else 0.0
    epoch_loss = np.mean(losses) if losses else 0.0
    return epoch_accuracy, epoch_loss


def train_epoch(model, data_loader, optimizer, device, scheduler, epoch_num, total_epochs, accumulation_steps):
    """Thực hiện một epoch huấn luyện với Gradient Accumulation."""
    model = model.train()
    total_loss = 0.0 
    correct_predictions = 0
    total_samples = 0
    num_batches = len(data_loader)
    optimizer.zero_grad()

    epoch_start_time = time.time()
    batch_start_time = time.time()

    print(f"\n--- Epoch {epoch_num}/{total_epochs} ---")

    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        with torch.no_grad():
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

        # --- Gradient Accumulation Logic ---
        # 1. Chuẩn hóa loss
        loss = loss / accumulation_steps
        total_loss += loss.item() * accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()



     
        batches_per_log_update = max(1, (num_batches // accumulation_steps) // 20) 
        current_update_step = (i + 1) // accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
            if current_update_step % batches_per_log_update == 0 or (i + 1) == num_batches:
                batch_end_time = time.time()
                elapsed_block_time = batch_end_time - batch_start_time
                time_per_update = elapsed_block_time / batches_per_log_update if current_update_step % batches_per_log_update == 0 else elapsed_block_time
                progress = (i + 1) / num_batches
                current_lr = scheduler.get_last_lr()[0]

                avg_loss_since_last_log = (total_loss * accumulation_steps) / (i + 1)

                print(f"  Step {(i+1):>5}/{num_batches} [{progress:>6.1%}] Updt {current_update_step:>5} | "
                      f"Avg Loss: {avg_loss_since_last_log:.4f} | " 
                      f"LR: {current_lr:.2e} | "
                      f"Time/Updt: {time_per_update:.3f}s")
                batch_start_time = time.time() 

    epoch_end_time = time.time()
    epoch_accuracy = correct_predictions.double() / total_samples if total_samples > 0 else 0.0
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch_num} Hoàn thành sau: {epoch_end_time - epoch_start_time:.2f} giây")
    return epoch_accuracy, epoch_loss

def main():
    """Vòng lặp huấn luyện chính."""
    print("--- Bắt đầu Quá trình Huấn luyện ---")
    print(f"Sử dụng thiết bị: {config.DEVICE}")

    # --- 1. Chuẩn bị Dữ liệu ---
    if not all([os.path.exists(f) for f in [config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE]]):
        print("Không tìm thấy dữ liệu đã xử lý. Đang chạy chuẩn bị dữ liệu gộp...")
        if not prepare_combined_data():
             print("Chuẩn bị dữ liệu thất bại. Kết thúc huấn luyện.")
             return
    else:
        print("Đã tìm thấy dữ liệu đã xử lý.")

    train_df, val_df, _ = load_processed_data()
    if train_df is None or val_df is None:
        print("Không thể tải dữ liệu đã xử lý. Kết thúc huấn luyện.")
        return

    # --- 2. Tải Model và Tokenizer ---
    print(f"Đang tải model '{config.MODEL_NAME}'...")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.NUM_LABELS)
    if model is None or tokenizer is None:
        print("Không thể tải model/tokenizer. Kết thúc.")
        return
    model.to(config.DEVICE)

    # --- 3. Tạo DataLoaders ---
    print("Đang tạo DataLoaders...")
    train_data_loader = create_data_loader(train_df, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE, shuffle=True)
    val_data_loader = create_data_loader(val_df, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE, shuffle=False)

    if train_data_loader is None or val_data_loader is None:
        print("Không thể tạo DataLoaders. Kết thúc.")
        return
    print(f"Số batch vật lý mỗi epoch: {len(train_data_loader)}")
    print(f"Số bước tích lũy gradient: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Số bước cập nhật trọng số mỗi epoch: {len(train_data_loader) // config.GRADIENT_ACCUMULATION_STEPS}")


    # --- 4. Thiết lập Optimizer và Scheduler ---
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE, eps=1e-8)

    # Tổng số bước cập nhật = (tổng số batch / số bước tích lũy) * số epoch
    total_update_steps = (len(train_data_loader) // config.GRADIENT_ACCUMULATION_STEPS) * config.NUM_EPOCHS
    num_warmup_steps = int(total_update_steps * 0.05)
    print(f"Tổng số bước cập nhật trọng số: {total_update_steps}, Số bước warmup: {num_warmup_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_update_steps 
    )

    # --- 5. Vòng lặp Huấn luyện ---
    best_val_accuracy = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs_no_improve = 0
    patience = 2

    print("--- Bắt đầu Huấn luyện ---")
    total_start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, config.DEVICE, scheduler,
            epoch + 1, config.NUM_EPOCHS, config.GRADIENT_ACCUMULATION_STEPS
        )
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # Validation
        val_acc, val_loss = evaluate_epoch(model, val_data_loader, config.DEVICE)
        print(f"Epoch {epoch + 1} - Val   Loss: {val_loss:.4f} | Val   Accuracy: {val_acc:.4f}")

        # Lưu lịch sử
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc.item())
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc.item())

        # Lưu model tốt nhất
        if val_acc > best_val_accuracy:
            print(f"  Validation accuracy cải thiện ({best_val_accuracy:.4f} --> {val_acc:.4f}). Đang lưu model...")
            best_val_accuracy = val_acc
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(config.MODEL_SAVE_PATH)
            tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
            history_path = os.path.join(config.MODEL_SAVE_PATH, 'training_history.json')
            try:
                with open(history_path, 'w') as f:
                     json.dump(training_history, f, indent=4)
                print(f"  Đã lưu model và lịch sử huấn luyện vào {config.MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"  Lỗi khi lưu file lịch sử huấn luyện: {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation accuracy không cải thiện ({val_acc:.4f}). ({epochs_no_improve}/{patience})")
    total_end_time = time.time()
    print("\n--- Huấn luyện Hoàn tất ---")
    print(f"Tổng thời gian huấn luyện: {(total_end_time - total_start_time) / 60:.2f} phút")
    print(f"Validation Accuracy tốt nhất đạt được: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()