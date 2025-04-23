import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
import time
import json # To save training history

import config
from data_processing import prepare_data, load_processed_data, create_data_loader
from model import load_model_and_tokenizer

def train_epoch(model, data_loader, optimizer, device, scheduler):
    """Performs one training epoch."""
    model = model.train()
    losses = []
    correct_predictions = 0
    total_samples = 0

    start_time = time.time()
    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

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

        loss.backward()
        # Clip gradients to prevent exploding gradients (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step() # Update learning rate schedule

        if (i + 1) % 50 == 0: # Print progress every 50 batches
             elapsed_time = time.time() - start_time
             print(f'  Batch {i+1}/{len(data_loader)} | Loss: {loss.item():.4f} | Time: {elapsed_time:.2f}s')
             start_time = time.time() # Reset timer for next block

    epoch_accuracy = correct_predictions.double() / total_samples
    epoch_loss = np.mean(losses)
    return epoch_accuracy, epoch_loss

def evaluate_epoch(model, data_loader, device):
    """Performs evaluation on a given dataset."""
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

    epoch_accuracy = correct_predictions.double() / total_samples
    epoch_loss = np.mean(losses)
    return epoch_accuracy, epoch_loss, all_preds, all_labels

def main():
    """Main training loop."""
    print("--- Starting Training Process ---")
    print(f"Using device: {config.DEVICE}")

    # --- 1. Prepare Data ---
    # Check if processed files exist, if not, create them
    if not all([os.path.exists(f) for f in [config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE]]):
        print("Processed data not found. Running data preparation...")
        if not prepare_data():
             print("Data preparation failed. Exiting training.")
             return
    else:
        print("Processed data found.")

    # Load processed data
    train_df, val_df, _ = load_processed_data() # We don't need test_df for training
    if train_df is None or val_df is None:
        print("Failed to load processed data. Exiting training.")
        return

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading model '{config.MODEL_NAME}'...")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.NUM_LABELS)
    if model is None or tokenizer is None:
        print("Failed to load model/tokenizer. Exiting.")
        return
    model.to(config.DEVICE)

    # --- 3. Create DataLoaders ---
    print("Creating DataLoaders...")
    train_data_loader = create_data_loader(train_df, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE, shuffle=True)
    val_data_loader = create_data_loader(val_df, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE, shuffle=False)

    if train_data_loader is None or val_data_loader is None:
        print("Failed to create DataLoaders. Exiting.")
        return

    # --- 4. Setup Optimizer and Scheduler ---
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_data_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # You can adjust warmup steps if needed
        num_training_steps=total_steps
    )

    # --- 5. Training Loop ---
    best_val_accuracy = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("--- Starting Training ---")
    total_start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 30)

        # Training step
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, config.DEVICE, scheduler)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # Validation step
        val_acc, val_loss, _, _ = evaluate_epoch(model, val_data_loader, config.DEVICE)
        print(f"Val   Loss: {val_loss:.4f} | Val   Accuracy: {val_acc:.4f}")

        epoch_end_time = time.time()
        print(f"Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc.item()) # convert tensor to float
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc.item()) # convert tensor to float

        # Save the best model based on validation accuracy
        if val_acc > best_val_accuracy:
            print(f"Validation accuracy improved ({best_val_accuracy:.4f} --> {val_acc:.4f}). Saving model...")
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(config.MODEL_SAVE_PATH)
            tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
            best_val_accuracy = val_acc
            # Save training history alongside the best model
            history_path = os.path.join(config.MODEL_SAVE_PATH, 'training_history.json')
            with open(history_path, 'w') as f:
                 json.dump(training_history, f, indent=4)
            print(f"Model and training history saved to {config.MODEL_SAVE_PATH}")


    total_end_time = time.time()
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {(total_end_time - total_start_time) / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()