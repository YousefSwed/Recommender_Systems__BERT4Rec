# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from config import MASK_PROB, EPOCHS, PATIENCE, LR, MODEL_SAVE_PATH
from evaluate import evaluate_model
import json
import os

random.seed(42)

def mask_items(seqs, num_items, mask_prob):
    MASK_ID = num_items + 1
    masked_seqs, labels = [], []

    for seq in seqs:
        masked, label = [], []
        for item in seq:
            if item != 0 and random.random() < mask_prob:
                masked.append(MASK_ID)
                label.append(item)
            else:
                masked.append(item)
                label.append(0)
        masked_seqs.append(masked)
        labels.append(label)

    return torch.LongTensor(masked_seqs), torch.LongTensor(labels)

def train_model(model, train_data, val_data, num_items, device, mask_prob=MASK_PROB):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,verbose=True, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)

    history = []
    best_ndcg = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        random.shuffle(train_data)

        loop = tqdm(range(0, len(train_data), 128), desc=f"Epoch {epoch}")
        for i in loop:
            batch = train_data[i:i+128]
            masked_inputs, labels = mask_items(batch, num_items, mask_prob)
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)
            mask = (masked_inputs == 0)
            logits = model(masked_inputs, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            train_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.detach().item())

        # Evaluate on validation set
        val_loss = evaluate_val_loss(model, val_data, criterion, num_items, device, mask_prob)
        val_metrics = evaluate_model(model, val_data, num_items, device, k_values=[10])
        val_ndcg = val_metrics['ndcg'][10]
        val_recall = val_metrics['recall'][10]

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_data),
            "val_loss": val_loss,
            "val_ndcg": val_ndcg,
            "val_recall": val_recall
        }
        history.append(log_entry)

        print(f"Epoch {epoch} | Train Loss: {log_entry['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | NDCG@10: {val_ndcg:.4f} | Recall@10: {val_recall:.4f}\n")

        # Early stopping based on NDCG@10
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered by NDCG@10.")
                break

        scheduler.step(val_ndcg)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open('results/model_performance.json', 'w') as f:
        json.dump(history, f, indent=2)

def evaluate_val_loss(model, val_data, criterion, num_items, device, mask_prob):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(range(0, len(val_data), 128), desc="Validating", leave=False)
        for i in loop:
            batch = val_data[i:i+128]
            masked_inputs, labels = mask_items(batch, num_items, mask_prob)
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)
            mask = (masked_inputs == 0)
            logits = model(masked_inputs, mask)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(val_data)
