import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
from tqdm import tqdm
from evaluate import compute_metrics
from config import config

MASK_TOKEN = 0
MASK_RATIO = 0.15
PATIENCE = config['pacience']

random.seed(42)

def mask_sequence(inputs, mask_token=MASK_TOKEN, mask_ratio=MASK_RATIO):
    labels = torch.full_like(inputs, -100)
    masked = inputs.clone()

    B, L = inputs.size()
    for i in range(B):
        for j in range(L):
            if inputs[i, j] == 0:
                continue
            if random.random() < mask_ratio:
                labels[i, j] = inputs[i, j]
                masked[i, j] = mask_token
    return masked, labels


def mask_last_item(inputs, mask_token=MASK_TOKEN):
    labels = torch.full_like(inputs, -100)
    masked = inputs.clone()

    B, L = inputs.size()
    for i in range(B):
        non_pad = (inputs[i] != 0).nonzero(as_tuple=False).squeeze()
        last = non_pad[-1].item()
        labels[i, last] = inputs[i, last]
        masked[i, last] = mask_token
    return masked, labels


def train_model(model, train_loader, val_loader,
                num_epochs=10, lr=1e-3, device='cpu'):
    os.makedirs('results', exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_ndcg, counter = -1, 0
    history = []
    model.to(device)

    for ep in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0
        for batch, in tqdm(train_loader, desc=f"Train Ep{ep}"):
            batch = batch.to(device)
            masked, labels = mask_sequence(batch)
            h = model(masked)
            logits = model.predict_masked(h, labels != -100)

            loss = criterion(logits, labels[labels != -100])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        tr_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_logits, all_lbls = [], []
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                masked, labels = mask_last_item(batch)
                h = model(masked)
                logits = model.predict_masked(h, labels != -100)

                val_loss += criterion(logits, labels[labels != -100]).item()

                # next-item prediction on last pos
                logits_next = model.predict_next(h)
                all_logits.append(logits_next)
                non_zero = (batch != 0).sum(dim=1) - 1
                gt = batch[torch.arange(batch.size(0)), non_zero]
                all_lbls.append(gt)
        vl_loss = val_loss / len(val_loader)

        # Concatenate and prepare mask_positions for evaluation
        logits_cat = torch.cat(all_logits, 0)
        lbls_cat = torch.cat(all_lbls, 0)
        mask_positions = torch.ones(lbls_cat.shape, dtype=torch.bool, device=lbls_cat.device)

        metrics = compute_metrics(logits_cat, lbls_cat, mask_positions, k=10)

        history.append({
            'epoch': ep,
            'train_loss': tr_loss,
            'val_loss': vl_loss,
            'val_recall': metrics['recall'],
            'val_ndcg': metrics['ndcg']
        })

        # Early stopping
        if metrics['ndcg'] > best_ndcg:
            best_ndcg = metrics['ndcg']
            torch.save(model.state_dict(), 'results/best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping")
                break
        
        print(f"Ep{ep} TR_L={tr_loss:.4f}, VL_L={vl_loss:.4f}, R10={metrics['recall']:.4f}, N10={metrics['ndcg']:.4f}, best_ndcg={best_ndcg:.4f}\n")
        scheduler.step()

    # Save history
    with open('results/model_performance.json', 'w') as f:
        json.dump(history, f, indent=2)
    return model, history