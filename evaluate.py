# evaluate.py

import torch
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

def evaluate_model(model, test_data, num_items, device, k_values=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
    model.eval()
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}

    batch_size = 512  # Use a large batch size to leverage GPU

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data[i:i + batch_size]
            input_seqs = torch.LongTensor(batch).to(device)
            mask = (input_seqs == 0)
            outputs = model(input_seqs, mask)  # shape: [batch_size, seq_len, num_items]
            last_outputs = outputs[:, -1, :]   # shape: [batch_size, num_items]

            topk = torch.topk(last_outputs, k=max(k_values), dim=-1).indices  # GPU operation

            for idx, seq in enumerate(batch):
                target_item = seq[-1]
                topk_items = topk[idx].tolist()

                for k in k_values:
                    top_k = topk_items[:k]
                    hit = int(target_item in top_k)
                    recalls[k].append(hit)

                    relevance = [1 if item == target_item else 0 for item in top_k]
                    ndcg = ndcg_score([relevance], [[1] * len(relevance)])
                    ndcgs[k].append(ndcg)

    metrics = {
        "recall": {k: np.mean(recalls[k]) for k in k_values},
        "ndcg": {k: np.mean(ndcgs[k]) for k in k_values},
    }

    return metrics
