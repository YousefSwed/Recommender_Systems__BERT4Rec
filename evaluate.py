# evaluate.py

import torch
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

def evaluate_model(model, test_data, num_items, device, k_values=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
    model.eval()
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}

    with torch.no_grad():
        for seq in tqdm(test_data, desc="Evaluating"):
            input_seq = torch.LongTensor([seq]).to(device)
            mask = (input_seq == 0)
            output = model(input_seq, mask)[0][-1]  # [num_items]
            topk_scores = torch.topk(output, max(k_values)).indices.cpu().numpy()

            target_item = seq[-1]
            for k in k_values:
                top_k = topk_scores[:k]
                hit = int(target_item in top_k)
                recalls[k].append(hit)

                # for NDCG
                relevance = [1 if item == target_item else 0 for item in top_k]
                ndcg = ndcg_score([relevance], [[1] * len(relevance)])
                ndcgs[k].append(ndcg)

    metrics = {
        "recall": {k: np.mean(recalls[k]) for k in k_values},
        "ndcg": {k: np.mean(ndcgs[k]) for k in k_values},
    }

    return metrics
