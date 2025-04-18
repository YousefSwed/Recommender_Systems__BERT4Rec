import numpy as np

def recall_at_k(ranked, gt, k):
    return 1 if gt in ranked[:k] else 0

def ndcg_at_k(ranked, gt, k):
    if gt in ranked[:k]:
        idx = ranked[:k].index(gt)
        return 1.0 / np.log2(idx + 2)
    return 0.0


def compute_metrics(logits, ground_truth, mask_positions=None, k=10):
    """
    Compute Recall@k and NDCG@k.

    - logits: torch.Tensor [N, V] of scores
    - ground_truth: torch.Tensor [N] of true item IDs
    - mask_positions: optional torch.BoolTensor [N] indicating which entries to evaluate
    - k: cutoff for metrics
    """
    # Optionally filter by mask_positions
    if mask_positions is not None:
        mask = mask_positions.cpu().numpy().astype(bool)
        logits = logits[mask]
        ground_truth = ground_truth[mask]

    # Convert to numpy for ranking
    scores = logits.cpu().numpy()
    truths = ground_truth.cpu().numpy()

    recall_list, ndcg_list = [], []
    for i, gt in enumerate(truths):
        ranked = list(np.argsort(-scores[i]))
        recall_list.append(recall_at_k(ranked, int(gt), k))
        ndcg_list.append(ndcg_at_k(ranked, int(gt), k))

    return { 'recall': np.mean(recall_list), 'ndcg': np.mean(ndcg_list) }