import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from BERT4Rec_model import BERT4Rec
from train import train_model, mask_last_item
from evaluate import compute_metrics
from config import config


def load_seqs(name):
    with open(os.path.join('preprocessed_data', name), 'rb') as f:
        return pickle.load(f)


def make_loader(seqs, batch_size=64, shuffle=False):
    t = torch.tensor(seqs, dtype=torch.long)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)


def plot_learning(history):
    ep = [h['epoch'] for h in history]
    tr = [h['train_loss'] for h in history]
    vl = [h['val_loss'] for h in history]
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.plot(ep, tr, label='Train')
    plt.plot(ep, vl, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/learning_curves.png')
    plt.close()


def plot_metrics(model, loader, device):
    ks = [5, 10, 15, 20, 50]
    R, N = [], []
    for k in ks:
        all_logits, all_lbls = [], []
        model.eval()
        with torch.no_grad():
            for batch, in loader:
                batch = batch.to(device)
                masked, _ = mask_last_item(batch)
                h = model(masked)
                logits = model.predict_next(h)
                non_zero = (batch != 0).sum(dim=1) - 1
                gt = batch[torch.arange(batch.size(0)), non_zero]
                all_logits.append(logits)
                all_lbls.append(gt)
        logits_cat = torch.cat(all_logits, 0)
        lbls_cat = torch.cat(all_lbls, 0)
        mask_positions = torch.ones(lbls_cat.size(0), dtype=torch.bool, device=lbls_cat.device)
        m = compute_metrics(logits_cat, lbls_cat, mask_positions, k)
        R.append(m['recall'])
        N.append(m['ndcg'])

    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.plot(ks, R, marker='o', label='Recall@k')
    plt.plot(ks, N, marker='o', label='NDCG@k')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('results/metrics_at_k.png')
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load preprocessed sequences
    train_seqs = load_seqs('train_seqs.pkl')
    val_seqs = load_seqs('val_seqs.pkl')
    test_seqs = load_seqs('test_seqs.pkl')

    train_loader = make_loader(train_seqs, shuffle=True)
    val_loader = make_loader(val_seqs)
    test_loader = make_loader(test_seqs)

    # Determine vocab size
    items = [i for seq in train_seqs for i in seq if i != 0]
    num_items = max(items)

    model = BERT4Rec(num_items=num_items, max_seq_len=config['max_seq_len'],
                 embed_dim=config['embed_dim'], num_layers=config['num_layers'], num_heads=config['num_heads'], dropout=config['dropout']).to(device)
    model, history = train_model(model, train_loader, val_loader,
                                 num_epochs=50, lr=1e-4, device=device)

    # Load best model
    model.load_state_dict(torch.load('results/best_model.pth', map_location=device))

    # Test evaluation
    all_logits, all_lbls = [], []
    model.eval()
    with torch.no_grad():
        for batch, in test_loader:
            batch = batch.to(device)
            masked, _ = mask_last_item(batch)
            h = model(masked)
            logits = model.predict_next(h)
            non_zero = (batch != 0).sum(dim=1) - 1
            gt = batch[torch.arange(batch.size(0)), non_zero]
            all_logits.append(logits)
            all_lbls.append(gt)
    logits_cat = torch.cat(all_logits, 0)
    lbls_cat = torch.cat(all_lbls, 0)

    # Only evaluate on each example (1D mask)
    mask_positions = torch.ones(lbls_cat.size(0), dtype=torch.bool, device=lbls_cat.device)
    metrics = compute_metrics(logits_cat, lbls_cat, mask_positions, k=10)
    print(f"Test @10 → Recall: {metrics['recall']:.4f}, NDCG: {metrics['ndcg']:.4f}")

    # Plotting
    os.makedirs('results', exist_ok=True)
    plot_learning(history)
    plot_metrics(model, test_loader, device)

if __name__ == '__main__':
    main()