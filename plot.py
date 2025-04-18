# plot.py

import json
import matplotlib.pyplot as plt
import os

def plot_learning_curves(json_file='results/model_performance.json', save_path='results/learning_curves.png'):
    with open(json_file) as f:
        history = json.load(f)

    epochs = [x["epoch"] for x in history]
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]

    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics, save_path='results/metrics_at_k.png'):
    ks = list(metrics['recall'].keys())
    recall_vals = [metrics['recall'][k] for k in ks]
    ndcg_vals = [metrics['ndcg'][k] for k in ks]

    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.plot(ks, recall_vals, marker='o', label="Recall@k")
    plt.plot(ks, ndcg_vals, marker='s', label="NDCG@k")
    plt.xlabel("k")
    plt.ylabel("Metric")
    plt.title("Recall and NDCG at k")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_experiment_results(results, save_path='results/config_comparison.png'):
    labels = [r["label"] for r in results]
    recall = [r["recall@10"] for r in results]
    ndcg = [r["ndcg@10"] for r in results]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], recall, width=width, label='Recall@10', color='orange')
    ax.bar([i + width/2 for i in x], ndcg, width=width, label='NDCG@10', color='orangered')

    ax.set_ylabel('Metric value')
    ax.set_title('Model Configuration Comparison on ML-1M')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    plt.close()