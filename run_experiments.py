import torch
import pickle
import os
import json
import matplotlib.pyplot as plt
from train import train_model
from evaluate import evaluate_model
from BERT4Rec_model import BERT4Rec
from config import PROCESSED_DIR, MODEL_SAVE_PATH
from plot import plot_experiment_results

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

EXPERIMENTS = [
    {
        "label": "Baseline (d=128,L=2,mask=15%)",
        "embed_dim": 128,
        "num_layers": 2,
        "mask_prob": 0.15
    },
    {
        "label": "Larger Hidden (d=256)",
        "embed_dim": 256,
        "num_layers": 2,
        "mask_prob": 0.15
    },
    {
        "label": "More Layers (L=4)",
        "embed_dim": 128,
        "num_layers": 4,
        "mask_prob": 0.15
    },
    {
        "label": "High Mask (50%)",
        "embed_dim": 128,
        "num_layers": 2,
        "mask_prob": 0.5
    }
]

def run_experiments():
    train_data = load_pickle(PROCESSED_DIR + 'train_seqs.pkl')
    val_data = load_pickle(PROCESSED_DIR + 'val_seqs.pkl')
    test_data = load_pickle(PROCESSED_DIR + 'test_seqs.pkl')

    num_items = max(max(seq) for seq in train_data + val_data + test_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for config in EXPERIMENTS:
        print(f"\n=== Running: {config['label']} ===\n")
        model = BERT4Rec(
            num_items=num_items,
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"]
        )

        train_model(model, train_data, val_data, num_items, device, mask_prob=config["mask_prob"])

        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(device)

        metrics = evaluate_model(model, test_data, num_items, device, k_values=[10])
        result_entry = {
            "label": config["label"],
            "recall@10": metrics["recall"][10],
            "ndcg@10": metrics["ndcg"][10]
        }
        results.append(result_entry)

    # Save results to JSON
    with open("results/config_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison chart
    plot_experiment_results(results)
    
    print("Experiments completed. Results saved to 'results/config_comparison.json' and plot saved to 'results/config_comparison.png'.")

if __name__ == "__main__":
    run_experiments()
