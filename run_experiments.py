import torch
import pickle
import json
from train import train_model
from evaluate import evaluate_model
from BERT4Rec_model import BERT4Rec
from config import PROCESSED_DIR, MODEL_SAVE_PATH
from plot import plot_experiment_results

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Define experiment sets
EXPERIMENT_SETS = {
    "embed_dim": [
        {"label": "Embed 32", "embed_dim": 32, "num_layers": 2, "mask_prob": 0.15},
        {"label": "Embed 64", "embed_dim": 64, "num_layers": 2, "mask_prob": 0.15},
        {"label": "Embed 128", "embed_dim": 128, "num_layers": 2, "mask_prob": 0.15},
        {"label": "Embed 256", "embed_dim": 256, "num_layers": 2, "mask_prob": 0.15},
        {"label": "Embed 512", "embed_dim": 512, "num_layers": 2, "mask_prob": 0.15}
    ],
    "num_layers": [
        {"label": "Layers 1", "embed_dim": 512, "num_layers": 1, "mask_prob": 0.15},
        {"label": "Layers 2", "embed_dim": 512, "num_layers": 2, "mask_prob": 0.15},
        {"label": "Layers 4", "embed_dim": 512, "num_layers": 4, "mask_prob": 0.15}
    ],
    "mask_prob": [
        {"label": "Mask 15%", "embed_dim": 512, "num_layers": 2, "mask_prob": 0.15},
        {"label": "Mask 30%", "embed_dim": 512, "num_layers": 2, "mask_prob": 0.3},
        {"label": "Mask 50%", "embed_dim": 512, "num_layers": 2, "mask_prob": 0.5}
    ]
}

def run_experiment_group(experiment_list, plot_name):
    train_data = load_pickle(PROCESSED_DIR + 'train_seqs.pkl')
    val_data = load_pickle(PROCESSED_DIR + 'val_seqs.pkl')
    test_data = load_pickle(PROCESSED_DIR + 'test_seqs.pkl')

    num_items = max(max(seq) for seq in train_data + val_data + test_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for config in experiment_list:
        print(f"\n=== Running: {config['label']} ===")
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

    # Save results
    output_json = f"results/{plot_name}_comparison.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_experiment_results(json_file=f"results/{plot_name}_comparison.json", save_path=f"results/{plot_name}_comparison.png")

    print(f"Saved results to {output_json} and plot to results/{plot_name}_comparison.png")

def run_all_experiments():
    for group_name, experiments in EXPERIMENT_SETS.items():
        run_experiment_group(experiments, plot_name=group_name)

if __name__ == "__main__":
    run_all_experiments()
    print("All experiments completed.")