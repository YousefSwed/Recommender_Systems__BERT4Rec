# main.py

import pickle
import torch
import os

from data_preprocessing import process_and_save
from BERT4Rec_model import BERT4Rec
from train import train_model
from evaluate import evaluate_model
from plot import plot_learning_curves, plot_metrics
from config import PROCESSED_DIR, RESULTS_DIR, MODEL_SAVE_PATH

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def main():
    # Step 0: Check if directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Data Preprocessing
    print("Processing data...")
    process_and_save()

    # Step 2: Load Data
    print("Loading preprocessed data...")
    train_data = load_pickle(PROCESSED_DIR + 'train_seqs.pkl')
    val_data = load_pickle(PROCESSED_DIR + 'val_seqs.pkl')
    test_data = load_pickle(PROCESSED_DIR + 'test_seqs.pkl')

    # Step 3: Build Model
    print("Building model...")
    num_items = max(max(seq) for seq in train_data + val_data + test_data)
    model = BERT4Rec(num_items)

    # Step 4: Train
    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_data, val_data, num_items, device)

    # Step 5: Load Best Model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)

    # Step 6: Evaluate
    print("Evaluating...")
    metrics = evaluate_model(model, test_data, num_items, device)
    print("Evaluation Results:", metrics)

    # Step 7: Save Plots
    print("Generating plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_learning_curves()
    plot_metrics()

    # Step 8: Save metrics
    with open(os.path.join(RESULTS_DIR, "model_metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()