# BERT4Rec Assignment

## 👥 Team

Group 33  

## 📚 Overview

This project implements a **sequential recommendation system** based on the **BERT4Rec** architecture using a custom Transformer in PyTorch. The model is trained using **masked item prediction**, inspired by masked language modeling, to predict the next item in a user's interaction sequence.

It is trained and evaluated on the **MovieLens 1M dataset**, and supports:

- Early stopping based on **validation NDCG@10**
- Configurable model dimensions, layers, and masking ratios
- Plotting of learning curves and recommendation metrics
- Comparison across multiple experiment configurations

---

## 📁 Directory Structure

```plaintext
BERT4Rec_Assignment/
├── BERT4Rec_model.py
├── config.py
├── data_preprocessing.py
├── evaluate.py
├── main.py
├── plot.py
├── run_experiments.py
├── train.py
├── README.md
├── ml-1m/
│   └── ratings.dat
├── preprocessed_data/
│   ├── train_seqs.pkl
│   ├── val_seqs.pkl
│   └── test_seqs.pkl
└── results/
    ├── best_model.pt               # Best model checkpoint
    ├── model_performance.json      # Epoch-wise loss & metrics
    ├── model_metrics.json          # Final test Recall/NDCG @k
    ├── learning_curves.png         # Training vs validation loss
    ├── metrics_at_k.png            # NDCG and Recall at various k
    ├── config_comparison.json      # Results from all config experiments
    └── config_comparison.png       # Grouped bar chart of those results
```

---

## 📦 Requirements

- Python 3.6+
- PyTorch (>= 1.6)
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

Install with pip:

```bash
pip install torch pandas numpy matplotlib scikit-learn tqdm
```

---

## 🚀 How to Run

### 1. Download the Dataset

Place the `ratings.dat` file into the `ml-1m/` directory.

### 2. Train and Evaluate the Model

```bash
python main.py
```

This will:

- Preprocess the data (70/15/15 split per user)
- Train the model with early stopping (based on validation NDCG@10)
- Save training metrics and plots
- Evaluate performance on the test set

### 3. Run Multiple Configuration Experiments

To compare different embedding sizes, layer counts, and masking ratios:

```bash
python run_experiments.py
```

### 4. View Results

Check the `results/` folder for:

- Learning curve: `learning_curves.png`
- Final evaluation metrics: `model_metrics.json`
- Metric-at-k plot: `metrics_at_k.png`
- Configuration comparison bar chart: `config_comparison.png`

---

## 📈 Evaluation Metrics

The model is evaluated using the **leave-one-out strategy** with:

- **Recall@k**
- **NDCG@k**

for `k = [5, 10, 15, 20, 50]`.

---

## 🛠 Model Features

- Custom Transformer encoder (PyTorch native)
- Positional & item embeddings
- Masked item prediction (15% by default)
- Early stopping with configurable patience
- Configurable model parameters (layers, hidden size, masking rate)

---
