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
    ├── best_model.pt                           # Best model checkpoint
    ├── model_performance.json                  # Epoch-wise loss & metrics
    ├── model_metrics.json                      # Final test Recall/NDCG @k
    ├── learning_curves.png                     # Training vs validation loss
    ├── metrics_at_k.png                        # NDCG and Recall at various k
    ├── compare_embed_dim_comparison.json       # Results from different hidden layers dimention experiments
    ├── compare_embed_dim_comparison.png        # Grouped bar chart of different hidden layers dimention results
    ├── compare_num_layers_comparison.json      # Results from different hidden layers experiments
    ├── compare_num_layers_comparison.png       # Grouped bar chart of different hidden layers results
    ├── compare_mask_prob_comparison.json       # Results from different mask probabilty experiments
    └── compare_mask_prob_comparison.png        # Grouped bar chart of different mask probabilty results
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

### Run on GPU (Optinal)

To train this module on a GPU, ensure that `CUDA` and `cuDNN` are installed on your device. Once installed, use the following command to install the appropriate version of PyTorch:

```bash
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu{version_number}
```

Replace `{version_number}` with the version of CUDA installed on your system.

For example, with `CUDA` version `12.8.1` and `cuDNN` version `9.8.0`, the command would be:

```bash
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note:** All training for this project was conducted on a GPU.

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

We conducted several experiments with the following configurations:

- Embedding dimensions: [32, 64, 128, 256, 512]
- Number of layers: [1, 2, 4]
- Masking ratios: [15%, 30%, 50%]

To execute the experiments, use the following command:

```bash
python run_experiments.py
```

### 4. View Results

Check the `results/` folder for:

- Epoch-wise loss and metrics: `model_performance.json`
- Learning curve: `learning_curves.png`
- Final evaluation metrics: `model_metrics.json`
- Metric-at-k plot: `metrics_at_k.png`
- Embedding dimension comparison bar chart: `compare_embed_dim_comparison.png`
- Results from embedding dimension experiments: `compare_embed_dim_comparison.json`
- Number of layers comparison bar chart: `compare_num_layers_comparison.png`
- Results from number of layers experiments: `compare_num_layers_comparison.json`
- Mask probability comparison bar chart: `compare_mask_prob_comparison.png`
- Results from mask probability experiments: `compare_mask_prob_comparison.json`
- Best model checkpoint: `best_model.pt`

---

## 📈 Evaluation Metrics

The model is evaluated using the **leave-one-out strategy** with:

- **Recall@k**
- **NDCG@k**

for `k = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]`.

---

## 🛠 Model Features

- Custom Transformer encoder (PyTorch native)
- Positional & item embeddings
- Masked item prediction (15% by default)
- Early stopping with configurable patience
- Configurable model parameters (layers, hidden size, masking rate)

---
