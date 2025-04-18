# BERT4Rec Assignment

Group 33

## Overview

This project implements a sequential recommendation model based on BERT4Rec using a transformer architecture. The model is trained on the MovieLens 1M dataset and uses a masked language modeling objective for next-item prediction.

## Directory Structure

```less
BERT4Rec_Assignment/
├── data_preprocessing.py
├── BERT4Rec_model.py
├── train.py
├── evaluate.py
├── main.py
├── README.md
├── ml-1m/ratings.dat
├── preprocessed_data/
│   ├── train_seqs.pkl
│   ├── val_seqs.pkl
│   └── test_seqs.pkl
└── results/
├── learning_curves.png
├── metrics_at_k.png
├── model_performance.json
└── best_model.pth
```

## Requirements

- Python 3.6+
- PyTorch (version 1.6+ recommended)
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

You can install the required packages using pip:

```bash
pip install torch pandas numpy matplotlib scikit-learn tqdm
```

## How to Run

1. **Preprocess the Data**

   Ensure that the MovieLens data file `ratings.dat` is located in the `ml-1m/` folder. Then run:

   ```bash
   python data_preprocessing.py
   ```

   This will generate preprocessed data files in the `preprocessed_data/` folder.

2. **Train and Evaluate the Model**

    Run the main script:

    ```bash
    python main.py
    ```

    This will train the model, evaluate it on the test set, save performance metrics to `results/model_performance.json`, and generate plots (`learning_curves.png` and `metrics_at_k.png`) in the `results/` directory.
