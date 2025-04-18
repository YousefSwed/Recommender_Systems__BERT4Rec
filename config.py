# config.py

SEQ_LEN = 20
MASK_PROB = 0.15

BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 50
PATIENCE = 3

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2

MODEL_SAVE_PATH = "results/best_model.pt"

DATA_DIR = "ml-1m/ratings.dat"
PROCESSED_DIR = "preprocessed_data/"
RESULTS_DIR = "results/"
