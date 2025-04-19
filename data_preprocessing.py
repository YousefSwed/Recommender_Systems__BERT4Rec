# data_preprocessing.py

import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from config import SEQ_LEN, PROCESSED_DIR, DATA_DIR

def load_data():
    df = pd.read_csv(DATA_DIR, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] >= 4].sort_values(by=['user', 'timestamp'])
    return df

def build_sequences(df):
    user_seqs = defaultdict(list)
    for _, row in df.iterrows():
        user_seqs[row['user']].append(row['item'])
    return user_seqs

def split_sequence(seq):
    n_total = len(seq)
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)

    train = seq[:train_end]
    val = seq[train_end:val_end]
    test = seq[val_end:]

    return train, val, test

def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

def process_and_save():
    df = load_data()
    user_seqs = build_sequences(df)

    train_data, val_data, test_data = [], [], []

    for user, seq in user_seqs.items():
        if len(seq) < 5:
            continue

        train, val, test = split_sequence(seq)
        train_data.append(pad_sequence(train, SEQ_LEN))
        val_data.append(pad_sequence(val, SEQ_LEN))
        test_data.append(pad_sequence(test, SEQ_LEN))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(PROCESSED_DIR + 'train_seqs.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(PROCESSED_DIR + 'val_seqs.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(PROCESSED_DIR + 'test_seqs.pkl', 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == "__main__":
    process_and_save()
