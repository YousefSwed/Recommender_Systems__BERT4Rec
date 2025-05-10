import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from config import SEQ_LEN, PROCESSED_DIR, DATA_DIR

def load_data():
    # Load the dataset from a CSV file and filter for ratings >= 4
    df = pd.read_csv(DATA_DIR, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] >= 4].sort_values(by=['user', 'timestamp'])  # Sort by user and timestamp
    return df

def build_sequences(df):
    # Build user-item interaction sequences
    user_seqs = defaultdict(list)
    for _, row in df.iterrows():
        user_seqs[row['user']].append(row['item'])  # Append item to the user's sequence
    return user_seqs

def split_sequence(seq):
    # Split a sequence into training, validation, and test parts
    n_total = len(seq)
    train_end = int(n_total * 0.7)  # 70% for training
    val_end = int(n_total * 0.85)  # Next 15% for validation

    train = seq[:train_end]
    val = seq[train_end:val_end]
    test = seq[val_end:]  # Remaining 15% for testing

    return train, val, test

def pad_sequence(seq, max_len):
    # Pad or truncate a sequence to a fixed length
    if len(seq) >= max_len:
        return seq[-max_len:]  # Truncate if longer than max_len
    return [0] * (max_len - len(seq)) + seq  # Pad with zeros if shorter

def process_and_save():
    # Process the data and save the sequences for training, validation, and testing
    df = load_data()
    user_seqs = build_sequences(df)

    train_data, val_data, test_data = [], [], []

    for user, seq in user_seqs.items():
        if len(seq) < 5:
            continue  # Skip users with fewer than 5 interactions

        train, val, test = split_sequence(seq)
        train_data.append(pad_sequence(train, SEQ_LEN))  # Pad training sequence
        val_data.append(pad_sequence(val, SEQ_LEN))  # Pad validation sequence
        test_data.append(pad_sequence(test, SEQ_LEN))  # Pad test sequence

    # Save processed sequences
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(PROCESSED_DIR + 'train_seqs.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(PROCESSED_DIR + 'val_seqs.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(PROCESSED_DIR + 'test_seqs.pkl', 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == "__main__":
    process_and_save()  # Execute the data processing and saving
