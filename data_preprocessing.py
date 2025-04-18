import pandas as pd
import pickle
import os
import numpy as np

MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
PAD_TOKEN = 0


def load_movielens(path):
    """
    Load and filter MovieLens data:
    - Keep only ratings >= 4 (positive implicit feedback).
    - Sort by timestamp.
    """
    cols = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='::', engine='python', names=cols)
    df = df[df.rating >= 4]
    return df.sort_values('timestamp')


def pad_trunc(seq, max_len=MAX_SEQ_LEN, pad_value=PAD_TOKEN):
    """Pad or truncate a list to exactly max_len."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq


def split_user_sequence(seq):
    """
    Given a full seq, return three subseqs:
      - train: seq[:-2]
      - val:   seq[:-1]
      - test:  seq[1:]
    Each will then be padded/truncated.
    """
    return seq[:-2], seq[:-1], seq[1:]


def build_datasets(df):
    """
    Split the dataset into training (70%), validation (15%), and testing (15%) by user.
    Generate three lists of padded sequences.
    """
    train_seqs, val_seqs, test_seqs = [], [], []

    # Get unique users and shuffle them
    unique_users = df['userId'].unique()
    np.random.shuffle(unique_users)

    # Split users into train (70%), val (15%), test (15%)
    num_users = len(unique_users)
    train_users = unique_users[:int(0.7 * num_users)]
    val_users = unique_users[int(0.7 * num_users):int(0.85 * num_users)]
    test_users = unique_users[int(0.85 * num_users):]

    # Process sequences for each user group
    for user_group, seq_list in zip([train_users, val_users, test_users],
                                    [train_seqs, val_seqs, test_seqs]):
        for user in user_group:
            user_seq = df[df['userId'] == user]['movieId'].tolist()
            if len(user_seq) < MIN_SEQ_LEN:
                continue
            seq_list.append(pad_trunc(user_seq))

    return train_seqs, val_seqs, test_seqs

def main():
    data_path = os.path.join('ml-1m', 'ratings.dat')
    os.makedirs('preprocessed_data', exist_ok=True)

    df = load_movielens(data_path)
    train_seqs, val_seqs, test_seqs = build_datasets(df)

    with open('preprocessed_data/train_seqs.pkl', 'wb') as f:
        pickle.dump(train_seqs, f)
    with open('preprocessed_data/val_seqs.pkl', 'wb') as f:
        pickle.dump(val_seqs, f)
    with open('preprocessed_data/test_seqs.pkl', 'wb') as f:
        pickle.dump(test_seqs, f)

    print(f"Saved: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences.")


if __name__ == '__main__':
    main()