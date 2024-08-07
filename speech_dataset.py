import torch
import numpy as np

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from io_utils import *
from feature_ext import *

def _load_name(name: str):
    audio_path, ann_path = get_file_paths(name)

    signal, frequency = read_audio(audio_path)

    # list[(duration seconds, int label)]
    annotations = read_annotation(ann_path)

    # TODO: how to deal with rounding errors ?
    signal_labels = []
    for (duration, label) in annotations:
        num_dur_samples = int(np.round(duration * frequency))
        for _ in range(num_dur_samples):
            signal_labels.append(label)


    return create_spectograms(
        signal,
        frequency,
        signal_labels
    )

# ---------------------------------

def load_all_names_flat():
    data = np.load("data/spects_flat.npz")
    return torch.from_numpy(data['X_flat']), torch.from_numpy(data['Y_flat'])

    all_names = get_all_names_list()

    X_flat = []
    Y_flat = []

    for name in all_names:
        X_seq, Y_seq = _load_name(name)

        X_flat.extend(X_seq)
        Y_flat.extend(Y_seq)

    X_flat = np.array(X_flat, dtype=np.float32)
    Y_flat = np.array(Y_flat, dtype=np.float32)

    np.savez('data/spects_flat.npz', X_flat=X_flat, Y_flat=Y_flat)

    return torch.from_numpy(X_flat), torch.from_numpy(Y_flat)

def chunk_into_sequences(X, Y, seq_len: int = 15):
    batch_size = X.shape[0]

    chunk_idx = list(
        range(0, batch_size - seq_len + 1, seq_len))

    X_chunked = torch.stack([
        X[i:i+seq_len]
        for i in chunk_idx
    ], dim=0)

    Y_chunked = torch.stack([
        Y[i:i+seq_len]
        for i in chunk_idx
    ], dim=0)

    return X_chunked, Y_chunked


def load_all_names_sequenced():
    data = np.load("data/spects.npz")
    return torch.from_numpy(data['X']), torch.from_numpy(data['Y'])

    X_flat, Y_flat = load_all_names_flat()
    X, Y = chunk_into_sequences(X_flat, Y_flat)

    np.savez('data/spects.npz', X=X.numpy(), Y=Y.numpy())

    return X, Y


def load_torch_dataset():
    X_seq_all, Y_seq_all = _load_full_data()

    X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(
        X_seq_all, Y_seq_all,
        test_size=0.33,
        shuffle=False
    )

    X_train, Y_train = chunk_into_sequences(X_train_all, Y_train_all)
    X_test, Y_test = chunk_into_sequences(X_test_all, Y_test_all)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset, test_dataset
    
