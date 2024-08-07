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


def _load_full_data():
    # data = np.load("data/spects.npz")
    # return torch.from_numpy(data['X_seq_all']), torch.from_numpy(data['Y_seq_all'])

    all_names = get_all_names_list()

    X_seq_all = []
    Y_seq_all = []

    for name in all_names:
        X_seq, Y_seq = _load_name(name)

        X_seq_all.extend(X_seq)
        Y_seq_all.extend(Y_seq)

    X_seq_all = np.array(X_seq_all, dtype=np.float32)
    Y_seq_all = np.array(Y_seq_all, dtype=np.float32)

    np.savez('data/spects.npz', X_seq_all=X_seq_all, Y_seq_all=Y_seq_all)

    return torch.from_numpy(X_seq_all), torch.from_numpy(Y_seq_all)


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
    
