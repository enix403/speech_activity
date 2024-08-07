# np.savez('data/spects.npz', X_seq_all=X_seq_all, Y_seq_all=Y_seq_all)

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


X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(
    X_seq_all, Y_seq_all,
    test_size=0.33,
    shuffle=False
)

print(f"{X_train.shape=} {Y_train.shape=}")
print(f"{X_test.shape=} {Y_test.shape=}")