import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_mitbih(train_path, test_path):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train CSV: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test CSV: {test_path}")

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    train_label_col = train_df.columns[-1]
    test_label_col = test_df.columns[-1]
    train_df = train_df.rename(columns={train_label_col: "label"})
    test_df = test_df.rename(columns={test_label_col: "label"})

    train_df["label"] = (train_df["label"].astype(int) >= 1).astype(int)
    test_df["label"] = (test_df["label"].astype(int) >= 1).astype(int)

    n_normal_tr = int((train_df["label"] == 0).sum())
    n_anomaly_tr = int((train_df["label"] == 1).sum())
    ratio_tr = (n_normal_tr / n_anomaly_tr) if n_anomaly_tr > 0 else float("inf")

    n_normal_te = int((test_df["label"] == 0).sum())
    n_anomaly_te = int((test_df["label"] == 1).sum())
    ratio_te = (n_normal_te / n_anomaly_te) if n_anomaly_te > 0 else float("inf")

    print("MIT-BIH loaded:")
    print(
        f"  Train: {len(train_df)} samples | "
        f"Normal={n_normal_tr} Anomaly={n_anomaly_tr} Ratio={ratio_tr:.1f}:1"
    )
    print(
        f"  Test:  {len(test_df)} samples | "
        f"Normal={n_normal_te} Anomaly={n_anomaly_te} Ratio={ratio_te:.1f}:1"
    )

    X_train = train_df.iloc[:, :187].values
    y_train = train_df["label"].values.astype(np.int32)
    X_test = test_df.iloc[:, :187].values
    y_test = test_df["label"].values.astype(np.int32)

    X_train = X_train.reshape(-1, 1, 187).astype(np.float32)
    X_test = X_test.reshape(-1, 1, 187).astype(np.float32)

    return X_train, y_train, X_test, y_test


def split_into_federated_users(X, y, n_users=47, seed=42):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same first dimension")
    if n_users <= 0:
        raise ValueError("n_users must be positive")

    rng = np.random.default_rng(seed)

    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]

    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    normal_chunks = np.array_split(normal_idx, n_users)
    anomaly_chunks = np.array_split(anomaly_idx, n_users)

    user_data_list = []
    train_sizes = []
    anomaly_rates = []

    for user_id in range(n_users):
        user_idx = np.concatenate([normal_chunks[user_id], anomaly_chunks[user_id]])
        rng.shuffle(user_idx)

        X_user = X[user_idx]
        y_user = y[user_idx]

        n_total = len(y_user)
        n_tr = int(0.6 * n_total)
        n_val = int(0.2 * n_total)

        X_train = X_user[:n_tr]
        y_train = y_user[:n_tr]
        X_val = X_user[n_tr:n_tr + n_val]
        y_val = y_user[n_tr:n_tr + n_val]
        X_test = X_user[n_tr + n_val:]
        y_test = y_user[n_tr + n_val:]

        anomaly_rate = float(y_user.mean()) if len(y_user) > 0 else 0.0

        user_data_list.append(
            {
                "user_id": int(user_id),
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }
        )

        train_sizes.append(len(X_train))
        anomaly_rates.append(anomaly_rate)

        print(
            f"User {user_id:03d}: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
            f"anomaly_rate={anomaly_rate:.3f}"
        )

    avg_train = float(np.mean(train_sizes)) if train_sizes else 0.0
    avg_rate = float(np.mean(anomaly_rates)) if anomaly_rates else 0.0
    print(
        f"Split complete: {n_users} users | "
        f"avg train size={avg_train:.0f} | avg anomaly rate={avg_rate:.3f}"
    )

    return user_data_list


def normalize_ecg_users(user_data_list):
    normalized = []

    for user_data in user_data_list:
        out = dict(user_data)

        X_train = out["X_train"].astype(np.float32)
        X_val = out["X_val"].astype(np.float32)
        X_test = out["X_test"].astype(np.float32)

        mean = float(X_train.mean()) if X_train.size > 0 else 0.0
        std = float(X_train.std()) if X_train.size > 0 else 1.0
        std = std + 1e-8

        out["X_train"] = ((X_train - mean) / std).astype(np.float32)
        out["X_val"] = ((X_val - mean) / std).astype(np.float32)
        out["X_test"] = ((X_test - mean) / std).astype(np.float32)
        out["mean"] = mean
        out["std"] = std

        normalized.append(out)

    return normalized


# Backward-compatible alias used by existing experiment scripts.
def split_by_patient_simulated(X, y, n_users=47, seed=42):
    return split_into_federated_users(X, y, n_users=n_users, seed=seed)


if __name__ == "__main__":
    train_csv = "data/raw/ecg/mitbih_train.csv"
    test_csv = "data/raw/ecg/mitbih_test.csv"

    X_train, y_train, X_test, y_test = load_mitbih(train_csv, test_csv)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    users = split_into_federated_users(X_all, y_all, n_users=47, seed=42)
    users = normalize_ecg_users(users)

    u0 = users[0]
    rate0 = float(u0["y_train"].mean()) if len(u0["y_train"]) > 0 else 0.0
    print(
        f"User 0: X_train={u0['X_train'].shape} y_train={u0['y_train'].shape} "
        f"anomaly_rate={rate0:.3f}"
    )
    print("ECG loader test passed!")
