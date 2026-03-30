"""Utilities to load and split the MIT-BIH heartbeat dataset for federated use."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _format_distribution(name, y):
    """Print class distribution and imbalance for a binary label vector."""
    y = np.asarray(y).reshape(-1)
    n_total = int(y.size)
    n_normal = int((y == 0).sum())
    n_anomaly = int((y == 1).sum())
    anomaly_rate = (n_anomaly / n_total) if n_total > 0 else 0.0
    imbalance_ratio = (n_normal / n_anomaly) if n_anomaly > 0 else float("inf")

    print(
        f"{name}: total={n_total}, normal={n_normal}, anomaly={n_anomaly}, "
        f"anomaly_rate={anomaly_rate:.4f}, imbalance_ratio(normal/anomaly)={imbalance_ratio:.4f}"
    )


def _stratified_split_three_way(X, y, seed):
    """Split arrays into train/val/test with stratification when possible."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.shape[0] == 0:
        return X, y, X, y, X, y

    def _can_stratify(labels):
        unique, counts = np.unique(labels, return_counts=True)
        return unique.size >= 2 and counts.min() >= 2

    n_samples = X.shape[0]
    if n_samples < 5:
        # Tiny fallback split to avoid train_test_split edge-case failures.
        i1 = max(1, int(0.6 * n_samples))
        i2 = max(i1 + 1, int(0.8 * n_samples))
        i2 = min(i2, n_samples)
        return (
            X[:i1],
            y[:i1],
            X[i1:i2],
            y[i1:i2],
            X[i2:],
            y[i2:],
        )

    stratify_all = y if _can_stratify(y) else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=seed,
        stratify=stratify_all,
    )

    stratify_tmp = y_tmp if _can_stratify(y_tmp) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_tmp,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mitbih(train_path, test_path):
    """Load MIT-BIH train/test CSVs and return Conv1d-ready NumPy arrays.

    Parameters
    ----------
    train_path : str
        Path to mitbih_train.csv.
    test_path : str
        Path to mitbih_test.csv.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        X_train, y_train, X_test, y_test where:
        - X_* shape is (N, 1, 187)
        - y_* shape is (N,)

    Notes
    -----
    Label mapping:
    - 0 -> 0 (normal)
    - 1,2,3,4 -> 1 (anomaly)
    """
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train_raw = train_df.iloc[:, -1].to_numpy(dtype=np.int32)
    X_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test_raw = test_df.iloc[:, -1].to_numpy(dtype=np.int32)

    y_train = (y_train_raw != 0).astype(np.int32)
    y_test = (y_test_raw != 0).astype(np.int32)

    # Conv1d input format: (N, C, L) with C=1 and L=187.
    X_train = X_train.reshape(-1, 1, 187)
    X_test = X_test.reshape(-1, 1, 187)

    print("MIT-BIH binary class distribution:")
    _format_distribution("Train", y_train)
    _format_distribution("Test", y_test)
    y_all = np.concatenate([y_train, y_test], axis=0)
    _format_distribution("Combined", y_all)

    return X_train, y_train, X_test, y_test


def split_by_patient_simulated(X, y, n_users=47, seed=42):
    """Simulate federated users by stratified chunking and per-user splitting.

    Since this MIT-BIH CSV version does not include patient IDs, users are
    simulated by distributing class-specific samples across users so each user
    gets a roughly balanced local dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Input beats with shape (N, 1, 187).
    y : numpy.ndarray
        Binary labels with shape (N,).
    n_users : int, optional
        Number of simulated federated users.
    seed : int, optional
        Random seed.

    Returns
    -------
    list[dict]
        List with one dictionary per user:
        {user_id, X_train, y_train, X_val, y_val, X_test, y_test}
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same first dimension")
    if n_users <= 0:
        raise ValueError("n_users must be positive")

    rng = np.random.default_rng(seed)

    idx_normal = np.where(y == 0)[0]
    idx_anomaly = np.where(y == 1)[0]
    rng.shuffle(idx_normal)
    rng.shuffle(idx_anomaly)

    normal_chunks = np.array_split(idx_normal, n_users)
    anomaly_chunks = np.array_split(idx_anomaly, n_users)

    user_splits = []

    for user_id in range(n_users):
        user_idx = np.concatenate([normal_chunks[user_id], anomaly_chunks[user_id]])
        if user_idx.size == 0:
            user_X = np.empty((0, 1, 187), dtype=np.float32)
            user_y = np.empty((0,), dtype=np.int32)
        else:
            rng.shuffle(user_idx)
            user_X = X[user_idx]
            user_y = y[user_idx]

        X_train, y_train, X_val, y_val, X_test, y_test = _stratified_split_three_way(
            user_X, user_y, seed=seed + user_id
        )

        user_splits.append(
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

    print(f"Created {len(user_splits)} simulated users from MIT-BIH data")
    return user_splits
