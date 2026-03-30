"""Synthetic time-series health data generator for federated learning.

Generates per-user sinusoidal signals with Gaussian noise and injected
anomalies (amplitude spikes). Each user has unique baseline parameters
so that clients are genuinely heterogeneous.
"""

import os
import numpy as np


TEST_STEPS = 1000
WINDOW_SIZE = 128
STRIDE = 64
MIN_TEST_ANOMALY_WINDOWS = 5
WINDOW_ANOMALY_THRESHOLD = 0.30


def generate_user_params(rng):
    """Sample unique baseline parameters for one user.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    dict
        Keys: mean, amplitude, frequency, noise_std.
    """
    return {
        "mean": rng.uniform(-2.0, 2.0),
        "amplitude": rng.uniform(0.5, 3.0),
        "frequency": rng.uniform(0.002, 0.02),
        "noise_std": rng.uniform(0.05, 0.5),
    }


def generate_signal(params, n_steps, rng):
    """Create a single-channel sinusoidal signal with Gaussian noise.

    Parameters
    ----------
    params : dict
        User baseline parameters (mean, amplitude, frequency, noise_std).
    n_steps : int
        Number of time steps to generate.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    numpy.ndarray
        Signal array of shape (n_steps, 1).
    """
    t = np.arange(n_steps, dtype=np.float64)
    signal = (
        params["mean"]
        + params["amplitude"] * np.sin(2 * np.pi * params["frequency"] * t)
        + rng.normal(0.0, params["noise_std"], size=n_steps)
    )
    return signal.reshape(-1, 1)


def _count_test_anomaly_windows(labels, test_steps=TEST_STEPS,
                                window_size=WINDOW_SIZE, stride=STRIDE):
    """Count anomaly-positive windows in the test tail of the sequence.

    A window is positive if more than 30% of its steps are anomalous.

    Parameters
    ----------
    labels : numpy.ndarray
        Binary anomaly labels of shape (n_steps,).
    test_steps : int, optional
        Number of final steps considered test region.
    window_size : int, optional
        Sliding window size.
    stride : int, optional
        Sliding window stride.

    Returns
    -------
    int
        Number of positive windows whose start lies in the test region.
    """
    n_steps = labels.shape[0]
    test_start = n_steps - test_steps
    starts = np.arange(0, n_steps - window_size + 1, stride)
    starts = starts[starts >= test_start]

    count = 0
    for start in starts:
        end = start + window_size
        if labels[start:end].mean() > WINDOW_ANOMALY_THRESHOLD:
            count += 1
    return int(count)


def _inject_forced_test_anomalies(signal, labels, params, rng,
                                  min_test_windows=MIN_TEST_ANOMALY_WINDOWS,
                                  test_steps=TEST_STEPS,
                                  window_size=WINDOW_SIZE,
                                  stride=STRIDE):
    """Inject anomalies aligned to test windows to guarantee test coverage.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal array of shape (n_steps, 1).
    labels : numpy.ndarray
        Binary labels array of shape (n_steps,).
    params : dict
        User baseline parameters.
    rng : numpy.random.Generator
        Random number generator instance.
    min_test_windows : int, optional
        Minimum number of anomaly-positive windows required in test region.
    test_steps : int, optional
        Number of final time steps considered test region.
    window_size : int, optional
        Sliding window size used by downstream preprocessing.
    stride : int, optional
        Sliding window stride used by downstream preprocessing.

    Returns
    -------
    int
        Number of newly injected anomaly steps.
    """
    n_steps = labels.shape[0]
    test_start = n_steps - test_steps
    candidate_starts = np.arange(0, n_steps - window_size + 1, stride)
    candidate_starts = candidate_starts[candidate_starts >= test_start]

    if candidate_starts.size < min_test_windows:
        raise ValueError("Not enough candidate test windows to enforce coverage")

    forced_indices = np.linspace(
        0, candidate_starts.size - 1, min_test_windows, dtype=int
    )
    forced_starts = candidate_starts[forced_indices]

    injected_steps = 0
    for start in forced_starts:
        # >30% of 128 requires at least 39 anomalous steps.
        duration = int(rng.integers(39, 51))
        end = min(start + duration, n_steps)

        if np.any(labels[start:end]):
            continue

        spike_factor = rng.uniform(3.0, 6.0)
        signal[start:end, 0] += spike_factor * params["amplitude"]
        labels[start:end] = 1
        injected_steps += (end - start)

    return int(injected_steps)


def inject_anomalies(signal, params, anomaly_rate, rng):
    """Inject labelled amplitude-spike anomalies into a signal.

    Anomalies are random contiguous segments whose values are multiplied by
    a factor between 3x and 6x the user's normal amplitude.

    Parameters
    ----------
    signal : numpy.ndarray
        Clean signal of shape (n_steps, 1).
    params : dict
        User baseline parameters (used for amplitude reference).
    anomaly_rate : float
        Fraction of total steps that should be anomalous (e.g. 0.05).
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    signal : numpy.ndarray
        Signal with injected anomalies, shape (n_steps, 1).
    labels : numpy.ndarray
        Binary labels, shape (n_steps,). 1 = anomaly, 0 = normal.
    """
    n_steps = signal.shape[0]
    labels = np.zeros(n_steps, dtype=np.int32)
    target_anomaly_steps = int(n_steps * anomaly_rate)

    # Force anomaly coverage in the final test region so no user has empty
    # anomaly windows during per-user test evaluation.
    injected = _inject_forced_test_anomalies(signal, labels, params, rng)

    while injected < target_anomaly_steps:
        duration = rng.integers(10, 51)  # 10-50 steps inclusive
        remaining = target_anomaly_steps - injected
        if remaining < 10:
            break
        duration = min(duration, remaining)

        start = rng.integers(0, n_steps - duration)
        if np.any(labels[start : start + duration]):
            continue  # avoid overlapping anomalies

        spike_factor = rng.uniform(3.0, 6.0)
        signal[start : start + duration, 0] += (
            spike_factor * params["amplitude"]
        )
        labels[start : start + duration] = 1
        injected += duration

    # Safety net: top up test anomalies until at least MIN_TEST_ANOMALY_WINDOWS.
    while _count_test_anomaly_windows(labels) < MIN_TEST_ANOMALY_WINDOWS:
        injected += _inject_forced_test_anomalies(
            signal,
            labels,
            params,
            rng,
            min_test_windows=1,
        )

    return signal, labels


def generate_user_data(user_id, n_steps=5000, anomaly_rate=0.10, seed=None):
    """Generate signal and labels for a single federated client.

    Parameters
    ----------
    user_id : int
        Integer identifier for the user (used for seeding).
    n_steps : int, optional
        Number of time steps per user (default 5000).
    anomaly_rate : float, optional
        Fraction of time steps that are anomalous (default 0.10).
    seed : int or None, optional
        Base random seed. The per-user seed is ``seed + user_id``.

    Returns
    -------
    signal : numpy.ndarray
        Shape (n_steps, 1).
    labels : numpy.ndarray
        Shape (n_steps,).
    """
    per_user_seed = (seed + user_id) if seed is not None else None
    rng = np.random.default_rng(per_user_seed)

    params = generate_user_params(rng)
    signal = generate_signal(params, n_steps, rng)
    signal, labels = inject_anomalies(signal, params, anomaly_rate, rng)
    return signal, labels


def generate_dataset(n_users=50, n_steps=5000, anomaly_rate=0.10,
                     output_dir="data/raw/synthetic", seed=42):
    """Generate and save synthetic data for all federated clients.

    Each user is saved as two files:
      - ``user_XXX_signal.npy`` of shape (n_steps, 1)
      - ``user_XXX_labels.npy`` of shape (n_steps,)

    Parameters
    ----------
    n_users : int, optional
        Number of federated clients to generate (default 50).
    n_steps : int, optional
        Time steps per user (default 5000).
    anomaly_rate : float, optional
        Fraction of anomalous steps (default 0.10).
    output_dir : str, optional
        Directory where .npy files are saved.
    seed : int, optional
        Base random seed for reproducibility (default 42).
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_users):
        signal, labels = generate_user_data(
            user_id=i, n_steps=n_steps, anomaly_rate=anomaly_rate, seed=seed,
        )
        signal_path = os.path.join(output_dir, f"user_{i:03d}_signal.npy")
        labels_path = os.path.join(output_dir, f"user_{i:03d}_labels.npy")
        np.save(signal_path, signal)
        np.save(labels_path, labels)
        test_anomaly_steps = int(labels[-TEST_STEPS:].sum())
        test_anomaly_windows = _count_test_anomaly_windows(labels)
        print(
            f"Generated user {i + 1}/{n_users} | "
            f"test anomaly steps: {test_anomaly_steps} | "
            f"test anomaly windows: {test_anomaly_windows}"
        )

    print(f"\nDone. {n_users} users saved to {output_dir}/")


if __name__ == "__main__":
    generate_dataset()
