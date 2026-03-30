"""Preprocessing utilities for federated time-series anomaly detection.

This module provides per-user preprocessing steps:
1) channel-wise normalization,
2) sliding-window extraction,
3) time-ordered train/val/test splitting,
4) one-shot user data loading pipeline.
"""

import os
import numpy as np


def sliding_windows(signal, labels, window_size=128, stride=64):
	"""Create overlapping windows and aggregate labels per window.

	Parameters
	----------
	signal : numpy.ndarray
		Time-series input array with shape (T, C), where T is the number of
		time steps and C is the number of channels.
	labels : numpy.ndarray
		Step-level binary anomaly labels with shape (T,). Values are expected
		to be 0 (normal) or 1 (anomaly).
	window_size : int, optional
		Number of time steps per window W (default 128).
	stride : int, optional
		Step size between consecutive window starts (default 64).

	Returns
	-------
	windows : numpy.ndarray
		Windowed signal array of shape (N, W, C), where N is the number of
		extracted windows and W is `window_size`.
	window_labels : numpy.ndarray
		Window-level binary labels of shape (N,). A window label is 1 if more
		than 30% of its time steps are anomalous, else 0.

	Raises
	------
	ValueError
		If input shapes are invalid or if window/stride are not positive.
	"""
	if signal.ndim != 2:
		raise ValueError("signal must have shape (T, C)")
	if labels.ndim != 1:
		raise ValueError("labels must have shape (T,)")
	if signal.shape[0] != labels.shape[0]:
		raise ValueError("signal and labels must have matching time dimension")
	if window_size <= 0 or stride <= 0:
		raise ValueError("window_size and stride must be positive")

	t_steps, channels = signal.shape
	if t_steps < window_size:
		return (
			np.empty((0, window_size, channels), dtype=signal.dtype),
			np.empty((0,), dtype=np.int32),
		)

	starts = np.arange(0, t_steps - window_size + 1, stride)
	n_windows = starts.size

	windows = np.empty((n_windows, window_size, channels), dtype=signal.dtype)
	window_labels = np.empty((n_windows,), dtype=np.int32)

	for i, start in enumerate(starts):
		end = start + window_size
		windows[i] = signal[start:end]
		anomaly_fraction = labels[start:end].mean()
		window_labels[i] = 1 if anomaly_fraction > 0.30 else 0

	return windows, window_labels


def normalize(signal, mean=None, std=None):
	"""Apply per-channel z-score normalization.

	Parameters
	----------
	signal : numpy.ndarray
		Input signal of shape (T, C).
	mean : numpy.ndarray or None, optional
		Per-channel mean used for normalization, shape (C,). If None, mean is
		computed from `signal` (training mode).
	std : numpy.ndarray or None, optional
		Per-channel standard deviation used for normalization, shape (C,). If
		None, std is computed from `signal` (training mode).

	Returns
	-------
	normalized_signal : numpy.ndarray
		Normalized signal with shape (T, C).
	mean : numpy.ndarray
		Per-channel mean used, shape (C,).
	std : numpy.ndarray
		Per-channel standard deviation used, shape (C,).

	Raises
	------
	ValueError
		If `signal` shape is invalid or only one of mean/std is provided.
	"""
	if signal.ndim != 2:
		raise ValueError("signal must have shape (T, C)")

	if (mean is None) != (std is None):
		raise ValueError("mean and std must both be provided or both be None")

	if mean is None and std is None:
		mean = signal.mean(axis=0)
		std = signal.std(axis=0)
	else:
		mean = np.asarray(mean)
		std = np.asarray(std)
		if mean.shape != (signal.shape[1],) or std.shape != (signal.shape[1],):
			raise ValueError("mean and std must have shape (C,)")

	# Guard against zero variance channels.
	safe_std = np.where(std == 0, 1.0, std)
	normalized_signal = (signal - mean) / safe_std

	return normalized_signal, mean, safe_std


def split_windows(windows, labels, train_ratio=0.6, val_ratio=0.2):
	"""Split windows into train/val/test sets by time order.

	No shuffling is applied. The first block is training, followed by
	validation, followed by test.

	Parameters
	----------
	windows : numpy.ndarray
		Windowed data of shape (N, W, C).
	labels : numpy.ndarray
		Window labels of shape (N,).
	train_ratio : float, optional
		Fraction of the earliest windows assigned to train (default 0.6).
	val_ratio : float, optional
		Fraction of subsequent windows assigned to validation (default 0.2).
		The remaining windows are assigned to test.

	Returns
	-------
	X_train : numpy.ndarray
		Training windows, shape (N_train, W, C).
	y_train : numpy.ndarray
		Training labels, shape (N_train,).
	X_val : numpy.ndarray
		Validation windows, shape (N_val, W, C).
	y_val : numpy.ndarray
		Validation labels, shape (N_val,).
	X_test : numpy.ndarray
		Test windows, shape (N_test, W, C).
	y_test : numpy.ndarray
		Test labels, shape (N_test,).

	Raises
	------
	ValueError
		If shapes are invalid or split ratios are out of range.
	"""
	if windows.ndim != 3:
		raise ValueError("windows must have shape (N, W, C)")
	if labels.ndim != 1:
		raise ValueError("labels must have shape (N,)")
	if windows.shape[0] != labels.shape[0]:
		raise ValueError("windows and labels must have the same first dimension")
	if not (0.0 <= train_ratio <= 1.0):
		raise ValueError("train_ratio must be between 0 and 1")
	if not (0.0 <= val_ratio <= 1.0):
		raise ValueError("val_ratio must be between 0 and 1")
	if train_ratio + val_ratio > 1.0:
		raise ValueError("train_ratio + val_ratio must be <= 1")

	n_samples = windows.shape[0]
	n_train = int(n_samples * train_ratio)
	n_val = int(n_samples * val_ratio)
	n_train_val = n_train + n_val

	X_train = windows[:n_train]
	y_train = labels[:n_train]
	X_val = windows[n_train:n_train_val]
	y_val = labels[n_train:n_train_val]
	X_test = windows[n_train_val:]
	y_test = labels[n_train_val:]

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_user_data(user_id, data_dir, window_size=128, stride=64):
	"""Load, normalize, window, and split one user's data.

	Parameters
	----------
	user_id : int
		User identifier used in filenames. Files are expected as:
		``user_{user_id:03d}_signal.npy`` and ``user_{user_id:03d}_labels.npy``.
	data_dir : str
		Directory containing per-user `.npy` files.
	window_size : int, optional
		Sliding window size W (default 128).
	stride : int, optional
		Sliding window stride (default 64).

	Returns
	-------
	dict
		Dictionary containing:
		- ``X_train``: shape (N_train, W, C)
		- ``y_train``: shape (N_train,)
		- ``X_val``: shape (N_val, W, C)
		- ``y_val``: shape (N_val,)
		- ``X_test``: shape (N_test, W, C)
		- ``y_test``: shape (N_test,)
		- ``mean``: per-channel normalization mean, shape (C,)
		- ``std``: per-channel normalization std, shape (C,)

	Raises
	------
	FileNotFoundError
		If required user signal/label files are missing.
	"""
	signal_path = os.path.join(data_dir, f"user_{user_id:03d}_signal.npy")
	labels_path = os.path.join(data_dir, f"user_{user_id:03d}_labels.npy")

	if not os.path.exists(signal_path):
		raise FileNotFoundError(f"Missing signal file: {signal_path}")
	if not os.path.exists(labels_path):
		raise FileNotFoundError(f"Missing labels file: {labels_path}")

	signal = np.load(signal_path)
	labels = np.load(labels_path)

	norm_signal, mean, std = normalize(signal)
	windows, window_labels = sliding_windows(
		norm_signal, labels, window_size=window_size, stride=stride
	)
	X_train, y_train, X_val, y_val, X_test, y_test = split_windows(
		windows, window_labels
	)

	return {
		"X_train": X_train,
		"y_train": y_train,
		"X_val": X_val,
		"y_val": y_val,
		"X_test": X_test,
		"y_test": y_test,
		"mean": mean,
		"std": std,
	}
