"""Dataset utilities for time-series anomaly detection.

This module bridges NumPy preprocessing outputs with PyTorch datasets and
dataloaders for centralized and federated experiments.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
	from src.data.preprocessing import load_user_data
except ModuleNotFoundError:
	# Allow running this file directly: python src/data/datasets.py
	current_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from src.data.preprocessing import load_user_data


class TimeSeriesDataset(Dataset):
	"""PyTorch dataset for windowed time-series samples.

	Parameters
	----------
	windows : numpy.ndarray
		Windowed signal array of shape (N, W, C), where N is number of
		windows, W is window length, and C is number of channels.
	labels : numpy.ndarray
		Window-level labels of shape (N,). Values are expected to be binary
		(0 or 1).

	Notes
	-----
	In ``__getitem__``, each window is converted from (W, C) to (C, W),
	matching PyTorch ``Conv1d`` input convention:
	``(batch, channels, length)``.
	"""

	def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
		if windows.ndim != 3:
			raise ValueError("windows must have shape (N, W, C)")
		if labels.ndim != 1:
			raise ValueError("labels must have shape (N,)")
		if windows.shape[0] != labels.shape[0]:
			raise ValueError("windows and labels must have the same first dimension")

		self.windows = windows
		self.labels = labels

	def __len__(self) -> int:
		"""Return the number of windows N in the dataset."""
		return self.windows.shape[0]

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return a single sample and label.

		Parameters
		----------
		idx : int
			Sample index in range ``[0, N)``.

		Returns
		-------
		x : torch.Tensor
			Window tensor with shape (C, W), dtype ``torch.float32``.
		y : torch.Tensor
			Label tensor with scalar shape ``()``, dtype ``torch.float32``.
		"""
		x_np = self.windows[idx]  # (W, C)
		y_np = self.labels[idx]

		x = torch.as_tensor(x_np, dtype=torch.float32).permute(1, 0)
		y = torch.as_tensor(y_np, dtype=torch.float32)
		return x, y


def create_dataloader(windows, labels, batch_size=32, shuffle=True):
	"""Create a PyTorch DataLoader from windowed arrays.

	Parameters
	----------
	windows : numpy.ndarray
		Input windows of shape (N, W, C).
	labels : numpy.ndarray
		Labels of shape (N,).
	batch_size : int, optional
		Number of samples per batch (default 32).
	shuffle : bool, optional
		Whether to shuffle dataset order each epoch (default True).

	Returns
	-------
	torch.utils.data.DataLoader
		DataLoader yielding batches in format:
		- ``x_batch``: shape (B, C, W)
		- ``y_batch``: shape (B,)
	"""
	dataset = TimeSeriesDataset(windows, labels)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_class_weights(labels):
	"""Compute positive class weight for weighted BCE loss.

	The returned weight follows:
	``pos_weight = n_negative / n_positive``

	Parameters
	----------
	labels : numpy.ndarray
		Binary labels of shape (N,).

	Returns
	-------
	torch.Tensor
		Scalar float tensor containing ``pos_weight``.

	Notes
	-----
	Prints class distribution and computed weight when called.
	"""
	labels = np.asarray(labels).reshape(-1)
	if labels.size == 0:
		raise ValueError("labels must not be empty")

	n_positive = int((labels == 1).sum())
	n_negative = int((labels == 0).sum())

	if n_positive == 0:
		pos_weight_value = 1.0
	else:
		pos_weight_value = float(n_negative / n_positive)

	print(
		"Class distribution: "
		f"negative={n_negative}, positive={n_positive}, "
		f"pos_weight={pos_weight_value:.6f}"
	)

	return torch.tensor(pos_weight_value, dtype=torch.float32)


def load_all_users_pooled(data_dir, n_users, window_size=128, stride=64):
	"""Load and concatenate all users' training windows.

	This utility is intended for centralized baseline training where all
	users contribute to a single pooled training set.

	Parameters
	----------
	data_dir : str
		Directory containing per-user ``.npy`` files.
	n_users : int
		Number of users to load, assuming IDs in range ``[0, n_users)``.
	window_size : int, optional
		Sliding window length W (default 128).
	stride : int, optional
		Sliding window stride (default 64).

	Returns
	-------
	X_train_all : numpy.ndarray
		Concatenated training windows of shape (N_total, W, C).
	y_train_all : numpy.ndarray
		Concatenated training labels of shape (N_total,).

	Notes
	-----
	Prints total window count and class balance after pooling.
	"""
	if n_users <= 0:
		raise ValueError("n_users must be positive")

	all_windows = []
	all_labels = []

	for user_id in range(n_users):
		user_data = load_user_data(
			user_id=user_id,
			data_dir=data_dir,
			window_size=window_size,
			stride=stride,
		)
		all_windows.append(user_data["X_train"])
		all_labels.append(user_data["y_train"])

	X_train_all = np.concatenate(all_windows, axis=0)
	y_train_all = np.concatenate(all_labels, axis=0)

	total_windows = int(y_train_all.shape[0])
	n_positive = int((y_train_all == 1).sum())
	n_negative = int((y_train_all == 0).sum())
	positive_rate = float(y_train_all.mean()) if total_windows > 0 else 0.0

	print(
		"Pooled training data: "
		f"total_windows={total_windows}, "
		f"negative={n_negative}, positive={n_positive}, "
		f"positive_rate={positive_rate:.4f}"
	)

	return X_train_all, y_train_all


if __name__ == "__main__":
	DATA_DIR = "data/raw/synthetic"

	X_train_all, y_train_all = load_all_users_pooled(
		data_dir=DATA_DIR,
		n_users=3,
		window_size=128,
		stride=64,
	)
	loader = create_dataloader(X_train_all, y_train_all, batch_size=32, shuffle=True)

	x_batch, y_batch = next(iter(loader))
	print(f"One batch X shape: {tuple(x_batch.shape)}")
	print(f"One batch y shape: {tuple(y_batch.shape)}")
