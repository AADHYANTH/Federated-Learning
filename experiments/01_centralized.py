"""Centralized training experiment for pooled synthetic users.

This script trains a global CNN anomaly detector using pooled user windows,
logs training with MLflow, and saves the best model by validation loss.
"""

from __future__ import annotations

import os
import sys

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Configuration constants
DATA_DIR = "data/raw/synthetic"
N_USERS = 50
WINDOW_SIZE = 128
STRIDE = 64
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
MODEL_SAVE_PATH = "data/processed/global_model_centralized.pt"


try:
	from src.data.datasets import (
		create_dataloader,
		get_class_weights,
		load_all_users_pooled,
	)
	from src.data.preprocessing import load_user_data
	from src.models.base_cnn import CNNAnomalyDetector
except ModuleNotFoundError:
	# Allow running from the experiments folder as a direct script.
	current_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(current_dir, ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from src.data.datasets import (
		create_dataloader,
		get_class_weights,
		load_all_users_pooled,
	)
	from src.data.preprocessing import load_user_data
	from src.models.base_cnn import CNNAnomalyDetector


def weighted_bce_loss(probs, targets, pos_weight):
	"""Compute weighted BCE using probability outputs.

	Parameters
	----------
	probs : torch.Tensor
		Predicted probabilities after sigmoid, shape (B,).
	targets : torch.Tensor
		Binary target labels, shape (B,).
	pos_weight : torch.Tensor
		Scalar tensor with positive class weight n_negative / n_positive.

	Returns
	-------
	torch.Tensor
		Scalar weighted BCE loss.
	"""
	eps = 1e-8
	probs = torch.clamp(probs, eps, 1.0 - eps)
	loss_pos = -pos_weight * targets * torch.log(probs)
	loss_neg = -(1.0 - targets) * torch.log(1.0 - probs)
	return (loss_pos + loss_neg).mean()


def load_all_users_val_pooled(data_dir, n_users, window_size=128, stride=64):
	"""Load and concatenate validation windows for all users.

	Parameters
	----------
	data_dir : str
		Directory containing per-user signal/label files.
	n_users : int
		Number of user IDs to load from [0, n_users).
	window_size : int, optional
		Sliding window length.
	stride : int, optional
		Sliding window stride.

	Returns
	-------
	tuple[numpy.ndarray, numpy.ndarray]
		Pooled validation windows and labels.
	"""
	all_windows = []
	all_labels = []

	for user_id in range(n_users):
		user_data = load_user_data(
			user_id=user_id,
			data_dir=data_dir,
			window_size=window_size,
			stride=stride,
		)
		all_windows.append(user_data["X_val"])
		all_labels.append(user_data["y_val"])

	X_val_all = np.concatenate(all_windows, axis=0)
	y_val_all = np.concatenate(all_labels, axis=0)
	return X_val_all, y_val_all


def run_epoch(model, dataloader, pos_weight, optimizer, device, training=True):
	"""Run one training or validation epoch and return mean loss."""
	if training:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	total_samples = 0

	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for x_batch, y_batch in dataloader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			if training:
				optimizer.zero_grad()

			probs = model(x_batch)
			loss = weighted_bce_loss(probs, y_batch, pos_weight)

			if training:
				loss.backward()
				optimizer.step()

			batch_size = x_batch.shape[0]
			total_loss += loss.item() * batch_size
			total_samples += batch_size

	if total_samples == 0:
		return 0.0
	return total_loss / total_samples


def train():
	"""Train centralized global model with pooled users and MLflow logging."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

	print("Loading pooled training data...")
	X_train_all, y_train_all = load_all_users_pooled(
		data_dir=DATA_DIR,
		n_users=N_USERS,
		window_size=WINDOW_SIZE,
		stride=STRIDE,
	)

	print("Loading pooled validation data...")
	X_val_all, y_val_all = load_all_users_val_pooled(
		data_dir=DATA_DIR,
		n_users=N_USERS,
		window_size=WINDOW_SIZE,
		stride=STRIDE,
	)

	train_loader = create_dataloader(
		X_train_all, y_train_all, batch_size=BATCH_SIZE, shuffle=True
	)
	val_loader = create_dataloader(
		X_val_all, y_val_all, batch_size=BATCH_SIZE, shuffle=False
	)

	pos_weight = get_class_weights(y_train_all).to(device)

	in_channels = int(X_train_all.shape[2])
	model = CNNAnomalyDetector(in_channels=in_channels, window_size=WINDOW_SIZE).to(device)
	optimizer = optim.Adam(model.parameters(), lr=LR)

	best_val_loss = float("inf")
	final_train_loss = float("nan")
	final_val_loss = float("nan")

	mlflow.set_experiment("centralized_anomaly_detection")
	with mlflow.start_run(run_name="centralized_cnn"):
		mlflow.log_params(
			{
				"data_dir": DATA_DIR,
				"n_users": N_USERS,
				"window_size": WINDOW_SIZE,
				"stride": STRIDE,
				"batch_size": BATCH_SIZE,
				"epochs": EPOCHS,
				"lr": LR,
				"model_save_path": MODEL_SAVE_PATH,
				"model_name": "CNNAnomalyDetector",
				"optimizer": "Adam",
				"loss": "WeightedBCE",
				"device": str(device),
				"pos_weight": float(pos_weight.item()),
			}
		)

		for epoch in range(1, EPOCHS + 1):
			train_loss = run_epoch(
				model=model,
				dataloader=train_loader,
				pos_weight=pos_weight,
				optimizer=optimizer,
				device=device,
				training=True,
			)
			val_loss = run_epoch(
				model=model,
				dataloader=val_loader,
				pos_weight=pos_weight,
				optimizer=optimizer,
				device=device,
				training=False,
			)

			final_train_loss = train_loss
			final_val_loss = val_loss

			print(
				f"Epoch {epoch:02d}/{EPOCHS} | "
				f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
			)

			mlflow.log_metric("train_loss", train_loss, step=epoch)
			mlflow.log_metric("val_loss", val_loss, step=epoch)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				torch.save(model.state_dict(), MODEL_SAVE_PATH)
				print(f"Saved new best model (val_loss={best_val_loss:.6f})")

		mlflow.log_metric("best_val_loss", best_val_loss)
		mlflow.log_artifact(MODEL_SAVE_PATH)

	print(f"Final train loss: {final_train_loss:.6f}")
	print(f"Final val loss: {final_val_loss:.6f}")
	print(f"Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
	train()
