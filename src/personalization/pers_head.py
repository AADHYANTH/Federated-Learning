"""Personalized prediction head for user-specific adaptation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PersonalizedHead(nn.Module):
	"""Small user-specific head operating on 128-d encoder features.

	Architecture: Linear(128, 1) -> Sigmoid
	"""

	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(128, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		"""Forward pass for feature vectors.

		Parameters
		----------
		x : torch.Tensor
			Encoder features of shape (batch, 128).

		Returns
		-------
		torch.Tensor
			Probabilities of shape (batch,).
		"""
		out = self.fc(x)
		out = self.sigmoid(out)
		return out.squeeze(1)


def fit_personalized_head(
	encoder,
	X_train,
	y_train,
	X_val,
	y_val,
	epochs=50,
	lr=1e-3,
):
	"""Train a personalized head while keeping encoder frozen.

	Parameters
	----------
	encoder : torch.nn.Module
		Encoder model exposing an ``encode(x)`` method returning (batch, 128).
	X_train : numpy.ndarray
		Training windows with shape (N_train, W, C).
	y_train : numpy.ndarray
		Training labels with shape (N_train,).
	X_val : numpy.ndarray
		Validation windows with shape (N_val, W, C).
	y_val : numpy.ndarray
		Validation labels with shape (N_val,).
	epochs : int, optional
		Number of epochs for head training.
	lr : float, optional
		Learning rate for head optimizer.

	Returns
	-------
	tuple
		(fitted_head, val_loss_history)
		where val_loss_history is a list of float validation losses.
	"""
	device = next(encoder.parameters()).device

	for p in encoder.parameters():
		p.requires_grad = False
	encoder.eval()

	head = PersonalizedHead().to(device)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(head.parameters(), lr=float(lr))

	X_train_t = torch.as_tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
	y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
	X_val_t = torch.as_tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
	y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

	train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
	val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)

	val_loss_history = []

	for _ in range(int(epochs)):
		head.train()
		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			with torch.no_grad():
				feats = encoder.encode(x_batch)

			preds = head(feats)
			loss = criterion(preds, y_batch)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		head.eval()
		running_val_loss = 0.0
		n_val = 0
		with torch.no_grad():
			for x_batch, y_batch in val_loader:
				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device)
				feats = encoder.encode(x_batch)
				preds = head(feats)
				loss = criterion(preds, y_batch)
				bs = int(x_batch.shape[0])
				running_val_loss += float(loss.item()) * bs
				n_val += bs

		val_loss = running_val_loss / max(n_val, 1)
		val_loss_history.append(float(val_loss))

	return head, val_loss_history
