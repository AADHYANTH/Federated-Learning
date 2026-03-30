"""Flower client implementation for federated anomaly detection."""

from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from src.data.datasets import create_dataloader, get_class_weights
from src.data.preprocessing import load_user_data
from src.models.base_cnn import CNNAnomalyDetector


DEFAULT_DATA_DIR = "data/raw/synthetic"
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOCAL_EPOCHS = 3
DEFAULT_DEVICE = "cpu"


def _weighted_bce_loss(probs, targets, pos_weight):
	"""Compute weighted binary cross-entropy for probability outputs.

	Parameters
	----------
	probs : torch.Tensor
		Predicted probabilities of shape (B,).
	targets : torch.Tensor
		Binary target tensor of shape (B,).
	pos_weight : torch.Tensor
		Scalar positive class weight equal to n_negative / n_positive.

	Returns
	-------
	torch.Tensor
		Scalar loss value.
	"""
	eps = 1e-8
	probs = torch.clamp(probs, eps, 1.0 - eps)
	loss_pos = -pos_weight * targets * torch.log(probs)
	loss_neg = -(1.0 - targets) * torch.log(1.0 - probs)
	return (loss_pos + loss_neg).mean()


class AnomalyDetectionClient(fl.client.NumPyClient):
	"""Federated Flower client for one user's anomaly detection model.

	Parameters
	----------
	user_id : int
		User/client identifier.
	data_dir : str
		Directory containing per-user synthetic signal/label files.
	window_size : int, optional
		Sliding window size used by preprocessing.
	stride : int, optional
		Sliding window stride used by preprocessing.
	batch_size : int, optional
		Batch size for local training and validation.
	local_epochs : int, optional
		Number of local epochs per federated round.
	device : str, optional
		Compute device string, e.g. "cpu" or "cuda".
	"""

	def __init__(
		self,
		user_id,
		data_dir="data/raw/synthetic",
		window_size=128,
		stride=64,
		batch_size=32,
		local_epochs=3,
		device="cpu",
	):
		self.user_id = int(user_id)
		self.data_dir = data_dir
		self.window_size = int(window_size)
		self.stride = int(stride)
		self.batch_size = int(batch_size)
		self.local_epochs = int(local_epochs)
		self.device = torch.device(device)

		user_data = load_user_data(
			user_id=self.user_id,
			data_dir=self.data_dir,
			window_size=self.window_size,
			stride=self.stride,
		)

		self.X_train = user_data["X_train"]
		self.y_train = user_data["y_train"]
		self.X_val = user_data["X_val"]
		self.y_val = user_data["y_val"]

		self.train_loader = create_dataloader(
			self.X_train, self.y_train, batch_size=self.batch_size, shuffle=True
		)
		self.val_loader = create_dataloader(
			self.X_val, self.y_val, batch_size=self.batch_size, shuffle=False
		)

		self.pos_weight = get_class_weights(self.y_train).to(self.device)

		self.model = CNNAnomalyDetector(in_channels=1, window_size=128).to(self.device)

	def get_parameters(self, config):
		"""Return model parameters as a list of NumPy arrays.

		Parameters
		----------
		config : dict
			Flower-provided configuration dictionary.

		Returns
		-------
		list[numpy.ndarray]
			Current model weights in state_dict order.
		"""
		del config
		return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

	def set_parameters(self, parameters):
		"""Load model parameters from a list of NumPy arrays.

		Parameters
		----------
		parameters : list[numpy.ndarray]
			Model weights in state_dict order.
		"""
		state_dict = self.model.state_dict()
		keys = list(state_dict.keys())
		new_state_dict = OrderedDict()

		for key, value in zip(keys, parameters):
			tensor = torch.tensor(
				value,
				dtype=state_dict[key].dtype,
				device=self.device,
			)
			new_state_dict[key] = tensor

		self.model.load_state_dict(new_state_dict, strict=True)

	def fit(self, parameters, config):
		"""Perform local training on this client's training windows.

		This method first loads global model parameters, then runs
		`local_epochs` of local optimization using weighted BCE loss.

		Parameters
		----------
		parameters : list[numpy.ndarray]
			Global model parameters sent by the server.
		config : dict
			Round configuration. Supports optional key ``lr``.

		Returns
		-------
		tuple
			(updated_parameters, num_train_examples, metrics_dict)
			where metrics_dict contains ``train_loss``.
		"""
		self.set_parameters(parameters)

		lr = float(config.get("lr", 1e-3))
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		self.model.train()
		epoch_losses = []

		for _ in range(self.local_epochs):
			running_loss = 0.0
			running_examples = 0

			for x_batch, y_batch in self.train_loader:
				x_batch = x_batch.to(self.device)
				y_batch = y_batch.to(self.device)

				optimizer.zero_grad()
				probs = self.model(x_batch)
				loss = _weighted_bce_loss(probs, y_batch, self.pos_weight)
				loss.backward()
				optimizer.step()

				batch_size = int(x_batch.shape[0])
				running_loss += float(loss.item()) * batch_size
				running_examples += batch_size

			epoch_loss = running_loss / max(running_examples, 1)
			epoch_losses.append(epoch_loss)

		train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

		return (
			self.get_parameters(config={}),
			int(self.X_train.shape[0]),
			{"train_loss": train_loss},
		)

	def evaluate(self, parameters, config):
		"""Evaluate the model on this client's local validation split.

		Parameters
		----------
		parameters : list[numpy.ndarray]
			Global model parameters sent by the server.
		config : dict
			Flower-provided configuration dictionary.

		Returns
		-------
		tuple
			(val_loss, num_val_examples, metrics_dict)
			where metrics_dict contains ``val_loss`` and ``ROC_AUC``.
		"""
		del config
		self.set_parameters(parameters)
		self.model.eval()

		total_loss = 0.0
		total_examples = 0
		all_scores = []
		all_labels = []

		with torch.no_grad():
			for x_batch, y_batch in self.val_loader:
				x_batch = x_batch.to(self.device)
				y_batch = y_batch.to(self.device)

				probs = self.model(x_batch)
				loss = _weighted_bce_loss(probs, y_batch, self.pos_weight)

				batch_size = int(x_batch.shape[0])
				total_loss += float(loss.item()) * batch_size
				total_examples += batch_size

				all_scores.append(probs.detach().cpu().numpy())
				all_labels.append(y_batch.detach().cpu().numpy())

		val_loss = total_loss / max(total_examples, 1)

		y_scores = np.concatenate(all_scores, axis=0) if all_scores else np.array([])
		y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

		if y_true.size == 0 or np.unique(y_true).size < 2:
			roc_auc = 0.0
		else:
			roc_auc = float(roc_auc_score(y_true, y_scores))

		return float(val_loss), int(self.X_val.shape[0]), {
			"val_loss": float(val_loss),
			"ROC_AUC": roc_auc,
		}


def client_fn(user_id):
	"""Create a Flower client instance for a given user ID.

	Parameters
	----------
	user_id : int
		User/client identifier.

	Returns
	-------
	AnomalyDetectionClient
		Configured client ready for Flower simulation.
	"""
	return AnomalyDetectionClient(
		user_id=int(user_id),
		data_dir=DEFAULT_DATA_DIR,
		window_size=DEFAULT_WINDOW_SIZE,
		stride=DEFAULT_STRIDE,
		batch_size=DEFAULT_BATCH_SIZE,
		local_epochs=DEFAULT_LOCAL_EPOCHS,
		device=DEFAULT_DEVICE,
	)
