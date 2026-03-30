"""Base 1D CNN model for time-series anomaly detection."""

import torch
import torch.nn as nn


class CNNAnomalyDetector(nn.Module):
	"""1D CNN anomaly detector producing per-window anomaly probabilities.

	Parameters
	----------
	in_channels : int, optional
		Number of input channels C in input tensors of shape
		(batch, C, window_length). Default is 1.
	window_size : int, optional
		Window length W. Kept for interface clarity and downstream use.
		Default is 128.

	Notes
	-----
	Expected input shape: (batch, in_channels, window_length)
	Output shape: (batch,)
	"""

	def __init__(self, in_channels=1, window_size=128):
		super().__init__()
		self.in_channels = in_channels
		self.window_size = window_size

		self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
		self.bn1 = nn.BatchNorm1d(32)
		self.pool1 = nn.MaxPool1d(kernel_size=2)

		self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
		self.bn2 = nn.BatchNorm1d(64)
		self.pool2 = nn.MaxPool1d(kernel_size=2)

		self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm1d(128)
		self.global_pool = nn.AdaptiveAvgPool1d(1)

		self.relu = nn.ReLU(inplace=True)
		self.fc1 = nn.Linear(128, 64)
		self.dropout = nn.Dropout(p=0.3)
		self.fc2 = nn.Linear(64, 1)
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		"""Encode input windows into 128-d feature vectors.

		Parameters
		----------
		x : torch.Tensor
			Input tensor of shape (batch, in_channels, window_length).

		Returns
		-------
		torch.Tensor
			Feature tensor of shape (batch, 128), extracted before the
			classification head.
		"""
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		x = self.global_pool(x)

		features = torch.flatten(x, start_dim=1)
		return features

	def forward(self, x):
		"""Run inference and return anomaly probabilities.

		Parameters
		----------
		x : torch.Tensor
			Input tensor of shape (batch, in_channels, window_length).

		Returns
		-------
		torch.Tensor
			Probability tensor of shape (batch,), with values in [0, 1].
		"""
		features = self.encode(x)
		x = self.fc1(features)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return x.squeeze(1)


def get_model_summary(model, in_channels=1, window_size=128):
	"""Print trainable parameter count and intermediate output shapes.

	Parameters
	----------
	model : torch.nn.Module
		Model instance, expected to be CNNAnomalyDetector-compatible.
	in_channels : int, optional
		Number of channels for dummy input generation. Default is 1.
	window_size : int, optional
		Sequence length for dummy input generation. Default is 128.
	"""
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters: {total_params}")

	was_training = model.training
	model.eval()

	with torch.no_grad():
		x = torch.randn(1, in_channels, window_size)
		print(f"Input: {tuple(x.shape)}")

		x = model.conv1(x)
		print(f"Layer1 Conv1d: {tuple(x.shape)}")
		x = model.bn1(x)
		print(f"Layer1 BatchNorm1d: {tuple(x.shape)}")
		x = model.relu(x)
		print(f"Layer1 ReLU: {tuple(x.shape)}")
		x = model.pool1(x)
		print(f"Layer1 MaxPool1d: {tuple(x.shape)}")

		x = model.conv2(x)
		print(f"Layer2 Conv1d: {tuple(x.shape)}")
		x = model.bn2(x)
		print(f"Layer2 BatchNorm1d: {tuple(x.shape)}")
		x = model.relu(x)
		print(f"Layer2 ReLU: {tuple(x.shape)}")
		x = model.pool2(x)
		print(f"Layer2 MaxPool1d: {tuple(x.shape)}")

		x = model.conv3(x)
		print(f"Layer3 Conv1d: {tuple(x.shape)}")
		x = model.bn3(x)
		print(f"Layer3 BatchNorm1d: {tuple(x.shape)}")
		x = model.relu(x)
		print(f"Layer3 ReLU: {tuple(x.shape)}")
		x = model.global_pool(x)
		print(f"Layer3 AdaptiveAvgPool1d: {tuple(x.shape)}")

		x = torch.flatten(x, start_dim=1)
		print(f"Flatten: {tuple(x.shape)}")
		x = model.fc1(x)
		print(f"Head Linear(128,64): {tuple(x.shape)}")
		x = model.relu(x)
		print(f"Head ReLU: {tuple(x.shape)}")
		x = model.dropout(x)
		print(f"Head Dropout(0.3): {tuple(x.shape)}")
		x = model.fc2(x)
		print(f"Head Linear(64,1): {tuple(x.shape)}")
		x = model.sigmoid(x)
		print(f"Head Sigmoid: {tuple(x.shape)}")
		x = x.squeeze(1)
		print(f"Output: {tuple(x.shape)}")

	if was_training:
		model.train()


if __name__ == "__main__":
	model = CNNAnomalyDetector(in_channels=1, window_size=128)
	x = torch.randn(32, 1, 128)

	scores = model(x)
	features = model.encode(x)

	print(f"Output shape: {tuple(scores.shape)}")
	print(f"Expected output shape (32,): {tuple(scores.shape) == (32,)}")
	print(f"Feature shape: {tuple(features.shape)}")

	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total trainable parameters: {total_params}")
