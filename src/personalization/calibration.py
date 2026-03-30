"""Per-user score calibration for anomaly probabilities."""

from __future__ import annotations

import numpy as np


class ScoreCalibrator:
	"""Per-user affine score calibrator with sigmoid output.

	The calibrator applies:
	p = sigmoid(a * score + b)

	Parameters
	----------
	a : float, optional
		Initial scale parameter.
	b : float, optional
		Initial bias parameter.
	"""

	def __init__(self):
		self.a_i = 1.0
		self.b_i = 0.0

	@staticmethod
	def _sigmoid(x):
		"""Numerically stable sigmoid."""
		x = np.clip(x, -50.0, 50.0)
		return 1.0 / (1.0 + np.exp(-x))

	def fit(self, base_scores, labels, lr=0.01, epochs=100):
		"""Fit calibration parameters on per-user labeled scores.

		Parameters
		----------
		base_scores : numpy.ndarray
			Model scores/probabilities with shape (N,).
		labels : numpy.ndarray
			Binary labels with shape (N,).
		lr : float, optional
			Gradient descent learning rate.
		epochs : int, optional
			Number of optimization epochs.
		"""
		scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
		y = np.asarray(labels, dtype=np.float64).reshape(-1)

		if scores.shape[0] != y.shape[0]:
			raise ValueError("base_scores and labels must have matching length")
		if scores.size == 0:
			raise ValueError("base_scores and labels must not be empty")

		eps = 1e-8
		n = scores.shape[0]

		for _ in range(int(epochs)):
			z = self.a_i * scores + self.b_i
			p = self._sigmoid(z)

			# Gradient of BCE wrt z for sigmoid outputs is (p - y).
			dz = p - y
			grad_a = np.sum(dz * scores) / n
			grad_b = np.sum(dz) / n

			self.a_i -= float(lr) * float(grad_a)
			self.b_i -= float(lr) * float(grad_b)

		final_p = self._sigmoid(self.a_i * scores + self.b_i)
		final_loss = -np.mean(
			y * np.log(final_p + eps) + (1.0 - y) * np.log(1.0 - final_p + eps)
		)
		print(
			f"Calibration complete. "
			f"a={self.a_i:.4f}, b={self.b_i:.4f}, loss={final_loss:.4f}"
		)

	def predict(self, base_scores):
		"""Predict calibrated probabilities from base scores.

		Parameters
		----------
		base_scores : numpy.ndarray
			Base model scores with shape (N,).

		Returns
		-------
		numpy.ndarray
			Calibrated probabilities with shape (N,).
		"""
		scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
		return self._sigmoid(self.a_i * scores + self.b_i)

	def save(self, path):
		"""Save calibrator parameters to a .npy file.

		Parameters
		----------
		path : str
			Destination file path.
		"""
		np.save(path, {"a": float(self.a_i), "b": float(self.b_i)}, allow_pickle=True)

	def load(self, path):
		"""Load calibrator parameters from a .npy file.

		Parameters
		----------
		path : str
			Path to .npy file produced by save().
		"""
		data = np.load(path, allow_pickle=True).item()
		self.a_i = float(data["a"])
		self.b_i = float(data["b"])
