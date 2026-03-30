"""Per-user decision threshold selection utilities."""

from __future__ import annotations

import numpy as np


class ThresholdSelector:
	"""Select and evaluate personalized decision thresholds."""

	@staticmethod
	def _confusion_rates(probabilities, labels, threshold):
		"""Compute TPR/FPR/FNR at a given threshold."""
		probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
		y = np.asarray(labels, dtype=np.int32).reshape(-1)

		y_hat = (probs >= float(threshold)).astype(np.int32)

		tp = int(((y_hat == 1) & (y == 1)).sum())
		tn = int(((y_hat == 0) & (y == 0)).sum())
		fp = int(((y_hat == 1) & (y == 0)).sum())
		fn = int(((y_hat == 0) & (y == 1)).sum())

		tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
		fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
		return float(tpr), float(fpr), float(fnr)

	def find_threshold(self, probabilities, labels, target_fpr=0.05):
		"""Find personalized threshold by sweeping from 0.01 to 0.99.

		Parameters
		----------
		probabilities : numpy.ndarray
			Predicted probabilities with shape (N,).
		labels : numpy.ndarray
			Binary labels with shape (N,).
		target_fpr : float, optional
			Desired upper bound on false positive rate.

		Returns
		-------
		dict
			Dictionary with keys: threshold, achieved_fpr, achieved_tpr.
		"""
		thresholds = np.arange(0.01, 1.00, 0.01)
		stats = []

		for thr in thresholds:
			tpr, fpr, _ = self._confusion_rates(probabilities, labels, thr)
			stats.append((float(thr), float(fpr), float(tpr)))

		valid = [s for s in stats if s[1] <= float(target_fpr)]

		if valid:
			# Smallest threshold meeting FPR target.
			chosen_thr, chosen_fpr, chosen_tpr = valid[0]
		else:
			# No threshold meets target: choose threshold with lowest FPR.
			chosen_thr, chosen_fpr, chosen_tpr = min(stats, key=lambda x: x[1])

		return {
			"threshold": float(chosen_thr),
			"achieved_fpr": float(chosen_fpr),
			"achieved_tpr": float(chosen_tpr),
		}

	def evaluate_at_threshold(self, probabilities, labels, threshold):
		"""Evaluate TPR/FPR/FNR at a fixed threshold.

		Parameters
		----------
		probabilities : numpy.ndarray
			Predicted probabilities with shape (N,).
		labels : numpy.ndarray
			Binary labels with shape (N,).
		threshold : float
			Decision threshold.

		Returns
		-------
		dict
			Dictionary with keys: TPR, FPR, FNR.
		"""
		tpr, fpr, fnr = self._confusion_rates(probabilities, labels, threshold)
		return {"TPR": tpr, "FPR": fpr, "FNR": fnr}
