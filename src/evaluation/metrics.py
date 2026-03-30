"""Evaluation metrics and per-user model evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.data.preprocessing import load_user_data


def _safe_divide(numerator, denominator):
	"""Safely divide two scalars, returning 0.0 when denominator is zero."""
	if denominator == 0:
		return 0.0
	return float(numerator / denominator)


def compute_metrics(y_true, y_scores, threshold=0.5):
	"""Compute binary classification metrics from true labels and scores.

	Parameters
	----------
	y_true : array-like
		Ground-truth binary labels with shape (N,).
	y_scores : array-like
		Predicted anomaly probabilities with shape (N,).
	threshold : float, optional
		Threshold used to binarize probabilities into predictions.

	Returns
	-------
	dict
		Dictionary with keys: TPR, FPR, FNR, precision, F1, ROC_AUC, PR_AUC.
	"""
	y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
	y_scores = np.asarray(y_scores).astype(np.float64).reshape(-1)

	if y_true.shape[0] != y_scores.shape[0]:
		raise ValueError("y_true and y_scores must have matching length")
	if y_true.size == 0:
		raise ValueError("y_true and y_scores must not be empty")

	y_pred = (y_scores >= threshold).astype(np.int32)
	unique_classes = np.unique(y_true)

	tp = int(((y_pred == 1) & (y_true == 1)).sum())
	tn = int(((y_pred == 0) & (y_true == 0)).sum())
	fp = int(((y_pred == 1) & (y_true == 0)).sum())
	fn = int(((y_pred == 0) & (y_true == 1)).sum())

	tp_r = _safe_divide(tp, tp + fn)
	fp_r = _safe_divide(fp, fp + tn)
	fn_r = _safe_divide(fn, fn + tp)
	precision = _safe_divide(tp, tp + fp)
	f1 = _safe_divide(2.0 * precision * tp_r, precision + tp_r)

	if unique_classes.size < 2:
		roc_auc = float("nan")
		# If there are no positives, AP is 0. If all are positives, AP is 1.
		pr_auc = 0.0 if int(y_true.sum()) == 0 else 1.0
	else:
		roc_auc = float(roc_auc_score(y_true, y_scores))
		pr_auc = float(average_precision_score(y_true, y_scores))

	return {
		"TPR": tp_r,
		"FPR": fp_r,
		"FNR": fn_r,
		"precision": precision,
		"F1": f1,
		"ROC_AUC": roc_auc,
		"PR_AUC": pr_auc,
	}


def evaluate_model_per_user(model, data_dir, n_users, window_size, stride, device="cpu"):
	"""Evaluate a trained model on each user's test split.

	Parameters
	----------
	model : torch.nn.Module
		Trained anomaly detector returning probabilities of shape (batch,).
	data_dir : str
		Directory containing user signal/label files.
	n_users : int
		Number of users to evaluate (IDs 0 to n_users-1).
	window_size : int
		Window size used during preprocessing.
	stride : int
		Stride used during preprocessing.
	device : str, optional
		Device string passed to torch.device, default is "cpu".

	Returns
	-------
	list[dict]
		One dictionary per user containing user_id and metrics.
		Each entry also includes roc_curve_fpr and roc_curve_tpr for plotting.
	"""
	device = torch.device(device)
	model = model.to(device)
	model.eval()

	per_user_metrics = []

	with torch.no_grad():
		for user_id in range(n_users):
			user_data = load_user_data(
				user_id=user_id,
				data_dir=data_dir,
				window_size=window_size,
				stride=stride,
			)

			X_test = user_data["X_test"]
			y_test = user_data["y_test"].astype(np.int32)

			x_tensor = torch.as_tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
			y_scores = model(x_tensor).detach().cpu().numpy().reshape(-1)

			metrics = compute_metrics(y_true=y_test, y_scores=y_scores, threshold=0.5)

			entry = {"user_id": user_id, **metrics}

			if np.unique(y_test).size < 2:
				entry["roc_curve_fpr"] = np.array([0.0, 1.0])
				entry["roc_curve_tpr"] = np.array([0.0, 1.0])
			else:
				fpr, tpr, _ = roc_curve(y_test, y_scores)
				entry["roc_curve_fpr"] = fpr
				entry["roc_curve_tpr"] = tpr

			per_user_metrics.append(entry)

	return per_user_metrics


def summarize_results(per_user_metrics):
	"""Print per-user table and summary statistics across users.

	Parameters
	----------
	per_user_metrics : list[dict]
		List of per-user metric dictionaries from evaluate_model_per_user().

	Returns
	-------
	pandas.DataFrame
		DataFrame containing one row per user and all metric columns.
	"""
	if len(per_user_metrics) == 0:
		raise ValueError("per_user_metrics must not be empty")

	df = pd.DataFrame(per_user_metrics)
	metric_cols = ["TPR", "FPR", "FNR", "precision", "F1", "ROC_AUC", "PR_AUC"]
	df_view = df[["user_id", *metric_cols]].copy()

	print("user_id | ROC_AUC | TPR | FPR | FNR")
	print("-" * 40)
	for _, row in df_view.iterrows():
		print(
			f"{int(row['user_id']):03d} | "
			f"{row['ROC_AUC']:.4f} | {row['TPR']:.4f} | "
			f"{row['FPR']:.4f} | {row['FNR']:.4f}"
		)

	print("\nMetric summary across users:")
	for metric in metric_cols:
		mean_v = float(np.nanmean(df_view[metric].to_numpy(dtype=float)))
		std_v = float(np.nanstd(df_view[metric].to_numpy(dtype=float)))
		print(f"{metric}: mean={mean_v:.4f}, std={std_v:.4f}")

	return df_view
