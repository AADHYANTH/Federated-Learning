"""Plotting utilities for per-user evaluation results."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_per_user_roc(per_user_metrics, save_path):
	"""Plot one ROC curve per user and a bold mean ROC curve.

	Parameters
	----------
	per_user_metrics : list[dict]
		Per-user metric dictionaries containing user_id, roc_curve_fpr,
		and roc_curve_tpr.
	save_path : str
		Path to save output PNG file.
	"""
	if len(per_user_metrics) == 0:
		raise ValueError("per_user_metrics must not be empty")

	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	fig, ax = plt.subplots(figsize=(9, 7))

	mean_fpr = np.linspace(0.0, 1.0, 200)
	interp_tprs = []

	for metrics in per_user_metrics:
		uid = int(metrics["user_id"])
		fpr = np.asarray(metrics.get("roc_curve_fpr", np.array([0.0, 1.0])))
		tpr = np.asarray(metrics.get("roc_curve_tpr", np.array([0.0, 1.0])))

		order = np.argsort(fpr)
		fpr = fpr[order]
		tpr = tpr[order]

		ax.plot(fpr, tpr, lw=1.0, alpha=0.35, label=f"User {uid:03d}")

		fpr_unique, unique_idx = np.unique(fpr, return_index=True)
		tpr_unique = tpr[unique_idx]
		interp_tpr = np.interp(mean_fpr, fpr_unique, tpr_unique)
		interp_tpr[0] = 0.0
		interp_tpr[-1] = 1.0
		interp_tprs.append(interp_tpr)

	if len(interp_tprs) > 0:
		mean_tpr = np.mean(np.stack(interp_tprs, axis=0), axis=0)
		ax.plot(mean_fpr, mean_tpr, color="black", lw=3.0, label="Mean ROC")

	ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.0)
	ax.set_title("Per-User ROC Curves")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.05)
	ax.grid(alpha=0.2)

	fig.tight_layout()
	fig.savefig(save_path, dpi=150)
	plt.close(fig)


def plot_metric_distributions(per_user_metrics, save_path):
	"""Plot side-by-side histograms for ROC_AUC, TPR, and FPR.

	Parameters
	----------
	per_user_metrics : list[dict]
		Per-user metric dictionaries containing ROC_AUC, TPR, and FPR.
	save_path : str
		Path to save output PNG file.
	"""
	if len(per_user_metrics) == 0:
		raise ValueError("per_user_metrics must not be empty")

	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	roc_auc = np.array([m["ROC_AUC"] for m in per_user_metrics], dtype=float)
	tpr = np.array([m["TPR"] for m in per_user_metrics], dtype=float)
	fpr = np.array([m["FPR"] for m in per_user_metrics], dtype=float)

	fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

	axes[0].hist(roc_auc[np.isfinite(roc_auc)], bins=12, color="#1f77b4", alpha=0.8)
	axes[0].set_title("ROC_AUC Distribution")
	axes[0].set_xlabel("ROC_AUC")
	axes[0].set_ylabel("Count")

	axes[1].hist(tpr[np.isfinite(tpr)], bins=12, color="#2ca02c", alpha=0.8)
	axes[1].set_title("TPR Distribution")
	axes[1].set_xlabel("TPR")
	axes[1].set_ylabel("Count")

	axes[2].hist(fpr[np.isfinite(fpr)], bins=12, color="#d62728", alpha=0.8)
	axes[2].set_title("FPR Distribution")
	axes[2].set_xlabel("FPR")
	axes[2].set_ylabel("Count")

	for ax in axes:
		ax.grid(alpha=0.2)

	fig.tight_layout()
	fig.savefig(save_path, dpi=150)
	plt.close(fig)
