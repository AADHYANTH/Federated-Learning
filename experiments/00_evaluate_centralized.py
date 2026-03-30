"""Evaluate centralized model on per-user test splits and log results."""

from __future__ import annotations

import os
import sys

import mlflow
import torch


DATA_DIR = "data/raw/synthetic"
MODEL_PATH = "data/processed/global_model_centralized.pt"
PLOTS_DIR = "data/processed/plots"
N_USERS = 50
WINDOW_SIZE = 128
STRIDE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


try:
	from src.evaluation.metrics import evaluate_model_per_user, summarize_results
	from src.evaluation.plots import plot_metric_distributions, plot_per_user_roc
	from src.models.base_cnn import CNNAnomalyDetector
except ModuleNotFoundError:
	# Allow running as: python experiments/00_evaluate_centralized.py
	current_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(current_dir, ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from src.evaluation.metrics import evaluate_model_per_user, summarize_results
	from src.evaluation.plots import plot_metric_distributions, plot_per_user_roc
	from src.models.base_cnn import CNNAnomalyDetector


def run_evaluation():
	"""Run centralized model evaluation, plots, and MLflow logging."""
	os.makedirs(PLOTS_DIR, exist_ok=True)

	model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE)
	state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
	model.load_state_dict(state_dict)
	model.to(DEVICE)
	model.eval()

	per_user_metrics = evaluate_model_per_user(
		model=model,
		data_dir=DATA_DIR,
		n_users=N_USERS,
		window_size=WINDOW_SIZE,
		stride=STRIDE,
		device=DEVICE,
	)

	df = summarize_results(per_user_metrics)

	roc_plot_path = os.path.join(PLOTS_DIR, "per_user_roc.png")
	dist_plot_path = os.path.join(PLOTS_DIR, "metric_distributions.png")
	plot_per_user_roc(per_user_metrics, roc_plot_path)
	plot_metric_distributions(per_user_metrics, dist_plot_path)
	print(f"Saved ROC plot to: {roc_plot_path}")
	print(f"Saved metric distribution plot to: {dist_plot_path}")

	mlflow.set_experiment("centralized_evaluation")
	with mlflow.start_run(run_name="centralized_model_eval"):
		mlflow.log_params(
			{
				"data_dir": DATA_DIR,
				"model_path": MODEL_PATH,
				"n_users": N_USERS,
				"window_size": WINDOW_SIZE,
				"stride": STRIDE,
				"device": DEVICE,
			}
		)

		for metrics in per_user_metrics:
			uid = int(metrics["user_id"])
			mlflow.log_metric("TPR", float(metrics["TPR"]), step=uid)
			mlflow.log_metric("FPR", float(metrics["FPR"]), step=uid)
			mlflow.log_metric("FNR", float(metrics["FNR"]), step=uid)
			mlflow.log_metric("precision", float(metrics["precision"]), step=uid)
			mlflow.log_metric("F1", float(metrics["F1"]), step=uid)
			mlflow.log_metric("ROC_AUC", float(metrics["ROC_AUC"]), step=uid)
			mlflow.log_metric("PR_AUC", float(metrics["PR_AUC"]), step=uid)

		mlflow.log_artifact(roc_plot_path)
		mlflow.log_artifact(dist_plot_path)

	print("MLflow logging complete.")


if __name__ == "__main__":
	run_evaluation()
