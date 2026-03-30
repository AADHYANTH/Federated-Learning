"""Personalization experiment using score calibration and per-user thresholds."""

from __future__ import annotations

import os
import sys

import mlflow
import numpy as np
import pandas as pd
import torch


DATA_DIR = "data/raw/synthetic"
MODEL_PATH = "data/processed/global_model_federated.pt"
N_USERS = 50
WINDOW_SIZE = 128
STRIDE = 64
TARGET_FPR = 0.05
RESULTS_PATH = "data/processed/personalization_results.csv"


try:
	from src.data.preprocessing import load_user_data
	from src.evaluation.metrics import compute_metrics
	from src.models.base_cnn import CNNAnomalyDetector
	from src.personalization.calibration import ScoreCalibrator
	from src.personalization.thresholds import ThresholdSelector
except ModuleNotFoundError:
	current_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(current_dir, ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from src.data.preprocessing import load_user_data
	from src.evaluation.metrics import compute_metrics
	from src.models.base_cnn import CNNAnomalyDetector
	from src.personalization.calibration import ScoreCalibrator
	from src.personalization.thresholds import ThresholdSelector


def _predict_probs(model, windows, device):
	"""Run frozen model inference on windows and return probabilities."""
	x = torch.as_tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
	with torch.no_grad():
		probs = model(x).detach().cpu().numpy().reshape(-1)
	return probs


def run_personalization_eval():
	"""Run per-user personalization and save/report aggregate results."""
	os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = CNNAnomalyDetector(in_channels=1, window_size=128)
	state_dict = torch.load(MODEL_PATH, map_location=device)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()

	selector = ThresholdSelector()
	rows = []
	skipped_users = 0

	mlflow.set_experiment("personalization_evaluation")
	with mlflow.start_run(run_name="personalization_eval"):
		mlflow.log_params(
			{
				"data_dir": DATA_DIR,
				"model_path": MODEL_PATH,
				"n_users": N_USERS,
				"window_size": WINDOW_SIZE,
				"stride": STRIDE,
				"target_fpr": TARGET_FPR,
			}
		)

		for user_id in range(N_USERS):
			user_data = load_user_data(
				user_id=user_id,
				data_dir=DATA_DIR,
				window_size=WINDOW_SIZE,
				stride=STRIDE,
			)

			X_val = user_data["X_val"]
			y_val = user_data["y_val"]
			X_test = user_data["X_test"]
			y_test = user_data["y_test"]

			base_scores_val = _predict_probs(model, X_val, device)
			global_scores_test = _predict_probs(model, X_test, device)

			calibrator = ScoreCalibrator()
			calibrator.fit(base_scores_val, y_val, lr=0.01, epochs=100)

			val_calibrated = calibrator.predict(base_scores_val)
			thr_info = selector.find_threshold(
				val_calibrated,
				y_val,
				target_fpr=TARGET_FPR,
			)
			tau_i = float(thr_info["threshold"])

			global_metrics = compute_metrics(y_test, global_scores_test)

			calibrated_scores_test = calibrator.predict(global_scores_test)
			personal_metrics = compute_metrics(
				y_test,
				calibrated_scores_test,
				threshold=tau_i,
			)

			# Skip users with NaN metrics (typically undefined ROC_AUC).
			if (
				np.isnan(global_metrics["ROC_AUC"])
				or np.isnan(personal_metrics["ROC_AUC"])
			):
				skipped_users += 1
				print(f"Skipping user {user_id}: NaN metric encountered")
				continue

			delta_tpr = float(personal_metrics["TPR"] - global_metrics["TPR"])
			delta_fpr = float(personal_metrics["FPR"] - global_metrics["FPR"])

			rows.append(
				{
					"user_id": int(user_id),
					"global_TPR": float(global_metrics["TPR"]),
					"global_FPR": float(global_metrics["FPR"]),
					"personal_TPR": float(personal_metrics["TPR"]),
					"personal_FPR": float(personal_metrics["FPR"]),
					"delta_TPR": delta_tpr,
					"delta_FPR": delta_fpr,
					"a_i": float(calibrator.a_i),
					"b_i": float(calibrator.b_i),
					"tau_i": tau_i,
					"global_ROC_AUC": float(global_metrics["ROC_AUC"]),
					"personal_ROC_AUC": float(personal_metrics["ROC_AUC"]),
				}
			)

			mlflow.log_metric("global_ROC_AUC", float(global_metrics["ROC_AUC"]), step=user_id)
			mlflow.log_metric("personal_ROC_AUC", float(personal_metrics["ROC_AUC"]), step=user_id)
			mlflow.log_metric("delta_TPR", delta_tpr, step=user_id)
			mlflow.log_metric("delta_FPR", delta_fpr, step=user_id)

		if not rows:
			raise RuntimeError("All users were skipped due to NaN metrics")

		df = pd.DataFrame(rows)

		# Save required columns only.
		df_out = df[
			[
				"user_id",
				"global_TPR",
				"global_FPR",
				"personal_TPR",
				"personal_FPR",
				"delta_TPR",
				"delta_FPR",
				"a_i",
				"b_i",
				"tau_i",
			]
		]
		df_out.to_csv(RESULTS_PATH, index=False)
		mlflow.log_artifact(RESULTS_PATH)

		users_improved_tpr = int((df["delta_TPR"] > 0.0).sum())
		users_reduced_fpr = int((df["delta_FPR"] < 0.0).sum())
		mean_delta_tpr = float(df["delta_TPR"].mean())
		mean_delta_fpr = float(df["delta_FPR"].mean())
		mean_global_auc = float(df["global_ROC_AUC"].mean())
		mean_personal_auc = float(df["personal_ROC_AUC"].mean())

		print("=== PERSONALIZATION RESULTS ===")
		print(f"Users improved TPR: {users_improved_tpr}/50")
		print(f"Users reduced FPR: {users_reduced_fpr}/50")
		print(f"Mean delta_TPR: {mean_delta_tpr:+.4f}")
		print(f"Mean delta_FPR: {mean_delta_fpr:+.4f}")
		print(f"Mean global ROC_AUC: {mean_global_auc:.4f}")
		print(f"Mean personal ROC_AUC: {mean_personal_auc:.4f}")
		print(f"Skipped users due to NaN metrics: {skipped_users}")

		mlflow.log_metric("users_improved_tpr", users_improved_tpr)
		mlflow.log_metric("users_reduced_fpr", users_reduced_fpr)
		mlflow.log_metric("mean_delta_TPR", mean_delta_tpr)
		mlflow.log_metric("mean_delta_FPR", mean_delta_fpr)
		mlflow.log_metric("mean_global_ROC_AUC", mean_global_auc)
		mlflow.log_metric("mean_personal_ROC_AUC", mean_personal_auc)
		mlflow.log_metric("skipped_users", skipped_users)


if __name__ == "__main__":
	run_personalization_eval()
