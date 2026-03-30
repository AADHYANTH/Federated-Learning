"""Federated training experiment using Flower simulation and custom FedAvg."""

from __future__ import annotations

import glob
import os
import shutil
import sys

import flwr as fl
import mlflow
import torch


DATA_DIR = "data/raw/synthetic"
N_USERS = 50
N_ROUNDS = 20
FRACTION_FIT = 0.4
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
WINDOW_SIZE = 128
STRIDE = 64
MODEL_SAVE_PATH = "data/processed/global_model_federated.pt"

CHECKPOINT_DIR = "data/processed/checkpoints"


try:
	from src.federated.client import AnomalyDetectionClient
	from src.federated.strategy import CheckpointFedAvg
	from src.models.base_cnn import CNNAnomalyDetector
except ModuleNotFoundError:
	# Allow running as: python experiments/02_federated.py
	current_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(current_dir, ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from src.federated.client import AnomalyDetectionClient
	from src.federated.strategy import CheckpointFedAvg
	from src.models.base_cnn import CNNAnomalyDetector


def _build_initial_parameters(model):
	"""Convert a PyTorch model state to Flower Parameters."""
	ndarrays = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
	return fl.common.ndarrays_to_parameters(ndarrays)


def _make_client_fn(data_dir, window_size, stride, batch_size, local_epochs):
	"""Create Flower simulation client factory bound to experiment config."""

	def flower_client_fn(context):
		# Flower >=1.23 passes a Context object. Keep fallback for string/int cid.
		if hasattr(context, "node_config"):
			partition_id = context.node_config.get("partition-id", 0)
			user_id = int(partition_id)
		else:
			user_id = int(context)

		client = AnomalyDetectionClient(
			user_id=user_id,
			data_dir=data_dir,
			window_size=window_size,
			stride=stride,
			batch_size=batch_size,
			local_epochs=local_epochs,
			device="cpu",
		)
		return client.to_client()

	return flower_client_fn


def run_federated_training():
	"""Run full federated simulation and save the final global model."""
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

	base_model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE)
	initial_parameters = _build_initial_parameters(base_model)

	min_fit_clients = max(1, int(FRACTION_FIT * N_USERS))

	strategy = CheckpointFedAvg(
		checkpoint_dir=CHECKPOINT_DIR,
		in_channels=1,
		window_size=WINDOW_SIZE,
		fraction_fit=FRACTION_FIT,
		fraction_evaluate=FRACTION_FIT,
		min_fit_clients=min_fit_clients,
		min_evaluate_clients=min_fit_clients,
		min_available_clients=N_USERS,
		initial_parameters=initial_parameters,
	)

	client_fn = _make_client_fn(
		data_dir=DATA_DIR,
		window_size=WINDOW_SIZE,
		stride=STRIDE,
		batch_size=BATCH_SIZE,
		local_epochs=LOCAL_EPOCHS,
	)

	mlflow.set_experiment("federated_training")
	with mlflow.start_run(run_name="federated_fedavg"):
		mlflow.log_params(
			{
				"data_dir": DATA_DIR,
				"n_users": N_USERS,
				"n_rounds": N_ROUNDS,
				"fraction_fit": FRACTION_FIT,
				"local_epochs": LOCAL_EPOCHS,
				"batch_size": BATCH_SIZE,
				"window_size": WINDOW_SIZE,
				"stride": STRIDE,
				"checkpoint_dir": CHECKPOINT_DIR,
				"model_save_path": MODEL_SAVE_PATH,
			}
		)

		history = fl.simulation.start_simulation(
			client_fn=client_fn,
			num_clients=N_USERS,
			config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
			strategy=strategy,
			client_resources={"num_cpus": 1},
		)

		final_round_path = os.path.join(CHECKPOINT_DIR, f"round_{N_ROUNDS}.pt")
		if os.path.exists(final_round_path):
			selected_checkpoint = final_round_path
		else:
			round_paths = glob.glob(os.path.join(CHECKPOINT_DIR, "round_*.pt"))
			if not round_paths:
				raise FileNotFoundError("No round checkpoints found after simulation")
			selected_checkpoint = sorted(
				round_paths,
				key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
			)[-1]

		shutil.copyfile(selected_checkpoint, MODEL_SAVE_PATH)
		mlflow.log_artifact(MODEL_SAVE_PATH)

		final_val_loss = float("nan")
		final_roc_auc = float("nan")

		if hasattr(history, "losses_distributed") and history.losses_distributed:
			final_val_loss = float(history.losses_distributed[-1][1])

		if (
			hasattr(history, "metrics_distributed")
			and isinstance(history.metrics_distributed, dict)
			and "mean_ROC_AUC" in history.metrics_distributed
			and history.metrics_distributed["mean_ROC_AUC"]
		):
			final_roc_auc = float(history.metrics_distributed["mean_ROC_AUC"][-1][1])

		mlflow.log_metric("final_val_loss", final_val_loss)
		mlflow.log_metric("final_ROC_AUC", final_roc_auc)

	print("Federated training complete. Final model saved.")
	print(
		f"Summary: rounds_completed={N_ROUNDS}, "
		f"final_val_loss={final_val_loss:.6f}, "
		f"final_ROC_AUC={final_roc_auc:.6f}"
	)


if __name__ == "__main__":
	run_federated_training()
