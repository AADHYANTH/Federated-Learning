"""Custom Flower strategy for federated anomaly detection training."""

from __future__ import annotations

import os
from collections import OrderedDict

import flwr as fl
import mlflow
import torch

from src.models.base_cnn import CNNAnomalyDetector


class CheckpointFedAvg(fl.server.strategy.FedAvg):
	"""FedAvg strategy with per-round checkpointing and MLflow metric logging.

	Parameters
	----------
	checkpoint_dir : str
		Directory used to store global model checkpoints per round.
	in_channels : int, optional
		Input channel count for model reconstruction when saving checkpoints.
	window_size : int, optional
		Window size used to instantiate the global model architecture.
	**kwargs
		Additional keyword arguments forwarded to ``fl.server.strategy.FedAvg``.
	"""

	def __init__(self, checkpoint_dir, in_channels=1, window_size=128, **kwargs):
		super().__init__(**kwargs)
		self.checkpoint_dir = checkpoint_dir
		self.in_channels = int(in_channels)
		self.window_size = int(window_size)
		os.makedirs(self.checkpoint_dir, exist_ok=True)

	def _save_parameters_checkpoint(self, parameters, server_round):
		"""Save aggregated global parameters as a PyTorch state_dict checkpoint."""
		ndarrays = fl.common.parameters_to_ndarrays(parameters)
		model = CNNAnomalyDetector(
			in_channels=self.in_channels,
			window_size=self.window_size,
		)
		state_keys = list(model.state_dict().keys())
		state_dict = OrderedDict()

		for key, array in zip(state_keys, ndarrays):
			state_dict[key] = torch.tensor(array)

		model.load_state_dict(state_dict, strict=True)
		path = os.path.join(self.checkpoint_dir, f"round_{server_round}.pt")
		torch.save(model.state_dict(), path)

	def aggregate_fit(self, server_round, results, failures):
		"""Aggregate client fit results and save global model checkpoint.

		This method logs the round participation count, delegates aggregation to
		FedAvg, then saves the aggregated global model to:
		``data/processed/checkpoints/round_{round_num}.pt``.
		"""
		print(
			f"Round {server_round}: aggregating fit from "
			f"{len(results)} participating clients"
		)

		aggregated_parameters, aggregated_metrics = super().aggregate_fit(
			server_round, results, failures
		)

		if aggregated_parameters is not None:
			self._save_parameters_checkpoint(aggregated_parameters, server_round)

		return aggregated_parameters, aggregated_metrics

	def aggregate_evaluate(self, server_round, results, failures):
		"""Aggregate evaluation metrics with weighted means and log to MLflow.

		Computes weighted averages (by number of client validation examples) for
		validation loss and ROC_AUC, prints round summary, and logs both metrics.
		"""
		del failures
		if not results:
			return None, {}

		total_examples = 0
		weighted_val_loss = 0.0
		weighted_roc_auc = 0.0

		for _, eval_res in results:
			n = int(eval_res.num_examples)
			total_examples += n
			weighted_val_loss += float(eval_res.loss) * n
			weighted_roc_auc += float(eval_res.metrics.get("ROC_AUC", 0.0)) * n

		if total_examples == 0:
			avg_val_loss = 0.0
			avg_roc_auc = 0.0
		else:
			avg_val_loss = weighted_val_loss / total_examples
			avg_roc_auc = weighted_roc_auc / total_examples

		print(
			f"Round {server_round} - val_loss: {avg_val_loss:.4f} | "
			f"mean_ROC_AUC: {avg_roc_auc:.4f}"
		)

		if mlflow.active_run() is not None:
			mlflow.log_metric("val_loss", float(avg_val_loss), step=server_round)
			mlflow.log_metric("mean_ROC_AUC", float(avg_roc_auc), step=server_round)

		return float(avg_val_loss), {"mean_ROC_AUC": float(avg_roc_auc)}
