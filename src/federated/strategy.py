import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import mlflow
from typing import List, Tuple, Dict, Optional, Union
import flwr as fl
from flwr.common import Parameters, Scalar, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
from src.models.base_cnn import CNNAnomalyDetector


class FedAvgWithLogging(fl.server.strategy.FedAvg):
	def __init__(self, initial_parameters, n_rounds, checkpoint_dir):
		super().__init__(
			fraction_fit=0.4,
			fraction_evaluate=0.4,
			min_fit_clients=5,
			min_evaluate_clients=5,
			min_available_clients=10,
			initial_parameters=initial_parameters,
		)
		self.n_rounds = n_rounds
		self.checkpoint_dir = checkpoint_dir
		os.makedirs(checkpoint_dir, exist_ok=True)
		self.round_train_losses = []
		self.round_val_losses = []
		self.round_roc_aucs = []
		self.best_roc_auc = 0.0
		self.best_round = 0

	def aggregate_fit(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, FitRes]],
		failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		print(f"\n{'='*50}")
		print(f"ROUND {server_round}/{self.n_rounds}")
		print(f"{'='*50}")
		print(f"Clients responded: {len(results)} | Failed: {len(failures)}")

		if not results:
			return None, {}

		total_examples = sum(fit_res.num_examples for _, fit_res in results)
		weighted_loss = sum(
			fit_res.metrics["train_loss"] * fit_res.num_examples
			for _, fit_res in results
		) / total_examples

		print(f"Avg train_loss: {weighted_loss:.4f}")
		self.round_train_losses.append(weighted_loss)

		aggregated = super().aggregate_fit(server_round, results, failures)

		if aggregated[0] is not None:
			weights = fl.common.parameters_to_ndarrays(aggregated[0])
			model = CNNAnomalyDetector(in_channels=1, window_size=128)
			params_dict = zip(model.state_dict().keys(), weights)
			state_dict = {k: torch.tensor(v) for k, v in params_dict}
			model.load_state_dict(state_dict, strict=True)
			save_path = os.path.join(self.checkpoint_dir, f"round_{server_round:03d}.pt")
			torch.save(model.state_dict(), save_path)
			print(f"Checkpoint saved: round_{server_round:03d}.pt")

		return aggregated

	def aggregate_evaluate(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, EvaluateRes]],
		failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
	) -> Tuple[Optional[float], Dict[str, Scalar]]:
		if not results:
			return None, {}

		total_examples = sum(eval_res.num_examples for _, eval_res in results)
		weighted_val_loss = sum(
			eval_res.metrics["val_loss"] * eval_res.num_examples
			for _, eval_res in results
		) / total_examples
		weighted_roc_auc = sum(
			eval_res.metrics["roc_auc"] * eval_res.num_examples
			for _, eval_res in results
		) / total_examples

		self.round_val_losses.append(weighted_val_loss)
		self.round_roc_aucs.append(weighted_roc_auc)

		print(f"Avg val_loss: {weighted_val_loss:.4f} | Avg ROC_AUC: {weighted_roc_auc:.4f}")

		if weighted_roc_auc > self.best_roc_auc:
			self.best_roc_auc = weighted_roc_auc
			self.best_round = server_round
			print(f"*** New best ROC_AUC: {weighted_roc_auc:.4f} ***")

		try:
			mlflow.log_metric("fed_train_loss", self.round_train_losses[-1], step=server_round)
			mlflow.log_metric("fed_val_loss", weighted_val_loss, step=server_round)
			mlflow.log_metric("fed_roc_auc", weighted_roc_auc, step=server_round)
		except Exception:
			pass

		return super().aggregate_evaluate(server_round, results, failures)

	def get_best_round(self) -> int:
		return self.best_round

	def print_final_summary(self):
		print(f"{'='*50}")
		print("FEDERATED TRAINING COMPLETE")
		print(f"{'='*50}")
		print(f"Total rounds: {self.n_rounds}")
		print(f"Best ROC_AUC: {self.best_roc_auc:.4f} at round {self.best_round}")
		print(f"Best val_loss: {min(self.round_val_losses):.4f}")
		print(f"Final ROC_AUC: {self.round_roc_aucs[-1]:.4f}")
		print(f"Final val_loss: {self.round_val_losses[-1]:.4f}")
		print(f"{'='*50}")
