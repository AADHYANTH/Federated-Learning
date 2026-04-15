import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import mlflow
import pandas as pd
import flwr as fl
from src.models.base_cnn import CNNAnomalyDetector
from src.federated.client import client_fn
from src.federated.strategy import FedAvgWithLogging
from src.data.preprocessing import load_user_data
from src.evaluation.metrics import compute_metrics


DATA_DIR = 'data/raw/synthetic'
N_USERS = 50
N_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
WINDOW_SIZE = 128
STRIDE = 64
CHECKPOINT_DIR = 'data/processed/checkpoints'
CENTRALIZED_MODEL_PATH = 'data/processed/global_model_centralized.pt'
FINAL_MODEL_PATH = 'data/processed/global_model_federated.pt'
MLFLOW_EXPERIMENT = 'federated_training'
RANDOM_SEED = 42


def get_initial_parameters():
	model = CNNAnomalyDetector(in_channels=1, window_size=128)
	model.load_state_dict(
		torch.load(CENTRALIZED_MODEL_PATH, map_location='cpu')
	)
	model.eval()

	ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
	print('Warm start: loaded centralized model as initial parameters')
	return fl.common.ndarrays_to_parameters(ndarrays), model


def run_comparison(n_users_to_compare=50):
	centralized_model = CNNAnomalyDetector(in_channels=1, window_size=128)
	centralized_model.load_state_dict(
		torch.load(CENTRALIZED_MODEL_PATH, map_location='cpu')
	)
	centralized_model.eval()

	federated_model = CNNAnomalyDetector(in_channels=1, window_size=128)
	federated_model.load_state_dict(
		torch.load(FINAL_MODEL_PATH, map_location='cpu')
	)
	federated_model.eval()

	results = []

	for user_id in range(n_users_to_compare):
		user_data = load_user_data(
			user_id=user_id,
			data_dir=DATA_DIR,
			window_size=WINDOW_SIZE,
			stride=STRIDE,
		)
		x_test = torch.tensor(user_data['X_test'], dtype=torch.float32).permute(0, 2, 1)
		y_test = user_data['y_test']

		with torch.no_grad():
			central_scores = centralized_model(x_test).cpu().numpy()
			fed_scores = federated_model(x_test).cpu().numpy()

		central_metrics = compute_metrics(y_test, central_scores)
		fed_metrics = compute_metrics(y_test, fed_scores)

		results.append(
			{
				'user_id': user_id,
				'centralized_ROC_AUC': central_metrics['ROC_AUC'],
				'federated_ROC_AUC': fed_metrics['ROC_AUC'],
				'centralized_TPR': central_metrics['TPR'],
				'federated_TPR': fed_metrics['TPR'],
				'centralized_FPR': central_metrics['FPR'],
				'federated_FPR': fed_metrics['FPR'],
				'delta_ROC_AUC': fed_metrics['ROC_AUC'] - central_metrics['ROC_AUC'],
				'delta_TPR': fed_metrics['TPR'] - central_metrics['TPR'],
				'delta_FPR': fed_metrics['FPR'] - central_metrics['FPR'],
			}
		)

	df = pd.DataFrame(results)

	print(df.to_string(index=False))

	mean_c_roc = float(np.nanmean(df['centralized_ROC_AUC'].values))
	mean_f_roc = float(np.nanmean(df['federated_ROC_AUC'].values))
	mean_d_roc = float(np.nanmean(df['delta_ROC_AUC'].values))

	mean_c_tpr = float(np.nanmean(df['centralized_TPR'].values))
	mean_f_tpr = float(np.nanmean(df['federated_TPR'].values))
	mean_d_tpr = float(np.nanmean(df['delta_TPR'].values))

	mean_c_fpr = float(np.nanmean(df['centralized_FPR'].values))
	mean_f_fpr = float(np.nanmean(df['federated_FPR'].values))
	mean_d_fpr = float(np.nanmean(df['delta_FPR'].values))

	better_or_equal = int((df['federated_ROC_AUC'] >= df['centralized_ROC_AUC']).sum())

	print('=' * 60)
	print('CENTRALIZED vs FEDERATED — FINAL COMPARISON')
	print('=' * 60)
	print('Metric        | Centralized | Federated  | Delta')
	print('-' * 60)
	print(f'Mean ROC_AUC  | {mean_c_roc:.4f}     | {mean_f_roc:.4f}    | {mean_d_roc:+.4f}')
	print(f'Mean TPR      | {mean_c_tpr:.4f}     | {mean_f_tpr:.4f}    | {mean_d_tpr:+.4f}')
	print(f'Mean FPR      | {mean_c_fpr:.4f}     | {mean_f_fpr:.4f}    | {mean_d_fpr:+.4f}')
	print('-' * 60)
	print(f'Users where federated ROC_AUC >= centralized: {better_or_equal}/50')

	output_csv = 'data/processed/centralized_vs_federated.csv'
	df.to_csv(output_csv, index=False)

	return df


def main():
	torch.manual_seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs('data/processed/plots', exist_ok=True)
	print(f'Starting federated training | users={N_USERS} rounds={N_ROUNDS}')
	print('Centralized baseline | ROC_AUC=0.9990 TPR=0.9487 FPR=0.0140')

	initial_parameters, _ = get_initial_parameters()

	strategy = FedAvgWithLogging(
		initial_parameters=initial_parameters,
		n_rounds=N_ROUNDS,
		checkpoint_dir=CHECKPOINT_DIR,
	)

	mlflow.set_experiment(MLFLOW_EXPERIMENT)
	with mlflow.start_run(run_name='federated_20rounds_50users'):
		mlflow.log_param('DATA_DIR', DATA_DIR)
		mlflow.log_param('N_USERS', N_USERS)
		mlflow.log_param('N_ROUNDS', N_ROUNDS)
		mlflow.log_param('LOCAL_EPOCHS', LOCAL_EPOCHS)
		mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
		mlflow.log_param('WINDOW_SIZE', WINDOW_SIZE)
		mlflow.log_param('STRIDE', STRIDE)
		mlflow.log_param('CHECKPOINT_DIR', CHECKPOINT_DIR)
		mlflow.log_param('CENTRALIZED_MODEL_PATH', CENTRALIZED_MODEL_PATH)
		mlflow.log_param('FINAL_MODEL_PATH', FINAL_MODEL_PATH)
		mlflow.log_param('MLFLOW_EXPERIMENT', MLFLOW_EXPERIMENT)
		mlflow.log_param('RANDOM_SEED', RANDOM_SEED)

		fl.simulation.start_simulation(
			client_fn=client_fn,
			num_clients=N_USERS,
			config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
			strategy=strategy,
			client_resources={'num_cpus': 1, 'num_gpus': 0.0},
		)

		strategy.print_final_summary()

	best_round = strategy.get_best_round()
	if best_round == 0:
		best_round = N_ROUNDS
	best_checkpoint = os.path.join(
		CHECKPOINT_DIR, f'round_{best_round:03d}.pt'
	)
	best_model = CNNAnomalyDetector(in_channels=1, window_size=128)
	best_model.load_state_dict(
		torch.load(best_checkpoint, map_location='cpu')
	)
	torch.save(best_model.state_dict(), FINAL_MODEL_PATH)
	print(
		f'Best model from round {best_round} '
		f'(ROC_AUC={strategy.best_roc_auc:.4f}) saved to {FINAL_MODEL_PATH}'
	)

	comparison_df = run_comparison(n_users_to_compare=50)
	with mlflow.start_run(run_name='federated_evaluation', nested=True):
		mlflow.log_metric('mean_delta_ROC_AUC', float(np.nanmean(comparison_df['delta_ROC_AUC'].values)))
		mlflow.log_metric('mean_delta_TPR', float(np.nanmean(comparison_df['delta_TPR'].values)))
		mlflow.log_metric('mean_delta_FPR', float(np.nanmean(comparison_df['delta_FPR'].values)))
		mlflow.log_artifact('data/processed/centralized_vs_federated.csv')

	print('\nWeek 2 complete. Run experiments/02b_evaluate_federated.py for full evaluation plots.')


if __name__ == '__main__':
	main()
