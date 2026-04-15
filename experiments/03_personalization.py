import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
	os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import mlflow
from sklearn.metrics import roc_auc_score
from src.models.base_cnn import CNNAnomalyDetector
from src.data.preprocessing import load_user_data
from src.personalization.calibration import ScoreCalibrator
from src.personalization.thresholds import ThresholdSelector
from src.personalization.pers_head import (
	PersonalizedHead, fit_personalized_head, extract_features
)


DATA_DIR = 'data/raw/synthetic'
N_USERS = 50
WINDOW_SIZE = 128
STRIDE = 64
FEDERATED_MODEL_PATH = 'data/processed/global_model_federated.pt'
RESULTS_PATH = 'data/processed/personalization_results.csv'
CALIBRATORS_DIR = 'data/processed/calibrators'
TARGET_FPR = 0.05
MIN_HEAD_TRAIN_POS = 2
MIN_HEAD_VAL_POS = 1
HEAD_VAL_AUC_FLOOR = 0.55
HEAD_MAX_VAL_FPR = 0.15
HEAD_MIN_SCORE_STD = 1e-4


def load_frozen_encoder():
	model = CNNAnomalyDetector(in_channels=1, window_size=128)
	model.load_state_dict(
		torch.load(FEDERATED_MODEL_PATH, map_location='cpu')
	)
	model.eval()
	for param in model.parameters():
		param.requires_grad = False
	return model


def get_base_scores(model, X_windows):
	x_tensor = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1)
	all_scores = []

	with torch.no_grad():
		for start in range(0, x_tensor.shape[0], 64):
			batch = x_tensor[start:start + 64]
			scores = model.forward(batch).cpu().numpy()
			all_scores.append(scores)

	if len(all_scores) == 0:
		return np.array([], dtype=np.float32)
	return np.concatenate(all_scores, axis=0)


def _rates_from_scores(scores, labels, threshold=0.5):
	preds = (scores >= threshold).astype(int)
	tp = np.sum((preds == 1) & (labels == 1))
	fp = np.sum((preds == 1) & (labels == 0))
	tn = np.sum((preds == 0) & (labels == 0))
	fn = np.sum((preds == 0) & (labels == 1))
	tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
	fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
	return tpr, fpr, fnr, tp, fp, tn, fn


def main():
	os.makedirs(CALIBRATORS_DIR, exist_ok=True)

	print(f"{'='*60}")
	print('WEEK 3 — PERSONALIZATION')
	print('Starting point — Federated global model:')
	print('  ROC_AUC=0.9993 | TPR=0.9136 | FPR=0.0000')
	print('Goal: improve per-user TPR while keeping FPR low')
	print(f"{'='*60}")

	encoder = load_frozen_encoder()
	selector = ThresholdSelector()
	results = []
	skipped = 0

	for user_id in range(N_USERS):
		print(f"\n[User {user_id:03d}/{N_USERS-1}]")

		data = load_user_data(user_id, DATA_DIR, WINDOW_SIZE, STRIDE)
		X_train, y_train = data['X_train'], data['y_train']
		X_val, y_val = data['X_val'], data['y_val']
		X_test, y_test = data['X_test'], data['y_test']

		if np.sum(y_test) == 0:
			print('  Skipping — no anomalies in test set')
			skipped += 1
			continue

		global_test_scores = get_base_scores(encoder, X_test)
		global_tpr, global_fpr, global_fnr, _, _, _, _ = _rates_from_scores(
			global_test_scores, y_test, threshold=0.5
		)
		if len(np.unique(y_test)) > 1:
			global_auc = roc_auc_score(y_test, global_test_scores)
		else:
			global_auc = 0.0
		print(f'  Global  — AUC={global_auc:.4f} TPR={global_tpr:.4f} FPR={global_fpr:.4f}')

		global_val_scores = get_base_scores(encoder, X_val)
		effective_target_fpr = TARGET_FPR
		calibrator = ScoreCalibrator()
		calibrator.fit(global_val_scores, y_val.astype(float), lr=0.01, epochs=200)

		cal_val_scores = calibrator.predict(global_val_scores)
		threshold_result = selector.find_threshold(
			cal_val_scores, y_val, target_fpr=effective_target_fpr
		)
		tau = threshold_result['threshold']

		cal_test_scores = calibrator.predict(global_test_scores)
		cal_metrics = selector.evaluate_at_threshold(
			cal_test_scores, y_test, tau
		)
		cal_tpr = cal_metrics['TPR']
		cal_fpr = cal_metrics['FPR']
		cal_fnr = cal_metrics['FNR']
		if len(np.unique(y_test)) > 1:
			cal_auc = roc_auc_score(y_test, cal_test_scores)
		else:
			cal_auc = global_auc
		print(f'  Calib  — AUC={cal_auc:.4f} TPR={cal_tpr:.4f} FPR={cal_fpr:.4f} tau={tau:.2f}')

		calibrator.save(f'{CALIBRATORS_DIR}/user_{user_id:03d}.npy')

		train_pos = int(np.sum(y_train == 1))
		val_pos = int(np.sum(y_val == 1))
		use_head = train_pos >= MIN_HEAD_TRAIN_POS and val_pos >= MIN_HEAD_VAL_POS

		if not use_head:
			head_auc = cal_auc
			head_tpr = cal_tpr
			head_fpr = cal_fpr
			head_threshold = tau
			print(
				'  Head   — skipped (insufficient positives), '
				f'fallback to calibration AUC={head_auc:.4f} TPR={head_tpr:.4f} FPR={head_fpr:.4f}'
			)
		else:
			head, val_loss_history = fit_personalized_head(
				encoder=encoder,
				X_train=X_train, y_train=y_train,
				X_val=X_val, y_val=y_val,
				feature_dim=128, epochs=100, lr=1e-3
			)
			head.eval()

			test_features = extract_features(encoder, X_test)
			with torch.no_grad():
				head_test_scores = head(
					torch.tensor(test_features, dtype=torch.float32)
				).cpu().numpy()

			val_features = extract_features(encoder, X_val)
			with torch.no_grad():
				head_val_scores = head(
					torch.tensor(val_features, dtype=torch.float32)
				).cpu().numpy()

			head_threshold = selector.find_threshold(
				head_val_scores, y_val, target_fpr=effective_target_fpr
			)['threshold']

			head_val_metrics = selector.evaluate_at_threshold(
				head_val_scores, y_val, head_threshold
			)
			if len(np.unique(y_val)) > 1:
				head_val_auc = float(roc_auc_score(y_val, head_val_scores))
			else:
				head_val_auc = 0.0

			head_is_degenerate = float(np.std(head_val_scores)) < HEAD_MIN_SCORE_STD
			head_is_unstable = (
				head_val_auc < HEAD_VAL_AUC_FLOOR
				or head_val_metrics['FPR'] > HEAD_MAX_VAL_FPR
			)

			if head_is_degenerate or head_is_unstable:
				head_auc = cal_auc
				head_tpr = cal_tpr
				head_fpr = cal_fpr
				head_threshold = tau
				print(
					'  Head   — unstable on val, '
					f'fallback to calibration AUC={head_auc:.4f} TPR={head_tpr:.4f} FPR={head_fpr:.4f}'
				)
			else:
				head_metrics = selector.evaluate_at_threshold(
					head_test_scores, y_test, head_threshold
				)
				head_tpr = head_metrics['TPR']
				head_fpr = head_metrics['FPR']
				if len(np.unique(y_test)) > 1:
					head_auc = roc_auc_score(y_test, head_test_scores)
				else:
					head_auc = global_auc
				print(f'  Head   — AUC={head_auc:.4f} TPR={head_tpr:.4f} FPR={head_fpr:.4f} tau={head_threshold:.2f}')

		_ = cal_fnr

		results.append({
			'user_id': user_id,
			'global_AUC': global_auc,
			'global_TPR': global_tpr,
			'global_FPR': global_fpr,
			'global_FNR': global_fnr,
			'cal_AUC': cal_auc,
			'cal_TPR': cal_tpr,
			'cal_FPR': cal_fpr,
			'cal_tau': tau,
			'cal_a': calibrator.a,
			'cal_b': calibrator.b,
			'delta_TPR_cal': cal_tpr - global_tpr,
			'delta_FPR_cal': cal_fpr - global_fpr,
			'head_AUC': head_auc,
			'head_TPR': head_tpr,
			'head_FPR': head_fpr,
			'head_tau': head_threshold,
			'delta_TPR_head': head_tpr - global_tpr,
			'delta_FPR_head': head_fpr - global_fpr,
		})

	df = pd.DataFrame(results)
	df.to_csv(RESULTS_PATH, index=False)

	print(f"\n{'='*60}")
	print('PERSONALIZATION RESULTS — WEEK 3 FINAL SUMMARY')
	print(f"{'='*60}")
	print('')
	print('Starting point (Federated Global Model):')
	print('  ROC_AUC=0.9993 | TPR=0.9136 | FPR=0.0000')
	print('')
	print('After Option A — Platt Scaling Calibration:')
	print(f"  Mean AUC:  {df['cal_AUC'].mean():.4f}")
	print(f"  Mean TPR:  {df['cal_TPR'].mean():.4f}  (delta: {df['delta_TPR_cal'].mean():+.4f})")
	print(f"  Mean FPR:  {df['cal_FPR'].mean():.4f}  (delta: {df['delta_FPR_cal'].mean():+.4f})")
	print(f"  Users improved TPR: {(df['delta_TPR_cal']>0).sum()}/{len(df)}")
	print(f"  Users reduced FPR: {(df['delta_FPR_cal']<0).sum()}/{len(df)}")
	print('')
	print('After Option B — Personalized Head:')
	print(f"  Mean AUC:  {df['head_AUC'].mean():.4f}")
	print(f"  Mean TPR:  {df['head_TPR'].mean():.4f}  (delta: {df['delta_TPR_head'].mean():+.4f})")
	print(f"  Mean FPR:  {df['head_FPR'].mean():.4f}  (delta: {df['delta_FPR_head'].mean():+.4f})")
	print(f"  Users improved TPR: {(df['delta_TPR_head']>0).sum()}/{len(df)}")
	print(f"  Users reduced FPR: {(df['delta_FPR_head']<0).sum()}/{len(df)}")
	print('')
	print(f'Skipped users (no test anomalies): {skipped}')
	print(f'Results saved to: {RESULTS_PATH}')
	print(f"{'='*60}")

	mlflow.set_experiment('personalization_evaluation')
	with mlflow.start_run(run_name='week3_personalization'):
		mlflow.log_metric('mean_global_AUC', float(df['global_AUC'].mean()))
		mlflow.log_metric('mean_global_TPR', float(df['global_TPR'].mean()))
		mlflow.log_metric('mean_global_FPR', float(df['global_FPR'].mean()))
		mlflow.log_metric('mean_cal_AUC', float(df['cal_AUC'].mean()))
		mlflow.log_metric('mean_cal_TPR', float(df['cal_TPR'].mean()))
		mlflow.log_metric('mean_cal_FPR', float(df['cal_FPR'].mean()))
		mlflow.log_metric('mean_head_AUC', float(df['head_AUC'].mean()))
		mlflow.log_metric('mean_head_TPR', float(df['head_TPR'].mean()))
		mlflow.log_metric('mean_head_FPR', float(df['head_FPR'].mean()))
		mlflow.log_metric('mean_delta_TPR_cal', float(df['delta_TPR_cal'].mean()))
		mlflow.log_metric('mean_delta_FPR_cal', float(df['delta_FPR_cal'].mean()))
		mlflow.log_metric('mean_delta_TPR_head', float(df['delta_TPR_head'].mean()))
		mlflow.log_metric('mean_delta_FPR_head', float(df['delta_FPR_head'].mean()))
		mlflow.log_metric('users_improved_tpr_cal', float((df['delta_TPR_cal'] > 0).sum()))
		mlflow.log_metric('users_reduced_fpr_cal', float((df['delta_FPR_cal'] < 0).sum()))
		mlflow.log_metric('users_improved_tpr_head', float((df['delta_TPR_head'] > 0).sum()))
		mlflow.log_metric('users_reduced_fpr_head', float((df['delta_FPR_head'] < 0).sum()))
		mlflow.log_metric('skipped_users', float(skipped))
		mlflow.log_artifact(RESULTS_PATH)


if __name__ == '__main__':
	main()
