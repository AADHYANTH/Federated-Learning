"""Edge deployment demo: ONNX export, latency benchmark, and streaming inference."""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch


MODEL_PATH = "data/processed/global_model_federated.pt"
ONNX_PATH = "data/processed/model_edge.onnx"
PERSONALIZATION_PATH = "data/processed/personalization_results.csv"
USER_ID = 0
DATA_DIR = "data/raw/synthetic"
WINDOW_SIZE = 128
STRIDE = 64
N_BENCHMARK_RUNS = 1000


try:
	from src.data.preprocessing import load_user_data
	from src.models.base_cnn import CNNAnomalyDetector
except ModuleNotFoundError:
	current_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(current_dir, ".."))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from src.data.preprocessing import load_user_data
	from src.models.base_cnn import CNNAnomalyDetector


def _sigmoid(x):
	"""Numerically stable sigmoid for scalar/array inputs."""
	x = np.clip(x, -50.0, 50.0)
	return 1.0 / (1.0 + np.exp(-x))


def _resolve_model_path(preferred_path):
	"""Resolve global model path with fallback to available federated models/checkpoints."""
	if os.path.exists(preferred_path):
		return preferred_path

	candidates = []
	candidates.extend(glob.glob("data/processed/global_model_federated*.pt"))
	candidates.extend(glob.glob("data/processed/checkpoints/round_*.pt"))
	candidates.extend(glob.glob("data/processed/checkpoints_smoke*/round_*.pt"))

	if not candidates:
		raise FileNotFoundError(
			f"Model not found at {preferred_path} and no fallback model/checkpoints exist"
		)

	def _sort_key(path):
		base = os.path.basename(path)
		if base.startswith("round_"):
			return int(os.path.splitext(base)[0].split("_")[-1])
		return -1

	selected = sorted(candidates, key=_sort_key)[-1]
	print(f"Model path {preferred_path} not found. Using fallback: {selected}")
	return selected


def _load_personalization_params(csv_path, user_id):
	"""Load per-user personalization parameters (a, b, tau) from results CSV."""
	path = csv_path
	if not os.path.exists(path):
		fallbacks = [
			"data/processed/personalization_results.csv",
			"data/processed/personalization_results_smoke.csv",
		]
		found = next((p for p in fallbacks if os.path.exists(p)), None)
		if found is None:
			raise FileNotFoundError(
				f"Personalization results not found at {csv_path} and no fallback CSV exists"
			)
		path = found
		print(f"Personalization file {csv_path} not found. Using fallback: {path}")

	df = pd.read_csv(path)
	row = df.loc[df["user_id"] == int(user_id)]
	if row.empty:
		raise ValueError(f"No row found for user_id={user_id} in {path}")
	row = row.iloc[0]

	# Accept multiple column naming conventions.
	a = float(row.get("a_i", row.get("a", row.get("calib_a", 1.0))))
	b = float(row.get("b_i", row.get("b", row.get("calib_b", 0.0))))
	tau = float(row.get("tau_i", row.get("tau", row.get("threshold", 0.5))))

	if "a_i" not in row.index and "a" not in row.index and "calib_a" not in row.index:
		print("Calibration scale a not found in CSV. Using default a=1.0")
	if "b_i" not in row.index and "b" not in row.index and "calib_b" not in row.index:
		print("Calibration bias b not found in CSV. Using default b=0.0")
	if "tau_i" not in row.index and "tau" not in row.index and "threshold" not in row.index:
		print("Threshold tau not found in CSV. Using default tau=0.5")

	return a, b, tau


def export_onnx(model_path=MODEL_PATH, onnx_path=ONNX_PATH):
	"""Export PyTorch CNN model to ONNX and verify with ONNX Runtime."""
	os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
	resolved_path = _resolve_model_path(model_path)

	model = CNNAnomalyDetector(in_channels=1, window_size=WINDOW_SIZE)
	state_dict = torch.load(resolved_path, map_location="cpu")
	model.load_state_dict(state_dict)
	model.eval()

	dummy_input = torch.randn(1, 1, 128, dtype=torch.float32)
	torch.onnx.export(
		model,
		dummy_input,
		onnx_path,
		export_params=True,
		opset_version=12,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["output"],
	)

	# Verify model loads and runs in ONNX Runtime.
	session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
	_ = session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})

	print(f"ONNX export successful. Model saved to {onnx_path}")
	return session


def benchmark_latency(session, n_runs=N_BENCHMARK_RUNS):
	"""Run repeated ONNX inference and print latency statistics."""
	input_name = session.get_inputs()[0].name

	latencies_ms = []
	for _ in range(int(n_runs)):
		x = np.random.randn(1, 1, 128).astype(np.float32)
		t0 = time.perf_counter()
		_ = session.run(None, {input_name: x})
		t1 = time.perf_counter()
		latencies_ms.append((t1 - t0) * 1000.0)

	arr = np.array(latencies_ms, dtype=np.float64)
	mean_ms = float(arr.mean())
	std_ms = float(arr.std())
	min_ms = float(arr.min())
	max_ms = float(arr.max())
	p95_ms = float(np.percentile(arr, 95))

	print("Latency benchmark (ms) over 1000 runs:")
	print(f"mean: {mean_ms:.4f}")
	print(f"std:  {std_ms:.4f}")
	print(f"min:  {min_ms:.4f}")
	print(f"max:  {max_ms:.4f}")
	print(f"p95:  {p95_ms:.4f}")
	print(f"Mean latency under 50ms: {mean_ms < 50.0}")


def run_streaming_demo(session, user_id=USER_ID):
	"""Simulate streaming inference with per-user personalization on test windows."""
	user_data = load_user_data(
		user_id=user_id,
		data_dir=DATA_DIR,
		window_size=WINDOW_SIZE,
		stride=STRIDE,
	)
	X_test = user_data["X_test"]  # (N, W, C)
	y_test = user_data["y_test"].astype(np.int32)

	a, b, tau = _load_personalization_params(PERSONALIZATION_PATH, user_id=user_id)
	print(f"Using personalization params for user {user_id}: a={a:.4f}, b={b:.4f}, tau={tau:.4f}")

	input_name = session.get_inputs()[0].name

	y_pred = []
	for i in range(X_test.shape[0]):
		window = X_test[i].T[np.newaxis, :, :].astype(np.float32)  # (1, 1, 128)
		base_score = float(session.run(None, {input_name: window})[0].reshape(-1)[0])
		p = float(_sigmoid(a * base_score + b))

		if p > tau:
			print(f"ANOMALY DETECTED at window {i}")
			y_pred.append(1)
		else:
			print(f"Normal at window {i}")
			y_pred.append(0)

	y_pred = np.asarray(y_pred, dtype=np.int32)

	tp = int(((y_pred == 1) & (y_test == 1)).sum())
	tn = int(((y_pred == 0) & (y_test == 0)).sum())
	fp = int(((y_pred == 1) & (y_test == 0)).sum())
	fn = int(((y_pred == 0) & (y_test == 1)).sum())

	tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

	print("\nStreaming summary")
	print(f"total windows: {X_test.shape[0]}")
	print(f"anomalies detected: {int((y_pred == 1).sum())}")
	print(f"true anomalies in ground truth: {int((y_test == 1).sum())}")
	print(f"TPR achieved: {tpr:.4f}")
	print(f"FPR achieved: {fpr:.4f}")


def main():
	"""Run all edge demo parts end-to-end."""
	session = export_onnx()
	benchmark_latency(session)
	run_streaming_demo(session, user_id=USER_ID)


if __name__ == "__main__":
	main()
