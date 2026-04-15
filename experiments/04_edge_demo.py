import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import numpy as np
import torch
import time
import pandas as pd
import onnx
import onnxruntime as ort
from src.models.base_cnn import CNNAnomalyDetector
from src.personalization.calibration import ScoreCalibrator


SYNTH_MODEL_PATH = 'data/processed/global_model_federated.pt'
ECG_MODEL_PATH = 'data/processed/ecg_model_federated.pt'
SYNTH_ONNX_PATH = 'data/processed/model_synth_edge.onnx'
ECG_ONNX_PATH = 'data/processed/model_ecg_edge.onnx'
PERS_RESULTS = 'data/processed/personalization_results.csv'
DATA_DIR = 'data/raw/synthetic'
WINDOW_SIZE_SYNTH = 128
WINDOW_SIZE_ECG = 187
N_LATENCY_RUNS = 1000
DEMO_USER_ID = 7


def export_to_onnx(model_path, onnx_path, window_size, label):
    print(f"\n[ONNX Export — {label}]")

    model = CNNAnomalyDetector(in_channels=1, window_size=window_size)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 1, window_size)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['signal_window'],
        output_names=['anomaly_score'],
        dynamic_axes={
            'signal_window': {0: 'batch_size'},
            'anomaly_score': {0: 'batch_size'}
        }
    )

    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    print(f"  Exported to {onnx_path}")
    print(f"  File size: {size_mb:.2f} MB")
    print("  ONNX validation: PASSED")
    return onnx_path


def benchmark_latency(onnx_path, window_size, label, n_runs=1000):
    print(f"\n[Latency Benchmark — {label}]")

    session = ort.InferenceSession(onnx_path,
        providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for _ in range(50):
        dummy = np.random.randn(1, 1, window_size).astype(np.float32)
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_runs):
        dummy = np.random.randn(1, 1, window_size).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    p50_ms = np.percentile(latencies, 50)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)

    print(f"  Runs:      {n_runs}")
    print(f"  Mean:      {mean_ms:.3f} ms")
    print(f"  Std:       {std_ms:.3f} ms")
    print(f"  Min:       {min_ms:.3f} ms")
    print(f"  Max:       {max_ms:.3f} ms")
    print(f"  P50:       {p50_ms:.3f} ms")
    print(f"  P95:       {p95_ms:.3f} ms")
    print(f"  P99:       {p99_ms:.3f} ms")

    if mean_ms < 50:
        print(f"  ✅ PASSED: Mean latency {mean_ms:.2f}ms < 50ms target")
    else:
        print(f"  ⚠️  Mean latency {mean_ms:.2f}ms > 50ms target")

    return {'mean': mean_ms, 'p95': p95_ms, 'p99': p99_ms}


def run_streaming_demo(onnx_path, demo_user_id):
    print(f"\n[Streaming Edge Demo — User {demo_user_id:03d}]")
    print("Simulating real-time anomaly detection on device...")
    print("-" * 50)

    pers_df = pd.read_csv(PERS_RESULTS)
    user_row = pers_df[pers_df['user_id'] == demo_user_id].iloc[0]
    a = float(user_row['cal_a'])
    b = float(user_row['cal_b'])
    tau = float(user_row['cal_tau'])

    print(f"  Personalization: a={a:.4f} b={b:.4f} threshold={tau:.4f}")

    signal = np.load(
      f'{DATA_DIR}/user_{demo_user_id:03d}_signal.npy')
    labels = np.load(
      f'{DATA_DIR}/user_{demo_user_id:03d}_labels.npy')

    if signal.ndim == 1:
      signal = signal.reshape(-1, 1)
    labels = labels.reshape(-1)

    train_portion = signal[:3000]
    mean = train_portion.mean()
    std = train_portion.std() + 1e-8
    signal = (signal - mean) / std

    session = ort.InferenceSession(onnx_path,
      providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    stride = WINDOW_SIZE_SYNTH // 2
    window_count = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    alerts_fired = []

    for start in range(0, len(signal) - WINDOW_SIZE_SYNTH, stride):
      window = signal[start:start + WINDOW_SIZE_SYNTH]
      window_labels = labels[start:start + WINDOW_SIZE_SYNTH]
      true_label = 1 if window_labels.mean() > 0.3 else 0

      window_t = window.T
      onnx_input = window_t.reshape(1, 1, WINDOW_SIZE_SYNTH).astype(np.float32)
      output = session.run(None, {input_name: onnx_input})
      base_score = float(np.asarray(output[0]).reshape(-1)[0])

      cal_score = 1 / (1 + np.exp(-(a * base_score + b)))

      alert = cal_score >= tau

      if alert and true_label == 1:
        true_positives += 1
      elif alert and true_label == 0:
        false_positives += 1
      elif (not alert) and true_label == 0:
        true_negatives += 1
      else:
        false_negatives += 1

      alerts_fired.append(bool(alert))

      if window_count % 10 == 0:
        status = "🚨 ANOMALY" if alert else "✅ Normal "
        correct = "✓" if (alert == bool(true_label)) else "✗"
        print(f"  Window {window_count:04d} | "
          f"base={base_score:.3f} cal={cal_score:.3f} | "
          f"{status} | truth={'ANOM' if true_label else 'norm'} {correct}")

      window_count += 1

    print("\n--- Demo Summary ---")
    print(f"Windows processed: {window_count}")
    alert_total = sum(alerts_fired)
    alert_pct = (alert_total / window_count * 100) if window_count > 0 else 0.0
    print(f"Alerts fired:      {alert_total} "
      f"({alert_pct:.1f}%)")
    print(f"True Positives:    {true_positives}")
    print(f"False Positives:   {false_positives}")
    print(f"True Negatives:    {true_negatives}")
    print(f"False Negatives:   {false_negatives}")
    tpr = true_positives / (true_positives + false_negatives) \
      if (true_positives + false_negatives) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) \
      if (false_positives + true_negatives) > 0 else 0
    print(f"Demo TPR: {tpr:.4f}")
    print(f"Demo FPR: {fpr:.4f}")


def main():
  print(f"\n{'#' * 60}")
  print("WEEK 4 — EDGE DEMO & ONNX EXPORT")
  print(f"{'#' * 60}")

  export_to_onnx(SYNTH_MODEL_PATH, SYNTH_ONNX_PATH,
    WINDOW_SIZE_SYNTH, "Synthetic Model")
  if os.path.exists(ECG_MODEL_PATH):
    export_to_onnx(ECG_MODEL_PATH, ECG_ONNX_PATH,
      WINDOW_SIZE_ECG, "ECG Model")
  else:
    print("ECG model not found — skipping ECG export")

  synth_latency = benchmark_latency(
    SYNTH_ONNX_PATH, WINDOW_SIZE_SYNTH, "Synthetic", n_runs=N_LATENCY_RUNS)
  _ = synth_latency
  if os.path.exists(ECG_ONNX_PATH):
    ecg_latency = benchmark_latency(
      ECG_ONNX_PATH, WINDOW_SIZE_ECG, "ECG", n_runs=N_LATENCY_RUNS)
    _ = ecg_latency

  run_streaming_demo(SYNTH_ONNX_PATH, DEMO_USER_ID)

  print("\n✅ Edge demo complete!")
  print(f"ONNX models: {SYNTH_ONNX_PATH}")
  print("Next: Run experiments/06_final_report.py "
    "then write README")


if __name__ == '__main__':
  main()
