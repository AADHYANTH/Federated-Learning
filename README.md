# Federated Health Anomaly

Federated Personalization for Time-Series Health Anomaly Detection.
This project trains anomaly models across distributed users, personalizes decisions per user, validates on real ECG data, and deploys to edge with ONNX.

## 1. Why this project matters

- Health data is highly personal. A model that works for one person can fail for another if personalization is missing.
- Federated learning enables collaborative training across users without centralizing raw signals, which is better for privacy.
- Edge deployment is critical for real-time alerts where low-latency inference matters more than cloud-only scoring.

## 2. Architecture (ASCII)

```text
					  +------------------+
					  |  Raw Data Sources|
					  | synthetic / ECG  |
					  +---------+--------+
									|
									v
					  +------------------+
					  |  Preprocessing   |
					  | normalize/split  |
					  +---------+--------+
									|
									v
					  +------------------+
					  |   CNN Encoder    |
					  | CNNAnomalyDetector|
					  +---------+--------+
									|
									v
					  +------------------+
					  | Federated Server |
					  | FedAvg rounds    |
					  +---------+--------+
									|
									v
					  +------------------+
					  | Personalization  |
					  | Platt + threshold|
					  +---------+--------+
									|
									v
					  +------------------+
					  |  Edge Inference  |
					  | ONNX + streaming |
					  +------------------+
```

## 3. Key results

| Model | Mean ROC-AUC | Mean TPR | Mean FPR | Notes |
|---|---:|---:|---:|---|
| Centralized | 1.0000 | 0.6000 | 0.0196 | Synthetic pooled baseline from centralized evaluation run |
| Federated Global | 0.2955 (smoke) | 0.0000 (smoke) | - (smoke) | From 1-round smoke simulation only; run full 20-round experiment for final numbers |
| Federated + Personalized | 1.0000 (smoke users with positives) | delta-based | delta-based | Personalization tracked via per-user delta TPR/FPR and improved-user percentage |

Notes:
- Centralized numbers above come from the current synthetic evaluation run.
- Federated and personalized rows should be refreshed using full runs and MLflow metrics in your environment.

## 4. Tech stack

- Python 3.10+
- PyTorch (modeling/training)
- Flower + Ray (federated simulation)
- NumPy + Pandas (data handling)
- Scikit-learn (metrics/utilities)
- MLflow (experiment tracking)
- ONNX + ONNX Runtime (edge export/inference)
- Matplotlib/Seaborn (plots)

## 5. Project structure

```text
federated-health-anomaly/
├── config.yaml
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── synthetic/
│   │   └── ecg/                         # mitbih_train.csv, mitbih_test.csv
│   └── processed/
│       ├── global_model_centralized.pt
│       ├── global_model_federated.pt
│       ├── model_edge.onnx
│       ├── personalization_results.csv
│       ├── real_vs_synthetic_comparison.csv
│       ├── checkpoints/
│       └── plots/
├── experiments/
│   ├── 00_evaluate_centralized.py
│   ├── 01_centralized.py
│   ├── 02_federated.py
│   ├── 03_personalization.py
│   ├── 04_edge_demo.py
│   └── 05_real_ecg_federated.py
└── src/
	 ├── data/
	 │   ├── synthetic_gen.py
	 │   ├── preprocessing.py
	 │   ├── datasets.py
	 │   └── ecg_loader.py
	 ├── models/
	 │   └── base_cnn.py
	 ├── federated/
	 │   ├── client.py
	 │   └── strategy.py
	 ├── personalization/
	 │   ├── calibration.py
	 │   ├── thresholds.py
	 │   └── pers_head.py
	 └── evaluation/
		  ├── metrics.py
		  └── plots.py
```

## 6. How to run (exact command order)

### Step A: Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step B: Generate synthetic data

```bash
python src/data/synthetic_gen.py
```

### Step C: Centralized baseline training

```bash
python experiments/01_centralized.py
```

### Step D: Federated training (synthetic)

```bash
python experiments/02_federated.py
```

### Step E: Personalization (synthetic)

```bash
python experiments/03_personalization.py
```

### Step F: Real ECG federated + personalization

Place MIT-BIH CSVs here first:
- data/raw/ecg/mitbih_train.csv
- data/raw/ecg/mitbih_test.csv

Then run:

```bash
python experiments/05_real_ecg_federated.py
```

### Step G: Edge export and streaming demo

```bash
python experiments/04_edge_demo.py
```

## 7. Results and plots

This project generates and saves three main chart outputs:

1. Per-user ROC curves (all users + mean curve)
	- Shows user-level variability and overall discriminative behavior.
	- Saved by evaluation pipeline to data/processed/plots/per_user_roc.png.

2. Metric distribution histograms (ROC-AUC, TPR, FPR)
	- Highlights spread and stability of model behavior across users.
	- Saved to data/processed/plots/metric_distributions.png.

3. Real-vs-synthetic personalization comparison table (CSV)
	- Compares mean delta TPR, mean delta FPR, and improved-user percentage.
	- Saved to data/processed/real_vs_synthetic_comparison.csv.

## 8. What I learned

- Personalization is not optional for health time-series; global thresholds can underperform for many users.
- Federated training can preserve privacy while still learning useful shared representations.
- Calibration and thresholding are lightweight but practical methods for user-specific adaptation.
- Edge deployment is feasible with ONNX Runtime, with low CPU latency for near real-time alerts.

## 9. Future improvements

- Add true patient-level grouping on real datasets (instead of simulated user chunks).
- Replace fixed Platt scaling with online calibration updates from streaming feedback.
- Add confidence-aware alerting and temporal smoothing to reduce false positives in production.

## 10. References

1. MIT-BIH Arrhythmia Database (PhysioNet):
	- Moody, G. B., and Mark, R. G. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50, 2001.
	- https://physionet.org/content/mitdb/

2. Flower framework:
	- Beutel, D. J. et al. Flower: A Friendly Federated Learning Research Framework. arXiv:2007.14390
	- https://flower.ai/

3. FedAvg paper:
	- McMahan, H. B. et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.
	- https://arxiv.org/abs/1602.05629