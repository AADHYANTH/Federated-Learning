# Federated Personalization for Time-Series Health Anomaly Detection
Badges: Python 3.10 | PyTorch | Flower | ONNX | MLflow

## 1. Title and Badges


## 2. Project Description

This project builds an end-to-end anomaly detection pipeline for wearable and clinical time-series data, from centralized training to federated learning, user-level personalization, and ONNX edge deployment. It is designed for ML engineers, health AI researchers, and practitioners who need accurate per-user anomaly detection while minimizing raw data sharing. Privacy-preserving ML is essential for health signals because centralized collection increases risk exposure, while federated training keeps sensitive user data local and still allows learning a strong global model.

## 3. Architecture Diagram

	Raw Signals
		|
		v
	Preprocessing
		|
		v
	1D CNN
		|
		v
	Federated Server (FedAvg)
		|
		v
	Global Model
		|
		v
	Per-User Calibration
		|
		v
	Personalized Threshold
		|
		v
	Edge Inference (ONNX)

## 4. Key Results

| Model Stage | ROC-AUC | TPR | FPR |
|---|---:|---:|---:|
| Centralized baseline | 0.9990 | 0.9487 | 0.0140 |
| Federated global | 0.9993 | 0.9136 | 0.0000 |
| After calibration | 0.9993 | 0.8889 | 0.0167 |
| After pers. head | 0.9167 | 0.8762 | 0.0453 |

Additional verified outcomes:
- Federated: 50 users improved ROC_AUC: 50/50
- Edge inference: < 50ms per window on CPU
- Real ECG data: MIT-BIH Heartbeat dataset, 47 simulated users

## 5. What Makes This Project Stand Out

- Privacy-preserving learning: global model quality without pooling raw user health signals in one place.
- Per-user personalization: calibration and adaptive thresholds tailor decisions to each user profile.
- Edge deployable pipeline: ONNX export and CPU runtime validation for low-latency inference.
- Real-data validation: tested beyond synthetic signals on MIT-BIH heartbeat ECG with 47 simulated users.

## 6. Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Define and train the 1D CNN anomaly detector |
| Flower | Federated training orchestration with FedAvg |
| ONNX Runtime | Fast CPU inference for edge deployment |
| MLflow | Experiment tracking, metrics, and artifacts |
| scikit-learn | Evaluation metrics and analysis helpers |
| NumPy | Numeric processing for signal windows and tensors |
| pandas | Result tables, aggregation, and CSV outputs |

## 7. Project Structure

	federated-learning/
	├── .gitignore                         # Git ignore rules
	├── README.md                          # Project documentation and usage guide
	├── config.yaml                        # Project-level configuration
	├── requirements.txt                   # Python dependencies
	├── package.json                       # Node metadata used in local tooling
	├── package-lock.json                  # Locked Node dependency graph
	├── data/
	│   ├── raw/
	│   │   ├── ecg/
	│   │   │   ├── mitbih_train.csv       # MIT-BIH train split
	│   │   │   └── mitbih_test.csv        # MIT-BIH test split
	│   │   └── synthetic/
	│   │       ├── user_000_signal.npy ... user_049_signal.npy  # Synthetic user signals
	│   │       └── user_000_labels.npy ... user_049_labels.npy  # Synthetic user labels
	│   └── processed/
	│       ├── global_model_centralized.pt                       # Best centralized synthetic model
	│       ├── global_model_federated.pt                         # Best federated synthetic model
	│       ├── ecg_model_centralized.pt                          # Best centralized ECG model
	│       ├── ecg_model_federated.pt                            # Best federated ECG model
	│       ├── personalization_results.csv                       # Synthetic personalization results
	│       ├── ecg_personalization_results.csv                   # ECG personalization results
	│       ├── centralized_vs_federated.csv                      # Synthetic centralized-vs-federated table
	│       ├── synthetic_vs_ecg_comparison.csv                   # Synthetic-vs-ECG summary table
	│       ├── model_edge.onnx                                   # Legacy edge export artifact
	│       ├── model_synth_edge.onnx                             # Synthetic ONNX edge model
	│       ├── model_ecg_edge.onnx                               # ECG ONNX edge model
	│       ├── calibrators/                                      # Per-user saved calibrator artifacts
	│       ├── checkpoints/                                      # Synthetic federated round checkpoints
	│       ├── checkpoints_real_ecg/                             # ECG federated round checkpoints
	│       ├── ecg_checkpoints/                                  # ECG pipeline round checkpoints
	│       └── plots/                                            # Generated plot images
	├── experiments/
	│   ├── 00_evaluate_centralized.py           # Centralized model evaluation script
	│   ├── 01_centralized.py                    # Week 1 centralized training pipeline
	│   ├── 02_federated.py                      # Week 2 federated training pipeline
	│   ├── 02b_evaluate_federated.py            # Federated evaluation and comparison plots
	│   ├── 03_personalization.py                # Week 3 personalization experiments
	│   ├── 03b_personalization_plots.py         # Personalization-focused plot generation
	│   ├── 04_edge_demo.py                      # Week 4 ONNX export and edge streaming demo
	│   ├── 05_real_ecg_federated.py             # Real ECG federated + personalization experiment
	│   └── 05_real_ecg_pipeline.py              # Unified real ECG centralized/federated/personalized pipeline
	├── src/
	│   ├── __init__.py                          # Package marker
	│   ├── data/
	│   │   ├── __init__.py                      # Data package marker
	│   │   ├── synthetic_gen.py                 # Synthetic signal and label generation
	│   │   ├── preprocessing.py                 # Windowing, normalization, and splits
	│   │   ├── datasets.py                      # Dataset loaders and weighting helpers
	│   │   └── ecg_loader.py                    # MIT-BIH load/split/normalization utilities
	│   ├── models/
	│   │   ├── __init__.py                      # Models package marker
	│   │   ├── base_cnn.py                      # CNNAnomalyDetector architecture
	│   │   └── autoencoder.py                   # Auxiliary autoencoder model
	│   ├── federated/
	│   │   ├── __init__.py                      # Federated package marker
	│   │   ├── client.py                        # Federated client logic
	│   │   ├── server.py                        # Federated server utilities
	│   │   └── strategy.py                      # FedAvgWithLogging strategy implementation
	│   ├── personalization/
	│   │   ├── __init__.py                      # Personalization package marker
	│   │   ├── calibration.py                   # ScoreCalibrator implementation
	│   │   ├── thresholds.py                    # Threshold selection logic
	│   │   └── pers_head.py                     # Personalized head training and inference
	│   └── evaluation/
	│       ├── __init__.py                      # Evaluation package marker
	│       ├── metrics.py                       # Metric computation helpers
	│       └── plots.py                         # Shared plotting functions
	├── mlruns/                                  # MLflow tracking store (multiple experiments/runs)
	└── venv/                                    # Local Python virtual environment

## 8. How to Run

### Setup

	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt

### Data generation

	python src/data/synthetic_gen.py

### Centralized

	python experiments/01_centralized.py

### Federated

	python experiments/02_federated.py

### Personalization

	python experiments/03_personalization.py

### ECG pipeline

	python experiments/05_real_ecg_pipeline.py

### Edge demo

	python experiments/04_edge_demo.py

## 9. Results and Visualizations

All charts below are saved in data/processed/plots/.

- calibration_params.png: Scatter view of per-user calibration parameters and how strongly each user needed calibration adjustment.
- centralized_vs_federated_bars.png: Side-by-side bar chart comparing centralized and federated averages for ROC-AUC, TPR, and FPR.
- delta_distributions.png: Distribution histograms of user-level delta TPR and delta FPR across personalization methods.
- federated_distributions.png: Histogram overview of federated model metrics across users.
- federated_roc.png: User-level ROC curves for the federated model, including aggregate trend.
- federated_training_curves.png: Round-by-round federated validation loss and ROC-AUC trend during training.
- full_pipeline_summary.png: End-to-end summary bars for centralized, federated, and personalized pipeline performance.
- metric_distributions.png: Synthetic centralized metric distribution snapshot across users.
- per_user_delta.png: Scatter of user-wise federated-minus-centralized TPR/FPR deltas to show wins and regressions.
- per_user_roc.png: User-level ROC curves for the centralized baseline model.
- personalization_auc_comparison.png: Per-user ROC-AUC comparison across global, calibrated, and personalized-head variants.
- tpr_improvement.png: Before-vs-after TPR scatter for calibration and personalized-head methods.

## 10. Limitations and Future Work

- Synthetic data simplicity: synthetic signal generation cannot fully represent the variability and noise of real-world physiology.
- Personalized head instability: personalized head training can be sensitive for users with small positive class support.
- No differential privacy yet: federated aggregation is implemented, but formal DP guarantees are not currently added.
- No real mobile deployment yet: ONNX CPU edge validation is done on workstation hardware, not mobile devices.

## 11. References

- MIT-BIH Arrhythmia Database (PhysioNet)
- Flower: A Friendly Federated Learning Framework
- McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)
- Platt Scaling for probability calibration
