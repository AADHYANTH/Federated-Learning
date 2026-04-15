import numpy as np
from sklearn.metrics import confusion_matrix


class ThresholdSelector:
    def find_threshold(self, probabilities, labels, target_fpr=0.05):
        probabilities = np.asarray(probabilities).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        if len(np.unique(labels)) < 2:
            return {
                "threshold": 0.5,
                "achieved_fpr": 0.0,
                "achieved_tpr": 0.0,
                "note": "no_positives",
            }

        thresholds = np.arange(0.01, 1.0, 0.01)
        all_results = []

        for tau in thresholds:
            preds = (probabilities >= tau).astype(int)

            if len(np.unique(preds)) > 1:
                tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            else:
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                tn = np.sum((preds == 0) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            result = {
                "threshold": float(tau),
                "achieved_fpr": float(fpr),
                "achieved_tpr": float(tpr),
            }
            all_results.append(result)

        valid = [r for r in all_results if r["achieved_fpr"] <= float(target_fpr)]

        if valid:
            best_tpr = max(r["achieved_tpr"] for r in valid)
            tpr_best = [r for r in valid if np.isclose(r["achieved_tpr"], best_tpr)]
            best_fpr = min(r["achieved_fpr"] for r in tpr_best)
            fpr_best = [r for r in tpr_best if np.isclose(r["achieved_fpr"], best_fpr)]
            # Prefer thresholds closer to 0.5 to avoid extreme per-user cutoff instability.
            best = min(fpr_best, key=lambda x: abs(x["threshold"] - 0.5))
        else:
            # No threshold satisfied target FPR: choose minimum FPR and threshold closest to 0.5.
            min_fpr = min(r["achieved_fpr"] for r in all_results)
            fpr_best = [r for r in all_results if np.isclose(r["achieved_fpr"], min_fpr)]
            best = min(fpr_best, key=lambda x: abs(x["threshold"] - 0.5))

        return {
            "threshold": float(best["threshold"]),
            "achieved_fpr": float(best["achieved_fpr"]),
            "achieved_tpr": float(best["achieved_tpr"]),
        }

    def evaluate_at_threshold(self, probabilities, labels, threshold):
        probabilities = np.asarray(probabilities).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        preds = (probabilities >= threshold).astype(int)

        if len(np.unique(preds)) < 2:
            pass

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "TPR": tpr,
            "FPR": fpr,
            "FNR": fnr,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }
