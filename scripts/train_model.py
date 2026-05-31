from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "amount_sum_1s",
    "amount_mean_1s",
    "amount_std_1s",
    "txn_count_1s",
    "amount_sum_5s",
    "amount_mean_5s",
    "amount_std_5s",
    "txn_count_5s",
    "amount_max_5s",
    "amount_sum_60s",
    "amount_mean_60s",
    "amount_std_60s",
    "txn_count_60s",
    "amount_max_60s",
    "amount_min_60s",
    "amount_zscore",
    "velocity_change_ratio",
    "unique_merchants_60s",
    "geo_distance_delta",
    "current_amount",
]
N_FEATURES = len(FEATURE_NAMES)


def _sample_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate feature rows for legitimate transactions."""
    mean_amount = rng.uniform(10, 100, size=n)
    std_amount = mean_amount * rng.uniform(0.1, 0.5, size=n)

    txn_1s = rng.integers(0, 1, size=n).astype(float)
    txn_5s = rng.integers(0, 2, size=n).astype(float)
    txn_60s = rng.integers(1, 5, size=n).astype(float)

    cur_amt = np.abs(rng.normal(mean_amount, std_amount))
    amt_1s = cur_amt * txn_1s
    amt_5s = cur_amt * txn_5s
    amt_60s = cur_amt * txn_60s

    zscore = rng.uniform(-1, 1.5, size=n)
    velocity = txn_1s / np.maximum(txn_60s, 1) * 60.0
    merchants = rng.integers(1, 5, size=n).astype(float)
    geo_dist = rng.exponential(10, size=n)

    return np.column_stack(
        [
            amt_1s,
            amt_1s / np.maximum(txn_1s, 1),
            np.zeros(n), # Force XGBoost to ignore 1s std
            txn_1s,
            amt_5s,
            amt_5s / np.maximum(txn_5s, 1),
            np.zeros(n), # Force XGBoost to ignore 5s std
            txn_5s,
            amt_5s * rng.uniform(1, 1.5, n),
            amt_60s,
            amt_60s / np.maximum(txn_60s, 1),
            std_amount,
            txn_60s,
            amt_60s * rng.uniform(1, 1.5, n),
            cur_amt * rng.uniform(0.5, 1, n),
            np.zeros(n), # Force XGBoost to ignore zscore
            velocity,
            merchants,
            geo_dist,
            cur_amt,
        ]
    )

def _sample_fraud(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate feature rows for fraudulent transactions."""
    mean_amount = rng.uniform(500, 5000, size=n)
    std_amount = mean_amount * rng.uniform(0.2, 0.8, size=n)

    txn_1s = rng.integers(0, 2, size=n).astype(float)
    txn_5s = rng.integers(2, 8, size=n).astype(float)
    txn_60s = rng.integers(10, 60, size=n).astype(float)

    cur_amt = np.abs(rng.normal(mean_amount, std_amount))
    amt_1s = cur_amt * txn_1s
    amt_5s = cur_amt * txn_5s
    amt_60s = cur_amt * txn_60s

    zscore = rng.uniform(2, 15, size=n)
    velocity = txn_1s / np.maximum(txn_60s, 1) * 60.0
    merchants = rng.integers(1, 4, size=n).astype(float)
    geo_dist = rng.exponential(500, size=n)

    return np.column_stack(
        [
            amt_1s,
            amt_1s / np.maximum(txn_1s, 1),
            np.zeros(n), # Force XGBoost to ignore 1s std
            txn_1s,
            amt_5s,
            amt_5s / np.maximum(txn_5s, 1),
            np.zeros(n), # Force XGBoost to ignore 5s std
            txn_5s,
            amt_5s * rng.uniform(1, 2, n),
            amt_60s,
            amt_60s / np.maximum(txn_60s, 1),
            std_amount * 3,
            txn_60s,
            amt_60s * rng.uniform(1, 2, n),
            cur_amt * rng.uniform(0.1, 0.8, n),
            np.zeros(n), # Force XGBoost to ignore zscore
            velocity,
            merchants,
            geo_dist,
            cur_amt,
        ]
    )


def generate_dataset(
    n_samples: int = 100_000,
    fraud_rate: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    X_legit = _sample_normal(n_legit, rng)
    X_fraud = _sample_fraud(n_fraud, rng)
    y_legit = np.zeros(n_legit)
    y_fraud = np.ones(n_fraud)

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([y_legit, y_fraud])

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train(output_dir: str = "models") -> None:
    Path(output_dir).mkdir(exist_ok=True)

    # ==============================================================================
    # LOCAL VS PRODUCTION TUNING GUIDE
    # ==============================================================================
    # This script generates synthetic data to train the XGBoost fraud model.
    # The `txn_1s` and `txn_5s` parameters below MUST match the physical throughput
    # (Requests Per Second) that your hardware is capable of processing.
    #
    # LOCAL TESTING (Current Settings):
    # If running on a local laptop, the Python CPU bottleneck restricts throughput 
    # to ~100-200 RPS. At this speed, fraudsters can only generate ~5 transactions 
    # per 5 seconds. The parameters below are tuned down so the model can correctly 
    # identify these slow bursts as fraud.
    #
    # PRODUCTION DEPLOYMENT (Hyperscale):
    # If deploying to a cloud cluster (e.g., AWS EC2) capable of 1000+ RPS, you must
    # increase the synthetic burst density to match production fraudster behavior:
    #   Normal users: txn_1s = (2, 8), txn_5s = (10, 30)
    #   Fraudsters:   txn_1s = (5, 15), txn_5s = (20, 60)
    #   XGBoost:      max_depth=8, learning_rate=0.01, n_estimators=1000
    # ==============================================================================

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Generating synthetic dataset …")
    X, y = generate_dataset(n_samples=200_000, fraud_rate=0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(
        "Train: %d  Test: %d  Fraud rate: %.2f%%", len(X_train), len(X_test), 100 * y.mean()
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    try:
        from xgboost import XGBClassifier

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info("Training XGBoost (scale_pos_weight=%.1f) …", scale_pos_weight)
        t0 = time.time()
        model = XGBClassifier(
            max_depth=6,
            learning_rate=0.3,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=20,
        )
        logger.info("XGBoost trained in %.1fs", time.time() - t0)

        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        logger.info("XGBoost ROC-AUC: %.4f  PR-AUC: %.4f", roc, pr_auc)
        print(
            classification_report(
                y_test, (y_prob > 0.5).astype(int), target_names=["legit", "fraud"]
            )
        )

        importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        logger.info("Top features: %s", top5)

        payload = {
            "model": model,
            "scaler": scaler,
            "feature_names": FEATURE_NAMES,
            "model_type": "xgboost",
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "threshold": 0.5,
            "version": "1.0.0",
        }
        path = os.path.join(output_dir, "fraud_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Model saved → %s", path)

    except ImportError:
        logger.warning("xgboost not installed. Falling back to LogisticRegression.")
        _train_logreg(X_train_s, X_test_s, y_train, y_test, scaler, output_dir)


def _train_logreg(X_train, X_test, y_train, y_test, scaler, output_dir) -> None:
    from sklearn.linear_model import LogisticRegression

    logger.info("Training LogisticRegression …")
    model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    logger.info("LR ROC-AUC: %.4f  PR-AUC: %.4f", roc, pr_auc)
    print(
        classification_report(y_test, (y_prob > 0.5).astype(int), target_names=["legit", "fraud"])
    )

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "model_type": "logistic_regression",
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "threshold": 0.5,
        "version": "1.0.0",
    }
    path = os.path.join(output_dir, "fraud_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Model saved → %s", path)


if __name__ == "__main__":
    train()
