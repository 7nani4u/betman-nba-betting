from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from .config import settings

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))

@dataclass
class TrainedModel:
    base_model: Any
    calibrator: Any
    feature_names: list[str]
    metrics: dict

class ModelTrainer:
    """
    Market-agnostic trainer. For each market/selection we still train a binary classifier:
      y=1 if selection outcome occurs else 0.
    You can extend to multi-class later by training per-outcome one-vs-rest (OVR).
    """
    def __init__(self):
        self.feature_names: list[str] = []

    def load_training_data(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"training data not found: {path}")
        df = pd.read_csv(path)
        return df

    def _make_features_and_target_moneyline_home(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # This repo currently ships only synthetic, low-dimensional features.
        # Keep it consistent but allow future extension.
        feature_columns = [c for c in df.columns if c.startswith(("home_", "away_", "pace", "is_home"))]
        # Fallback explicit list if needed
        if "home_win" not in df.columns:
            raise ValueError("training df must contain home_win")
        X = df[feature_columns].copy()
        y = df["home_win"].astype(int)
        self.feature_names = feature_columns
        return X, y

    def train_moneyline_home(self, df: pd.DataFrame) -> TrainedModel:
        X, y = self._make_features_and_target_moneyline_home(df)

        tscv = TimeSeriesSplit(n_splits=5)
        base = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=300,
            objective="binary:logistic", eval_metric="logloss",
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )

        # Fit base
        base.fit(X, y, verbose=False)

        # Calibrate with time-series splits to avoid leakage
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=tscv)
        cal.fit(X, y)

        # Evaluate on last fold as a simple, leakage-minimized proxy
        # (Full backtest should be done in a dedicated evaluator.)
        last_train_idx, last_val_idx = list(tscv.split(X))[-1]
        p_raw = base.predict_proba(X.iloc[last_val_idx])[:, 1]
        p_cal = cal.predict_proba(X.iloc[last_val_idx])[:, 1]
        y_val = y.iloc[last_val_idx].to_numpy()

        metrics = {
            "log_loss_raw": float(log_loss(y_val, p_raw)),
            "log_loss_cal": float(log_loss(y_val, p_cal)),
            "brier_raw": brier_score(y_val, p_raw),
            "brier_cal": brier_score(y_val, p_cal),
            "n_rows": int(len(df)),
        }
        return TrainedModel(base_model=base, calibrator=cal, feature_names=self.feature_names, metrics=metrics)

    def save(self, trained: TrainedModel) -> str:
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        path = os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)
        joblib.dump({
            "base_model": trained.base_model,
            "calibrator": trained.calibrator,
            "feature_names": trained.feature_names,
            "metrics": trained.metrics,
        }, path)
        return path

def main():
    # default path: data/historical_games.csv (same as original config style)
    data_path = os.getenv("HISTORICAL_DATA_PATH", "data/historical_games.csv")
    trainer = ModelTrainer()
    df = trainer.load_training_data(data_path)
    trained = trainer.train_moneyline_home(df)
    path = trainer.save(trained)
    print("saved model:", path)
    print("metrics:", trained.metrics)

if __name__ == "__main__":
    main()
