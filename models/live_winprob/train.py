import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Basic settings and file paths
MODEL_DIR = Path("models/live_winprob")
FEATURES = [
    "minute", "minutes_remaining", "score_diff", "xg_diff_roll",
    "manpower_diff", "xg_share", "is_active_game",
    "score_manpower_interaction", "score_pressure", "goal_momentum"
]
MINUTE_BUCKETS = [0, 15, 30, 45, 60, 75, 91]
HOLDOUT_MIN_MATCH_WEEK = 33

def load_data():
    snapshots = pd.read_csv(PROCESSED_DATA_DIR / "live_snapshots.csv")
    matches = pd.read_csv(PROCESSED_DATA_DIR / "prematch_matches.csv")[["match_id", "match_week"]]
    return snapshots.merge(matches, on="match_id", how="left").dropna(subset=["match_week"])

def save_calibration_plot(model, name, holdout, encoder, output_path):
    X, y = holdout[FEATURES], encoder.transform(holdout["final_outcome"])
    probs = model.predict_proba(X)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    for i, class_name in enumerate(encoder.classes_):
        p_true, p_pred = calibration_curve((y == i).astype(int), probs[:, i], n_bins=10)
        plt.plot(p_pred, p_true, "s-", label=f"Outcome: {class_name}")
    
    plt.title(f"Calibration Curve: {name.replace('_', ' ').title()}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    df = load_data()
    # Keep the later matches aside for testing so the model is judged on new data.
    train = df[df["match_week"] < HOLDOUT_MIN_MATCH_WEEK]
    holdout = df[df["match_week"] >= HOLDOUT_MIN_MATCH_WEEK]
    
    encoder = LabelEncoder().fit(df["final_outcome"])
    X_train, y_train = train[FEATURES], encoder.transform(train["final_outcome"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "xgboost": XGBClassifier(
            objective="multi:softprob", num_class=len(encoder.classes_),
            max_depth=4, n_estimators=200, learning_rate=0.05, n_jobs=-1
        )
    }

    comparison = {}
    for name, model in models.items():
        log.info(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump({"model": model, "encoder": encoder, "features": FEATURES}, MODEL_DIR / f"{name}_model.pkl")
        
        save_calibration_plot(model, name, holdout, encoder, MODEL_DIR / f"{name}_calibration.png")
        comparison[name] = {"overall_log_loss": log_loss(encoder.transform(holdout["final_outcome"]), model.predict_proba(holdout[FEATURES]))}

    # Take a quick look at which features matter most in the XGBoost model.
    xgb_model = models["xgboost"]
    importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    
    # Print a simple summary to the terminal so it is easy to compare the models.
    winner = "xgboost" if comparison["xgboost"]["overall_log_loss"] < comparison["logistic_regression"]["overall_log_loss"] else "logistic_regression"
    
    print(f"\n{'='*40}\nMODEL COMPARISON SUMMARY\n{'='*40}")
    for name in models:
        print(f"{name:20} | Log Loss: {comparison[name]['overall_log_loss']:.4f}")
    
    print(f"\n{'='*40}\nTOP 5 FEATURES (XGBOOST)\n{'='*40}")
    print(importances.head(5).to_string())
    
    print(f"\n{'='*40}\nWINNER: {winner.upper()}\n{'='*40}\n")
    
    with open(MODEL_DIR / "comparison_report.json", "w") as f:
        json.dump({**comparison, "winner": winner, "feature_importance": importances.to_dict()}, f, indent=2)

if __name__ == "__main__":
    main()