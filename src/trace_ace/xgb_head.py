from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BETA = {"a": 0.5841194397814309, "b": 1.1992585849892892, "bias": -0.41520205179065706}
TRIALS_PATH = Path(__file__).resolve().parents[2] / "configs" / "xgb_frozen_trials.json"


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def beta_calibrate(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return sigmoid(BETA["a"] * np.log(p) - BETA["b"] * np.log1p(-p) + BETA["bias"])


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    response_id = archive["response_id"].astype(str)
    hidden = np.asarray(archive["hidden"], dtype=np.float32)
    logits = np.asarray(archive["logits"], dtype=np.float32)
    if hidden.ndim != 2 or logits.shape != (len(hidden), 2) or len(response_id) != len(hidden):
        raise ValueError("feature archive has inconsistent shapes")
    if len(np.unique(response_id)) != len(response_id) or not np.isfinite(hidden).all() or not np.isfinite(logits).all():
        raise ValueError("feature archive has duplicate IDs or non-finite values")
    return response_id, hidden, logits


def feature_matrix(trial: dict[str, object], hidden: np.ndarray, logits: np.ndarray) -> np.ndarray:
    if trial["feature_set"] == "hidden":
        return hidden
    if trial["feature_set"] == "hidden_logits":
        return np.column_stack([hidden, logits]).astype(np.float32, copy=False)
    raise ValueError(f"unknown feature set: {trial['feature_set']}")


def xgb_parameters(trial: dict[str, object], device: str) -> dict[str, object]:
    excluded = {"trial_id", "best_iteration", "feature_set", "label_smoothing"}
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": device,
        "max_bin": 256,
        "seed": 20260805 + int(str(trial["trial_id"]).rsplit("-", 1)[1]),
        "validate_parameters": True,
        **{key: value for key, value in trial.items() if key not in excluded},
    }


def load_trials(path: Path) -> list[dict[str, object]]:
    trials = json.loads(path.read_text(encoding="utf-8"))
    if len(trials) != 5 or len({trial["trial_id"] for trial in trials}) != 5:
        raise ValueError("the frozen ensemble must contain five unique trials")
    return trials


def train(args: argparse.Namespace) -> None:
    import xgboost as xgb

    response_id, hidden, logits = load_features(args.features)
    labels = pd.read_csv(args.labels, dtype={"response_id": str})[["response_id", "is_correct"]]
    aligned = pd.DataFrame({"response_id": response_id}).merge(labels, on="response_id", validate="one_to_one")
    if len(aligned) != len(response_id):
        raise ValueError("labels do not cover every feature ID")
    y = aligned["is_correct"].to_numpy(dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for trial in load_trials(args.trials):
        matrix = feature_matrix(trial, hidden, logits)
        smoothing = float(trial["label_smoothing"])
        target = y * (1.0 - smoothing) + 0.5 * smoothing
        booster = xgb.train(
            xgb_parameters(trial, args.device),
            xgb.QuantileDMatrix(matrix, label=target, max_bin=256),
            num_boost_round=int(trial["best_iteration"]) + 1,
        )
        filename = f"{trial['trial_id']}.ubj"
        booster.save_model(str(args.output / filename))
        manifest.append({"model": filename, **trial})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def predict(args: argparse.Namespace) -> None:
    import xgboost as xgb

    response_id, hidden, logits = load_features(args.features)
    manifest = json.loads((args.models / "manifest.json").read_text(encoding="utf-8"))
    predictions = []
    for trial in manifest:
        booster = xgb.Booster()
        booster.load_model(str(args.models / trial["model"]))
        predictions.append(booster.inplace_predict(feature_matrix(trial, hidden, logits)))
    raw = np.mean(predictions, axis=0)
    output = pd.DataFrame({"response_id": response_id, "probability_raw": raw, "probability": beta_calibrate(raw)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the frozen hidden-state XGBoost ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--features", type=Path, required=True)
    train_parser.add_argument("--labels", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--trials", type=Path, default=TRIALS_PATH)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.set_defaults(function=train)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--features", type=Path, required=True)
    predict_parser.add_argument("--models", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.set_defaults(function=predict)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
