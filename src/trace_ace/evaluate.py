from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .calibration import beta_calibrate, robust_platt_calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen raw probabilities")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--probability-column", default="probability")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, dtype={"response_id": str})
    labels = pd.read_csv(args.labels, dtype={"response_id": str})
    frame = predictions.merge(labels[["response_id", "is_correct"]], on="response_id", validate="one_to_one")
    if len(frame) != len(predictions) or frame.isna().any().any():
        raise ValueError("prediction and label IDs are not a complete one-to-one match")
    y = frame["is_correct"].to_numpy(dtype=np.int8)
    raw = frame[args.probability_column].to_numpy(dtype=np.float64)
    endpoints = {
        "raw": raw,
        "beta": beta_calibrate(raw),
        "robust_platt": robust_platt_calibrate(raw),
    }
    result = {
        name: {"log_loss": log_loss(y, probability), "auroc": roc_auc_score(y, probability)}
        for name, probability in endpoints.items()
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
