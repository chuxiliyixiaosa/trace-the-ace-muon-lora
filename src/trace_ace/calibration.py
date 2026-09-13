from __future__ import annotations

import numpy as np


BETA_COEFFICIENT = np.asarray([0.446605868616543, 0.8419456079514123])
BETA_INTERCEPT = -0.3636169543268169
ROBUST_PLATT_COEFFICIENT = 0.6027178495
ROBUST_PLATT_INTERCEPT = 0.1141993277


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def beta_calibrate(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    design = np.column_stack([np.log(p), -np.log1p(-p)])
    return sigmoid(design @ BETA_COEFFICIENT + BETA_INTERCEPT)


def robust_platt_calibrate(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logit = np.log(p) - np.log1p(-p)
    return sigmoid(ROBUST_PLATT_COEFFICIENT * logit + ROBUST_PLATT_INTERCEPT)
