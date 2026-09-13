import json

import numpy as np

from trace_ace.xgb_head import beta_calibrate, feature_matrix, load_trials


def test_frozen_trials_and_feature_matrix(tmp_path):
    trials_path = tmp_path / "trials.json"
    trials_path.write_text(json.dumps([{"trial_id": f"hidden-{i:03d}", "feature_set": "hidden"} for i in range(5)]))
    trials = load_trials(trials_path)
    hidden = np.ones((3, 4), dtype=np.float32)
    logits = np.zeros((3, 2), dtype=np.float32)
    assert feature_matrix(trials[0], hidden, logits).shape == (3, 4)


def test_xgb_beta_calibration_is_finite_and_bounded():
    calibrated = beta_calibrate(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0.0) & (calibrated < 1.0)).all()
