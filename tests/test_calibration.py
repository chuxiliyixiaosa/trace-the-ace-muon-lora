import numpy as np

from trace_ace.calibration import beta_calibrate, robust_platt_calibrate


def test_calibrators_are_finite_and_bounded():
    probability = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
    for calibrated in (beta_calibrate(probability), robust_platt_calibrate(probability)):
        assert np.isfinite(calibrated).all()
        assert ((calibrated > 0) & (calibrated < 1)).all()


def test_robust_platt_is_monotonic():
    probability = np.linspace(0.001, 0.999, 100)
    assert (np.diff(robust_platt_calibrate(probability)) > 0).all()
