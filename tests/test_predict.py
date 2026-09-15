import numpy as np
import pytest

from trace_ace.predict import hidden_state_index, last_nonpadding_index
from trace_ace.calibration import robust_platt_calibrate


def test_layer28_is_not_the_final_layer_of_36_block_model():
    states = np.arange(37)
    assert states[hidden_state_index(28, len(states))] == 28
    assert hidden_state_index(28, len(states)) != 36


@pytest.mark.parametrize('layer,count', [(0, 37), (-1, 37), (37, 37), (28, 28)])
def test_unavailable_layer_is_rejected(layer, count):
    with pytest.raises(ValueError):
        hidden_state_index(layer, count)


@pytest.mark.parametrize('mask,expected', [([1, 1, 0], 1), ([0, 1, 1], 2), ([1], 0)])
def test_pooling_supports_either_padding_side(mask, expected):
    assert last_nonpadding_index(np.asarray(mask)) == expected


def test_empty_pooling_is_rejected():
    with pytest.raises(ValueError):
        last_nonpadding_index(np.zeros(3))


def test_frozen_mapping_matches_paper_formula():
    p = np.asarray([.1, .5, .9])
    expected = 1 / (1 + np.exp(-(.6027178495 * np.log(p / (1 - p)) + .1141993277)))
    np.testing.assert_allclose(robust_platt_calibrate(p), expected)
