import pytest

from experiments.analyze_pusht_margin_instability import auc, distinct_action_features, reversal
from experiments.validate_pusht_near_tie_instability import fit_logistic, sigmoid

import numpy as np


def test_auc_handles_ties():
    assert auc([0, 1, 1, 2], [False, False, True, True]) == pytest.approx(0.875)


def test_distinct_action_margin_ignores_duplicate_plan():
    candidates = [
        {"physical_action_sha256": "a", "predicted_score": 1.0, "encoded_real_score": 1.2},
        {"physical_action_sha256": "a", "predicted_score": 1.0, "encoded_real_score": 0.9},
        {"physical_action_sha256": "b", "predicted_score": 1.5, "encoded_real_score": 1.4},
        {"physical_action_sha256": "c", "predicted_score": 3.0, "encoded_real_score": 2.8},
    ]
    result = distinct_action_features(candidates)
    assert result["distinct_actions"] == 3
    assert result["raw_margin"] == pytest.approx(0.5)
    assert len(result["absolute_errors"]) == 4


def test_reversal_is_strict():
    assert reversal({"P_margin_selected_minus_counterfactual": -0.2, "O_margin_selected_minus_counterfactual": 0.1})
    assert not reversal({"P_margin_selected_minus_counterfactual": -0.2, "O_margin_selected_minus_counterfactual": 0.0})
    assert not reversal(None)


def test_logistic_probability_orders_separable_examples():
    x = np.asarray([[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]])
    beta = fit_logistic(x, np.asarray([0.0, 0.0, 1.0, 1.0]), ridge=0.1)
    predictions = sigmoid(x @ beta)
    assert np.all(np.diff(predictions) > 0)
