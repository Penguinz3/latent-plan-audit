import numpy as np
import pytest

from latent_score_decomposition import (
    CANCELLATION_OFFSETTING,
    CLEAN_PRESERVATION,
    INDETERMINATE_ABSTAIN,
    OBJECTIVE_MEDIATED_HARMFUL_REVERSAL,
    ROLLOUT_MEDIATED_HARMFUL_REVERSAL,
    classify_pair,
    decompose_latent_scores,
    pair_margin_audit,
    pair_regret,
    pair_taxonomy,
    reconstruction_check,
)


def test_visual_decomposition_reconstructs_exact_score_error_per_candidate():
    predicted = np.array([[2.0, -1.0], [0.5, 3.0]])
    realized = np.array([[1.0, 0.0], [1.5, 2.0]])
    goal = np.array([0.0, 1.0])

    result = decompose_latent_scores(predicted, realized, goal, alpha=0.1)

    expected_p = np.mean((predicted - goal) ** 2, axis=1)
    expected_o = np.mean((realized - goal) ** 2, axis=1)
    np.testing.assert_allclose(result["predicted_score"], expected_p)
    np.testing.assert_allclose(result["observed_score"], expected_o)
    np.testing.assert_allclose(
        result["score_error"],
        result["directional_correction"] + result["magnitude_correction"],
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(result["reconstruction_residual"], 0.0, atol=1e-14)
    assert reconstruction_check(result)


def test_proprioception_is_alpha_weighted_and_reconstructs_componentwise():
    result = decompose_latent_scores(
        predicted_visual=np.array([[2.0], [3.0]]),
        realized_visual=np.array([[1.0], [1.0]]),
        goal_visual=np.array([0.0]),
        predicted_proprio=np.array([[4.0], [2.0]]),
        realized_proprio=np.array([[2.0], [3.0]]),
        goal_proprio=np.array([0.0]),
        alpha=0.25,
    )

    assert result["predicted_score"].tolist() == pytest.approx([8.0, 10.0])
    assert result["observed_score"].tolist() == pytest.approx([2.0, 3.25])
    assert result["weighted_predicted_proprio_mse"].tolist() == pytest.approx([4.0, 1.0])
    assert result["weighted_observed_proprio_mse"].tolist() == pytest.approx([1.0, 2.25])
    np.testing.assert_allclose(result["proprio_reconstruction_residual"], 0.0)
    np.testing.assert_allclose(
        result["reconstruction_residual"], 0.0, atol=1e-14, rtol=0.0
    )


def test_pair_margin_identity_and_reversal_are_reported():
    result = pair_margin_audit(
        predicted_scores=np.array([1.0, 1.4]),
        observed_scores=np.array([1.6, 1.1]),
    )

    assert result["predicted_index"] == 0
    assert result["baseline_index"] == 1
    assert result["predicted_margin"] == pytest.approx(0.4)
    assert result["relative_correction"] == pytest.approx(0.9)
    assert result["observed_margin"] == pytest.approx(-0.5)
    assert result["observed_ranking_reversal"]
    assert result["identity_residual"] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    ("predicted", "observed", "native", "expected"),
    [
        ([1.0, 2.0], [1.2, 2.2], [0.8, 0.2], CLEAN_PRESERVATION),
        ([1.0, 2.0], [2.2, 1.2], [0.2, 0.8], ROLLOUT_MEDIATED_HARMFUL_REVERSAL),
        ([1.0, 2.0], [1.2, 2.2], [0.2, 0.8], OBJECTIVE_MEDIATED_HARMFUL_REVERSAL),
        ([1.0, 2.0], [2.2, 1.2], [0.8, 0.2], CANCELLATION_OFFSETTING),
    ],
)
def test_pair_taxonomy_matches_all_strict_binary_patterns(
    predicted, observed, native, expected
):
    report = pair_taxonomy(predicted, observed, native)
    assert report["label"] == expected
    assert np.isfinite(report["R_P"]) and np.isfinite(report["R_O"])


def test_ties_abstain_and_regret_is_undefined():
    tied = pair_taxonomy([1.0, 1.0], [1.0, 2.0], [0.0, 1.0])
    assert tied["label"] == INDETERMINATE_ABSTAIN
    assert np.isnan(tied["R_P"]) and np.isnan(tied["R_O"])
    assert np.isnan(pair_regret([0.0, 1.0], [1.0, 1.0]))
    assert classify_pair([np.nan, 1.0], [1.0, 2.0], [0.0, 1.0]) == INDETERMINATE_ABSTAIN


def test_protocol_negative_return_cost_orientation_matches_return_orientation():
    returns = np.array([0.2, 0.8])
    costs = -returns
    assert classify_pair([1.0, 2.0], [1.2, 1.8], returns) == OBJECTIVE_MEDIATED_HARMFUL_REVERSAL
    assert classify_pair(
        [1.0, 2.0], [1.2, 1.8], costs, native_higher_better=False
    ) == OBJECTIVE_MEDIATED_HARMFUL_REVERSAL
    assert pair_regret(costs, [1.0, 2.0], native_higher_better=False) == pytest.approx(0.6)
