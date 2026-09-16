"""Exact, decision-relevant decomposition of a latent MSE planning score.

For a predicted latent ``zhat``, realized encoded latent ``z``, and goal ``g``:

    mean((zhat - g)^2) - mean((z - g)^2)
      = 2 * mean((z - g) * (zhat - z)) + mean((zhat - z)^2)

The first term is the signed goal-directed error and the second is its squared
magnitude. Visual MSE is mandatory; proprioceptive MSE is optional and uses
the official ``alpha`` weight (0.1 by default).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_TIE_ATOL = 1e-10
DEFAULT_TIE_RTOL = 1e-8

CLEAN_PRESERVATION = "clean_preservation"
ROLLOUT_MEDIATED_HARMFUL_REVERSAL = "rollout_mediated_harmful_reversal"
OBJECTIVE_MEDIATED_HARMFUL_REVERSAL = "objective_mediated_harmful_reversal"
CANCELLATION_OFFSETTING = "cancellation_offsetting"
INDETERMINATE_ABSTAIN = "indeterminate_abstain"


def _array(value: Any, name: str, *, one_dimensional: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim == 0 or (one_dimensional and result.ndim != 1):
        expected = "one-dimensional" if one_dimensional else "non-scalar"
        raise ValueError(f"{name} must be a finite {expected} array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _candidate_array(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    # A 1-D latent is one candidate with many features. Scalar candidates
    # should be passed as shape (candidates, 1) to make the axis unambiguous.
    return result[None, :] if result.ndim == 1 else result


def _component(
    predicted: Any, realized: Any, goal: Any, name: str
) -> dict[str, np.ndarray]:
    prediction = _candidate_array(predicted, f"predicted_{name}")
    observation = _candidate_array(realized, f"realized_{name}")
    if prediction.shape != observation.shape:
        raise ValueError(f"predicted_{name} and realized_{name} must have the same shape")
    target = _array(goal, f"goal_{name}") if np.ndim(goal) else np.asarray(goal, dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError(f"goal_{name} must be finite")
    try:
        predicted_delta = prediction - target
        observed_delta = observation - target
    except ValueError as exc:
        raise ValueError(f"goal_{name} is not broadcastable to latent shape") from exc
    if predicted_delta.shape != prediction.shape or observed_delta.shape != observation.shape:
        raise ValueError(f"goal_{name} is not broadcastable to latent shape")

    residual = prediction - observation
    baseline = observation - target
    axes = tuple(range(1, prediction.ndim))
    predicted_mse = np.mean(np.square(predicted_delta), axis=axes)
    observed_mse = np.mean(np.square(observed_delta), axis=axes)
    directional = 2.0 * np.mean(baseline * residual, axis=axes)
    magnitude = np.mean(np.square(residual), axis=axes)
    error = predicted_mse - observed_mse
    return {
        "predicted_mse": np.asarray(predicted_mse),
        "observed_mse": np.asarray(observed_mse),
        "score_error": np.asarray(error),
        "directional_correction": np.asarray(directional),
        "magnitude_correction": np.asarray(magnitude),
        "reconstruction_residual": np.asarray(error - directional - magnitude),
    }


def _alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("alpha must be a finite non-negative number")
    return float(alpha)


def decompose_latent_scores(
    predicted_visual: Any,
    realized_visual: Any,
    goal_visual: Any,
    *,
    predicted_proprio: Any | None = None,
    realized_proprio: Any | None = None,
    goal_proprio: Any | None = None,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Return per-candidate P/O scores and their exact error decomposition.

    Candidate index is axis zero and all remaining axes are averaged by MSE.
    ``P`` is the predicted cost; ``O`` is the identical cost on realized
    encoded latents. The proprioceptive arguments must be supplied together.
    """
    weight = _alpha(alpha)
    visual = _component(predicted_visual, realized_visual, goal_visual, "visual")
    proprio_given = (
        predicted_proprio is not None,
        realized_proprio is not None,
        goal_proprio is not None,
    )
    if any(proprio_given) and not all(proprio_given):
        raise ValueError("proprioceptive predicted, realized, and goal arrays must be supplied together")

    proprio = None
    if all(proprio_given):
        proprio = _component(
            predicted_proprio, realized_proprio, goal_proprio, "proprio"
        )
        if proprio["predicted_mse"].shape != visual["predicted_mse"].shape:
            raise ValueError("visual and proprioceptive arrays must have the same candidate count")

    def total(key: str) -> np.ndarray:
        value = visual[key].copy()
        if proprio is not None:
            source = "predicted_mse" if key == "predicted_mse" else (
                "observed_mse" if key == "observed_mse" else key
            )
            value += weight * proprio[source]
        return value

    predicted_score = total("predicted_mse")
    observed_score = total("observed_mse")
    score_error = predicted_score - observed_score
    directional = total("directional_correction")
    magnitude = total("magnitude_correction")
    result: dict[str, Any] = {
        "predicted_score": predicted_score,
        "observed_score": observed_score,
        "score_error": score_error,
        "directional_correction": directional,
        "magnitude_correction": magnitude,
        "reconstruction_residual": score_error - directional - magnitude,
        "reconstruction_ok": np.isclose(
            score_error - directional - magnitude, 0.0, atol=1e-12, rtol=1e-12
        ),
        "predicted_visual_mse": visual["predicted_mse"],
        "observed_visual_mse": visual["observed_mse"],
        "visual_score_error": visual["score_error"],
        "visual_directional_correction": visual["directional_correction"],
        "visual_magnitude_correction": visual["magnitude_correction"],
        "visual_reconstruction_residual": visual["reconstruction_residual"],
        "alpha": weight,
        "has_proprio": proprio is not None,
    }
    if proprio is None:
        for key in (
            "predicted_proprio_mse",
            "observed_proprio_mse",
            "proprio_score_error",
            "proprio_directional_correction",
            "proprio_magnitude_correction",
            "proprio_reconstruction_residual",
            "weighted_predicted_proprio_mse",
            "weighted_observed_proprio_mse",
            "weighted_proprio_score_error",
            "weighted_proprio_directional_correction",
            "weighted_proprio_magnitude_correction",
            "weighted_proprio_reconstruction_residual",
        ):
            result[key] = None
    else:
        result.update(
            {
                "predicted_proprio_mse": proprio["predicted_mse"],
                "observed_proprio_mse": proprio["observed_mse"],
                "proprio_score_error": proprio["score_error"],
                "proprio_directional_correction": proprio[
                    "directional_correction"
                ],
                "proprio_magnitude_correction": proprio["magnitude_correction"],
                "proprio_reconstruction_residual": proprio[
                    "reconstruction_residual"
                ],
                "weighted_predicted_proprio_mse": weight * proprio[
                    "predicted_mse"
                ],
                "weighted_observed_proprio_mse": weight * proprio[
                    "observed_mse"
                ],
                "weighted_proprio_score_error": weight * proprio["score_error"],
                "weighted_proprio_directional_correction": weight
                * proprio["directional_correction"],
                "weighted_proprio_magnitude_correction": weight
                * proprio["magnitude_correction"],
                "weighted_proprio_reconstruction_residual": weight
                * proprio["reconstruction_residual"],
            }
        )
    return result


def reconstruction_check(
    decomposition: dict[str, Any], *, atol: float = 1e-12, rtol: float = 1e-12
) -> bool:
    """Check that a decomposition reconstructs every candidate's P-minus-O."""
    if (
        isinstance(atol, bool)
        or isinstance(rtol, bool)
        or not math.isfinite(float(atol))
        or not math.isfinite(float(rtol))
        or float(atol) < 0.0
        or float(rtol) < 0.0
    ):
        raise ValueError("reconstruction tolerances must be finite and non-negative")
    try:
        residual = np.asarray(decomposition["reconstruction_residual"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("decomposition must contain reconstruction_residual") from exc
    return bool(
        residual.ndim == 1
        and np.isfinite(residual).all()
        and np.allclose(residual, 0.0, atol=float(atol), rtol=float(rtol))
    )


def _tolerances(atol: float, rtol: float) -> tuple[float, float]:
    if (
        isinstance(atol, bool)
        or isinstance(rtol, bool)
        or not math.isfinite(float(atol))
        or not math.isfinite(float(rtol))
        or float(atol) < 0.0
        or float(rtol) < 0.0
    ):
        raise ValueError("tie tolerances must be finite and non-negative")
    return float(atol), float(rtol)


def _tie(left: float, right: float, atol: float, rtol: float) -> bool:
    return abs(left - right) <= atol + rtol * max(1.0, abs(left), abs(right))


def _scores(value: Any, name: str) -> np.ndarray:
    return _array(value, name, one_dimensional=True)


def _pair_indices(count: int, a: int, b: int) -> tuple[int, int]:
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, (int, np.integer))
        or not isinstance(b, (int, np.integer))
        or not 0 <= int(a) < count
        or not 0 <= int(b) < count
        or int(a) == int(b)
    ):
        raise ValueError("candidate indices must be distinct valid indices")
    return int(a), int(b)


def pair_margin_audit(
    predicted_scores: Any,
    observed_scores: Any,
    *,
    predicted_index: int | None = None,
    baseline_index: int | None = None,
    tie_atol: float = DEFAULT_TIE_ATOL,
    tie_rtol: float = DEFAULT_TIE_RTOL,
) -> dict[str, Any]:
    """Audit ``O_b-O_p = (P_b-P_p) - (e_b-e_p)`` exactly."""
    atol, rtol = _tolerances(tie_atol, tie_rtol)
    predicted = _scores(predicted_scores, "predicted_scores")
    observed = _scores(observed_scores, "observed_scores")
    if predicted.shape != observed.shape or predicted.size < 2:
        raise ValueError("predicted_scores and observed_scores must match with at least two candidates")
    p = int(np.argmin(predicted)) if predicted_index is None else int(predicted_index)
    if not 0 <= p < predicted.size:
        raise ValueError("predicted_index must be valid")
    if baseline_index is None:
        remaining = np.arange(predicted.size) != p
        b = int(np.flatnonzero(remaining)[np.argmin(predicted[remaining])])
    else:
        b = int(baseline_index)
        if not 0 <= b < predicted.size or b == p:
            raise ValueError("baseline_index must be valid and distinct")
    error = predicted - observed
    margin = float(predicted[b] - predicted[p])
    correction = float(error[b] - error[p])
    observed_margin = float(observed[b] - observed[p])
    return {
        "predicted_index": p,
        "baseline_index": b,
        "predicted_margin": margin,
        "relative_correction": correction,
        "observed_margin": observed_margin,
        "identity_residual": float(observed_margin - (margin - correction)),
        "prediction_error": error,
        "predicted_tie": _tie(predicted[p], predicted[b], atol, rtol),
        "observed_tie": _tie(observed[p], observed[b], atol, rtol),
        "observed_ranking_reversal": (
            not _tie(predicted[p], predicted[b], atol, rtol)
            and not _tie(observed[p], observed[b], atol, rtol)
            and observed[b] < observed[p]
        ),
    }


def pair_regret(
    native_values: Any,
    predicted_scores: Any,
    *,
    candidate_a: int = 0,
    candidate_b: int = 1,
    native_higher_better: bool = True,
    tie_atol: float = DEFAULT_TIE_ATOL,
    tie_rtol: float = DEFAULT_TIE_RTOL,
) -> float:
    """Return native regret for the predicted choice on a pair.

    Native values are higher-better returns by default. Set
    ``native_higher_better=False`` for the protocol's lower-better ``N=-J_H``.
    A score tie returns NaN because the protocol abstains unless an external
    planner tie-break is explicitly part of the audited system.
    """
    if not isinstance(native_higher_better, (bool, np.bool_)):
        raise ValueError("native_higher_better must be boolean")
    atol, rtol = _tolerances(tie_atol, tie_rtol)
    native = _scores(native_values, "native_values")
    predicted = _scores(predicted_scores, "predicted_scores")
    if native.shape != predicted.shape or native.size < 2:
        raise ValueError("native_values and predicted_scores must match with at least two candidates")
    a, b = _pair_indices(native.size, candidate_a, candidate_b)
    if _tie(predicted[a], predicted[b], atol, rtol):
        return float("nan")
    chosen = a if predicted[a] < predicted[b] else b
    if native_higher_better:
        return max(0.0, max(native[a], native[b]) - native[chosen])
    return max(0.0, native[chosen] - min(native[a], native[b]))


def pair_taxonomy(
    predicted_scores: Any,
    observed_scores: Any,
    native_values: Any,
    *,
    candidate_a: int = 0,
    candidate_b: int = 1,
    native_higher_better: bool = True,
    tie_atol: float = DEFAULT_TIE_ATOL,
    tie_rtol: float = DEFAULT_TIE_RTOL,
) -> dict[str, Any]:
    """Classify strict P/O/N ordering and report ``R_P`` and ``R_O``.

    P and O are lower-better costs. Native values are higher-better returns
    by default; pass ``native_higher_better=False`` for lower-better N costs.
    Any numerical tie or non-finite input abstains and returns NaN regrets.
    """
    if not isinstance(native_higher_better, (bool, np.bool_)):
        raise ValueError("native_higher_better must be boolean")
    atol, rtol = _tolerances(tie_atol, tie_rtol)
    try:
        predicted = np.asarray(predicted_scores, dtype=np.float64)
        observed = np.asarray(observed_scores, dtype=np.float64)
        native = np.asarray(native_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be one-dimensional numeric arrays") from exc
    if any(value.ndim != 1 for value in (predicted, observed, native)):
        raise ValueError("scores must be one-dimensional numeric arrays")
    if predicted.shape != observed.shape or predicted.shape != native.shape:
        raise ValueError("predicted_scores, observed_scores, and native_values must match")
    if predicted.size < 2:
        raise ValueError("at least two candidates are required")
    a, b = _pair_indices(predicted.size, candidate_a, candidate_b)
    if not all(np.isfinite(value).all() for value in (predicted, observed, native)):
        return {
            "label": INDETERMINATE_ABSTAIN,
            "predicted_preference": None,
            "observed_preference": None,
            "native_preference": None,
            "predicted_tie": False,
            "observed_tie": False,
            "native_tie": False,
            "invalid": True,
            "R_P": float("nan"),
            "R_O": float("nan"),
        }

    def preference(left: float, right: float, lower_better: bool) -> int | None:
        if _tie(left, right, atol, rtol):
            return None
        if lower_better:
            return a if left < right else b
        return a if left > right else b

    p_pref = preference(predicted[a], predicted[b], True)
    o_pref = preference(observed[a], observed[b], True)
    n_pref = preference(native[a], native[b], not native_higher_better)
    if p_pref is None or o_pref is None or n_pref is None:
        label = INDETERMINATE_ABSTAIN
    elif p_pref == o_pref == n_pref:
        label = CLEAN_PRESERVATION
    elif p_pref != n_pref and o_pref == n_pref:
        label = ROLLOUT_MEDIATED_HARMFUL_REVERSAL
    elif p_pref != n_pref and o_pref == p_pref:
        label = OBJECTIVE_MEDIATED_HARMFUL_REVERSAL
    elif p_pref == n_pref and o_pref != p_pref:
        label = CANCELLATION_OFFSETTING
    else:
        label = INDETERMINATE_ABSTAIN

    if label == INDETERMINATE_ABSTAIN:
        regret_p = regret_o = float("nan")
    else:
        regret_p = pair_regret(
            native,
            predicted,
            candidate_a=a,
            candidate_b=b,
            native_higher_better=native_higher_better,
            tie_atol=atol,
            tie_rtol=rtol,
        )
        regret_o = pair_regret(
            native,
            observed,
            candidate_a=a,
            candidate_b=b,
            native_higher_better=native_higher_better,
            tie_atol=atol,
            tie_rtol=rtol,
        )
    return {
        "label": label,
        "predicted_preference": p_pref,
        "observed_preference": o_pref,
        "native_preference": n_pref,
        "predicted_tie": p_pref is None,
        "observed_tie": o_pref is None,
        "native_tie": n_pref is None,
        "invalid": False,
        "R_P": regret_p,
        "R_O": regret_o,
    }


def classify_pair(*args: Any, **kwargs: Any) -> str:
    """Return only the P/O/N taxonomy label."""
    return str(pair_taxonomy(*args, **kwargs)["label"])
