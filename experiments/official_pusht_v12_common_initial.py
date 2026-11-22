"""PWA-PushT v1.2 common-initial-only protocol.

This module is deliberately separate from the stopped v1.1 final-inventory
path.  It contains the small, model-independent pieces needed by the v1.2
runner, supervisor, and analyzer: one seeded iteration-0 Gaussian population,
finite P/O checks, proposal-support diagnostics, and immutable v1.2 bindings.

The initial population is a proposal inventory, not a natural/final inventory.
No native outcome is used while sampling, scoring, or diagnosing support.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


V12_SCHEMA_VERSION = "PWA-PushT-v1.2"
V12_ARTIFACT_SCHEMA_VERSION = "official-pusht-decision-audit-v1.2-common-initial"
V12_ANALYSIS_VERSION = "PWA-PushT-analysis-v1.2-common-initial"
V12_CERTIFICATE_SCHEMA_VERSION = "PWA-PushT-reproducibility-certificate-v1.2-common-initial"
V12_BUNDLE_SCHEMA_VERSION = "PWA-PushT-certificate-bundle-v1.2-common-initial"
V12_SUPERVISOR_VERSION = "PWA-PushT-fresh-process-supervisor-v1.2-common-initial"
V12_SOURCE_BUNDLE_VERSION = "PWA-PushT-source-bundle-v1.2-common-initial"
V12_CERTIFICATE_ISSUANCE_SCHEMA_VERSION = "PWA-PushT-certificate-issuance-v1.2-common-initial"
V12_CERTIFICATE_ISSUANCE_RELATIVE_PATH = "experiments/official_pusht_v12_certificate_issuance.json"
V12_LOCKED_MODELS = ("official_jepa_wm", "official_dino_wm")
V12_LOCKED_SEEDS = (1, 2, 3)
V12_LOCKED_SEED = V12_LOCKED_SEEDS[0]
V12_EXCLUDED_TRAJECTORY = 7
V12_PROTOCOL = {
    "inventory": "common_initial_only",
    "proposal_source": "iteration_0_seeded_gaussian_cem_population",
    "outcome_blind": True,
    "planner_name": "cem",
    "objective_type": "L2",
    "sum_all_diffs": False,
    "alpha": 0.1,
    "iterations": 1,
    "sample_count": 300,
    "elites": 10,
    "var_scale": 1.0,
    "horizon": 6,
    "num_act_stepped": 6,
    "frameskip": 5,
    "action_skip": 1,
    "num_frames": 1,
    "num_proprios": 1,
    "action_dim": 10,
    "physical_action_dim": 2,
    "menu_seeds": list(V12_LOCKED_SEEDS),
    "score_batch_size": 8,
    "score_check_batch_size": 4,
    "score_reproduction_tolerance": 2e-5,
    "chunk_equivalence_tolerance": 2e-5,
    "decomposition_tolerance": 1e-8,
    "tie_atol": 1e-10,
    "tie_rtol": 1e-8,
}

# This bundle intentionally excludes the v1.1 manifest, certificate, and
# analyzer.  v1.2 code can reuse the numerical decomposition implementation,
# but its authorization hash and schema are independent.
V12_SOURCE_BUNDLE_RELATIVE_FILES = (
    "experiments/official_pusht_decision_audit.py",
    "experiments/official_pusht_v12_common_initial.py",
    "experiments/official_pusht_v12_batch_study.py",
    "experiments/official_pusht_v12_certificate_supervisor.py",
    "experiments/analyze_official_pusht_v12.py",
    "experiments/official_pusht_batch_study.py",
    "latent_score_decomposition.py",
    "PWA_PUSHT_V12_PREREGISTRATION.md",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v12_protocol_sha256() -> str:
    return canonical_sha256(V12_PROTOCOL)


def v12_source_bundle(root: Path | None = None) -> dict[str, Any]:
    project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    return {
        "version": V12_SOURCE_BUNDLE_VERSION,
        "files": {
            relative: file_sha256(project_root / relative)
            for relative in V12_SOURCE_BUNDLE_RELATIVE_FILES
        },
    }


def v12_source_bundle_sha256(root: Path | None = None) -> str:
    bundle = v12_source_bundle(root)
    missing = [name for name, digest in bundle["files"].items() if not digest]
    if missing:
        raise ValueError("v1.2 source bundle is incomplete: " + ", ".join(missing))
    return canonical_sha256(bundle)


def validate_v12_args(*, planning_repeats: int = 0, repeat_count: int = 1) -> None:
    """Validate invocation arguments before importing/loading a model.

    ``planning_repeats=0`` is the scheduled row mode.  A fresh-process
    certificate supplies its own three repeats; a same-process engineering
    check may request three or more.  Values 1 and 2 are never meaningful.
    """
    if isinstance(planning_repeats, bool) or not isinstance(planning_repeats, (int, np.integer)):
        raise ValueError("planning_repeats must be an integer")
    if int(planning_repeats) < 0 or (0 < int(planning_repeats) < 3):
        raise ValueError("planning_repeats must be zero or at least three")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, (int, np.integer)):
        raise ValueError("repeat_count must be an integer")
    if int(repeat_count) < 1:
        raise ValueError("repeat_count must be positive")


@torch.no_grad()
def capture_initial_only_cem(
    unroll: Callable[[Any, Any], Any],
    objective: Callable[[Any, Any], Any],
    z_init: Any,
    *,
    num_samples: int = 300,
    horizon: int = 6,
    action_dim: int = 10,
    var_scale: float = 1.0,
    seed: int = V12_LOCKED_SEED,
    batch_size: int = 8,
    check_batch_size: int | None = 4,
    equivalence_tolerance: float = float(V12_PROTOCOL["chunk_equivalence_tolerance"]),
) -> dict[str, Any]:
    """Sample and score exactly one (iteration-0) CEM population.

    The returned mapping contains no final population and no optimizer mean.
    Gaussian draws and the official mean-at-index-zero convention match the
    first iteration of the released CEM planner.
    """
    if num_samples != 300 or horizon != 6 or action_dim != 10:
        raise ValueError("v1.2 fixes a 300 x 6 x 10 iteration-0 population")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) not in V12_LOCKED_SEEDS:
        raise ValueError(f"v1.2 seed must be one of {V12_LOCKED_SEEDS}")
    if not math.isfinite(float(var_scale)) or float(var_scale) < 0:
        raise ValueError("var_scale must be finite and non-negative")
    if batch_size < 1 or (check_batch_size is not None and check_batch_size < 1):
        raise ValueError("score batch sizes must be positive")
    generator = torch.Generator(device=z_init.device).manual_seed(int(seed))
    mean = torch.zeros(horizon, action_dim, device=z_init.device)
    actions = mean[:, None, :] + float(var_scale) * torch.randn(
        horizon, num_samples, action_dim, device=z_init.device, generator=generator
    )
    actions[:, 0, :] = mean

    def score(population: Any, size: int) -> tuple[Any, Any]:
        scores, terminal_parts = [], {}
        for start in range(0, population.shape[1], size):
            batch = population[:, start : start + size]
            predicted = unroll(z_init, batch)
            value = torch.as_tensor(objective(predicted, batch)).reshape(-1)
            scores.append(value)
            for key, tensor in predicted.items():
                terminal_parts.setdefault(key, []).append(tensor[-1].detach().clone())
        result = torch.cat(scores, dim=0)
        terminal = {key: torch.cat(parts, dim=0) for key, parts in terminal_parts.items()}
        return result, terminal

    scores, terminal = score(actions, batch_size)
    if scores.numel() != num_samples:
        raise ValueError("objective must return one score per sampled proposal")
    if not torch.isfinite(scores).all():
        raise ValueError("v1.2 iteration-0 proposal scores are non-finite")
    trace: dict[str, Any] | None = None
    if check_batch_size is not None and check_batch_size != batch_size:
        checked, _ = score(actions, check_batch_size)
        if not torch.isfinite(checked).all():
            raise ValueError("v1.2 iteration-0 reference scores are non-finite")
        drift = float(torch.max(torch.abs(scores - checked)).item())
        trace = {
            "iteration": 0,
            "population_size": num_samples,
            "primary_batch_size": batch_size,
            "reference_batch_size": check_batch_size,
            "max_abs_score_difference": drift,
            # Chunking is a hardware deviation; lock the released winner
            # selector and bounded score drift, not incidental ordering among
            # non-winning candidates.
            "score_selector_agrees": bool(torch.argmin(scores) == torch.argmin(checked)),
            "tolerance": float(equivalence_tolerance),
        }
        if drift > float(equivalence_tolerance) or not trace["score_selector_agrees"]:
            raise RuntimeError(
                "v1.2 iteration-0 chunk equivalence failed: "
                f"max_abs_score_difference={drift}, "
                f"selector_agrees={trace['score_selector_agrees']}, "
                f"tolerance={float(equivalence_tolerance)}"
            )
    return {
        "initial_samples": actions.detach().clone(),
        "initial_scores": scores.detach().clone(),
        "initial_predicted": terminal,
        "iterations": 1,
        "num_samples": num_samples,
        "horizon": horizon,
        "action_dim": action_dim,
        "var_scale": float(var_scale),
        "seed": int(seed),
        "score_batch_size": int(batch_size),
        "chunk_trace": [] if trace is None else [trace],
        "inventory_kind": "common_initial",
        "outcome_blind": True,
    }


# Descriptive aliases make the mode obvious at call sites and preserve a
# straightforward migration path for code that calls capture_* by name.
capture_cem_initial_only = capture_initial_only_cem
capture_common_initial_inventory = capture_initial_only_cem


def expand_model_actions(actions: Any, *, frameskip: int = 5, physical_action_dim: int = 2) -> torch.Tensor:
    tensor = torch.as_tensor(actions)
    if tensor.ndim != 3 or tensor.shape[-1] != frameskip * physical_action_dim:
        raise ValueError("expected [candidates, steps, frameskip * physical_action_dim] actions")
    return tensor.reshape(tensor.shape[0], tensor.shape[1] * frameskip, physical_action_dim)


def _finite_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an array") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array.astype(np.float64, copy=False)).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_p_o_arrays(
    predicted_visual: Any,
    realized_visual: Any,
    goal_visual: Any,
    *,
    predicted_proprio: Any | None = None,
    realized_proprio: Any | None = None,
    goal_proprio: Any | None = None,
    alpha: float = 0.1,
    decomposition_tolerance: float = float(V12_PROTOCOL["decomposition_tolerance"]),
) -> dict[str, Any]:
    """Compute the unchanged P/O decomposition and fail closed on nonfinite data."""
    from latent_score_decomposition import decompose_latent_scores

    arrays = {
        "predicted_visual": predicted_visual, "realized_visual": realized_visual,
        "goal_visual": goal_visual,
    }
    if any(value is not None for value in (predicted_proprio, realized_proprio, goal_proprio)):
        arrays.update({
            "predicted_proprio": predicted_proprio, "realized_proprio": realized_proprio,
            "goal_proprio": goal_proprio,
        })
    for name, value in arrays.items():
        _finite_array(value, name)
    decomposition = decompose_latent_scores(
        predicted_visual, realized_visual, goal_visual,
        predicted_proprio=predicted_proprio, realized_proprio=realized_proprio,
        goal_proprio=goal_proprio, alpha=alpha,
    )
    p = _finite_array(decomposition["predicted_score"], "P")
    o = _finite_array(decomposition["observed_score"], "O")
    residual = _finite_array(decomposition["reconstruction_residual"], "decomposition residual")
    max_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    if max_residual > float(decomposition_tolerance):
        raise ValueError("latent decomposition identity exceeds v1.2 tolerance")
    return {
        "P": p.astype(np.float64), "O": o.astype(np.float64),
        "decomposition": decomposition,
        "decomposition_max_abs_error": max_residual,
    }


def _strict_choice(values: Any, *, lower_better: bool) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        return None
    best = np.min(values) if lower_better else np.max(values)
    ties = np.isclose(values, best, atol=float(V12_PROTOCOL["tie_atol"]), rtol=float(V12_PROTOCOL["tie_rtol"]))
    return int(np.flatnonzero(ties)[0]) if int(ties.sum()) == 1 else None


def p_o_g_j_summary(
    predicted_scores: Any, observed_scores: Any, native_returns: Any, native_state_distances: Any,
) -> dict[str, Any]:
    """Return finite-inventory P/O/G/J winners and preregistered deltas."""
    p = _finite_array(predicted_scores, "P", (300,)).astype(np.float64)
    o = _finite_array(observed_scores, "O", (300,)).astype(np.float64)
    j = _finite_array(native_returns, "J", (300,)).astype(np.float64)
    distance = _finite_array(native_state_distances, "G distance", (300,)).astype(np.float64)
    p_idx, o_idx = _strict_choice(p, lower_better=True), _strict_choice(o, lower_better=True)
    j_idx = int(np.argmax(j))
    g_idx = int(np.argmin(distance))
    best_j, best_g = float(np.max(j)), float(np.min(distance))
    return {
        "candidate_count": 300,
        "predicted_choice": p_idx,
        "encoded_real_choice": o_idx,
        "native_reward_choice": j_idx,
        "goal_aligned_choice": g_idx,
        "reward_comparison_abstained": p_idx is None or o_idx is None,
        "goal_comparison_abstained": p_idx is None or o_idx is None,
        "R_P_J": best_j - float(j[p_idx]) if p_idx is not None else None,
        "R_O_J": best_j - float(j[o_idx]) if o_idx is not None else None,
        "R_P_G": float(distance[p_idx] - best_g) if p_idx is not None else None,
        "R_O_G": float(distance[o_idx] - best_g) if o_idx is not None else None,
        "Delta_J": float(j[o_idx] - j[p_idx]) if p_idx is not None and o_idx is not None else None,
        "Delta_G": float(distance[p_idx] - distance[o_idx]) if p_idx is not None and o_idx is not None else None,
        "P_finite": True,
        "O_finite": True,
    }


def _support_distance(proposals: np.ndarray, released: np.ndarray) -> np.ndarray:
    if released.size == 0:
        return np.asarray([], dtype=np.float64)
    proposal = proposals.reshape(proposals.shape[0], -1).astype(np.float64)
    reference = released.reshape(released.shape[0], -1).astype(np.float64)
    # Chunking keeps this diagnostic bounded without touching model outcomes.
    distances = []
    for start in range(0, len(proposal), 64):
        distances.append(np.min(np.linalg.norm(proposal[start : start + 64, None] - reference[None], axis=-1), axis=1))
    return np.concatenate(distances) if distances else np.asarray([], dtype=np.float64)


def proposal_support_diagnostics(
    normalized_actions: Any,
    physical_actions: Any,
    *,
    released_train_actions: Any | None = None,
    released_val_actions: Any | None = None,
    unclipped_physical_actions: Any | None = None,
    action_low: Any | None = None,
    action_high: Any | None = None,
) -> dict[str, Any]:
    """Describe proposal magnitude, clipping, saturation, and released support.

    This function accepts no outcomes and therefore cannot tune proposal
    diagnostics using J/G.  Released action arrays are reference support only.
    """
    normalized = _finite_array(normalized_actions, "normalized proposals")
    physical = _finite_array(physical_actions, "physical proposals")
    if normalized.shape[0] != 300 or physical.shape[0] != 300:
        raise ValueError("v1.2 support diagnostics require exactly 300 proposals")
    unclipped = physical if unclipped_physical_actions is None else _finite_array(unclipped_physical_actions, "unclipped physical proposals", physical.shape)
    clipped_mask = np.any(np.not_equal(unclipped, physical), axis=tuple(range(1, physical.ndim)))
    saturation_mask = np.zeros(300, dtype=bool)
    bound_mask = np.zeros_like(physical, dtype=bool)
    if action_low is not None:
        bound_mask |= np.isclose(physical, np.asarray(action_low), atol=0.0, rtol=0.0)
    if action_high is not None:
        bound_mask |= np.isclose(physical, np.asarray(action_high), atol=0.0, rtol=0.0)
    if bound_mask.ndim > 1:
        saturation_mask = np.any(bound_mask, axis=tuple(range(1, bound_mask.ndim)))
    train = np.asarray([]) if released_train_actions is None else _finite_array(released_train_actions, "released train actions")
    val = np.asarray([]) if released_val_actions is None else _finite_array(released_val_actions, "released val actions")
    def flatten_reference(value: np.ndarray) -> np.ndarray:
        if not value.size:
            return np.empty((0, int(np.prod(normalized.shape[1:]))), dtype=np.float64)
        flat = value.reshape(value.shape[0], -1)
        if flat.shape[1] != int(np.prod(normalized.shape[1:])):
            # Released actions are often stored as physical [N, 30, 2]
            # while proposals are model-shaped [300, 6, 10].  Compare only
            # when their complete per-example vectors have the same width;
            # otherwise leave support undefined rather than inventing a
            # broadcast or using outcomes to choose a projection.
            return np.empty((0, int(np.prod(normalized.shape[1:]))), dtype=np.float64)
        return flat.astype(np.float64, copy=False)

    reference = np.concatenate([flatten_reference(train), flatten_reference(val)], axis=0)
    distances = _support_distance(normalized, reference)
    flat_normalized = normalized.reshape(300, -1).astype(np.float64)
    flat_physical = physical.reshape(300, -1).astype(np.float64)
    return {
        "outcome_blind": True,
        "proposal_count": 300,
        "normalized": {
            "mean_abs": float(np.mean(np.abs(flat_normalized))),
            "rms": float(np.sqrt(np.mean(np.square(flat_normalized)))),
            "max_abs": float(np.max(np.abs(flat_normalized))),
            "per_proposal_l2_mean": float(np.mean(np.linalg.norm(flat_normalized, axis=1))),
            "per_proposal_l2_q95": float(np.quantile(np.linalg.norm(flat_normalized, axis=1), 0.95)),
        },
        "physical": {
            "mean_abs": float(np.mean(np.abs(flat_physical))),
            "rms": float(np.sqrt(np.mean(np.square(flat_physical)))),
            "max_abs": float(np.max(np.abs(flat_physical))),
        },
        "environment_clipping": {
            "proposal_count_with_clipping": int(np.count_nonzero(clipped_mask)),
            "proposal_fraction_with_clipping": float(np.mean(clipped_mask)),
            "component_count_clipped": int(np.count_nonzero(unclipped != physical)),
            "interpretation": "released Push-T wrapper applies no action clipping; structural zero unless explicit unclipped actions are supplied",
        },
        "environment_saturation": {
            "proposal_count_at_bound": int(np.count_nonzero(saturation_mask)),
            "proposal_fraction_at_bound": float(np.mean(saturation_mask)),
            "component_count_at_bound": int(np.count_nonzero(bound_mask)),
        },
        "reference_action_chunks": {
            "train_action_count": int(train.reshape(-1, train.shape[-1]).shape[0]) if train.size else 0,
            "val_action_count": int(val.reshape(-1, val.shape[-1]).shape[0]) if val.size else 0,
            "reference_action_count": int(reference.shape[0]),
            "nearest_normalized_l2_mean": float(np.mean(distances)) if len(distances) else None,
            "nearest_normalized_l2_q95": float(np.quantile(distances, 0.95)) if len(distances) else None,
            "nearest_normalized_l2_max": float(np.max(distances)) if len(distances) else None,
            "support_defined": bool(len(distances)),
            "interpretation": "descriptive nearest frozen action-chunk reference; not a learned support set or claim of in-distribution behavior",
        },
        "tuning": "none; descriptive reference diagnostics only",
    }


# Names used by downstream callers.
compute_proposal_support_diagnostics = proposal_support_diagnostics
diagnose_proposal_support = proposal_support_diagnostics


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    x, y = _average_ranks(left), _average_ranks(right)
    x -= x.mean(); y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator else None


def standard_outer_metrics(
    predicted_scores: Any,
    realized_scores: Any,
    native_returns: Any,
    native_state_distances: Any,
    *,
    normalized_actions: Any | None = None,
    predicted_latents: Any | None = None,
    realized_latents: Any | None = None,
    predicted_proprio: Any | None = None,
    realized_proprio: Any | None = None,
    action_only_threshold: float = 3.0,
    _subset: bool = False,
) -> dict[str, Any]:
    """Compute standard ranking/prediction baselines for the 300 proposals.

    These metrics are descriptive baselines.  They do not replace the matched
    P→O intervention.  ``action_only_subset`` is the mandatory support
    robustness subset S={proposal: max(abs(normalized action)) <= 3}.
    """
    expected_shape = None if _subset else (300,)
    p = _finite_array(predicted_scores, "P", expected_shape).astype(np.float64)
    o = _finite_array(realized_scores, "O", expected_shape).astype(np.float64)
    j = _finite_array(native_returns, "J", expected_shape).astype(np.float64)
    distance = _finite_array(native_state_distances, "G distance", expected_shape).astype(np.float64)
    g = -distance

    def rank_accuracy(scores: np.ndarray, outcome: np.ndarray, *, outcome_higher: bool) -> float | None:
        if len(scores) < 2:
            return None
        total = correct = 0
        for i in range(len(scores) - 1):
            score_delta = scores[i + 1 :] - scores[i]
            outcome_delta = outcome[i + 1 :] - outcome[i]
            usable = (score_delta != 0) & (outcome_delta != 0)
            total += int(np.count_nonzero(usable))
            if outcome_higher:
                correct += int(np.count_nonzero((score_delta[usable] < 0) == (outcome_delta[usable] > 0)))
            else:
                correct += int(np.count_nonzero((score_delta[usable] < 0) == (outcome_delta[usable] < 0)))
        return float(correct / total) if total else None

    p_idx = _strict_choice(p, lower_better=True)
    o_idx = _strict_choice(o, lower_better=True)
    j_range = float(np.max(j) - np.min(j))
    g_range = float(np.max(g) - np.min(g))
    metrics = {
        "proposal_count": int(len(p)),
        "score_error_mse": float(np.mean(np.square(p - o))),
        "latent_prediction_mse": None,
        "absolute_P_minus_O": {
            "mean": float(np.mean(np.abs(p - o))),
            "median": float(np.median(np.abs(p - o))),
            "max": float(np.max(np.abs(p - o))),
        },
        "spearman": {
            "P_vs_J": _spearman(p, -j),
            "P_vs_G": _spearman(p, -g),
            "O_vs_J": _spearman(o, -j),
            "O_vs_G": _spearman(o, -g),
        },
        "pairwise_ranking_accuracy": {
            "P_vs_J": rank_accuracy(p, j, outcome_higher=True),
            "P_vs_G": rank_accuracy(p, g, outcome_higher=True),
            "O_vs_J": rank_accuracy(o, j, outcome_higher=True),
            "O_vs_G": rank_accuracy(o, g, outcome_higher=True),
        },
        "normalized_regret_at_1": {
            "P_J": (float(np.max(j) - j[p_idx]) / j_range) if p_idx is not None and j_range > 0 else None,
            "P_G": (float(np.max(g) - g[p_idx]) / g_range) if p_idx is not None and g_range > 0 else None,
            "O_J": (float(np.max(j) - j[o_idx]) / j_range) if o_idx is not None and j_range > 0 else None,
            "O_G": (float(np.max(g) - g[o_idx]) / g_range) if o_idx is not None and g_range > 0 else None,
            "normalization": "current analyzed inventory range",
        },
        "matched_recovery": {
            "predicted_choice_index": p_idx,
            "realized_latent_choice_index": o_idx,
            "R_P_J": float(np.max(j) - j[p_idx]) if p_idx is not None else None,
            "R_O_J": float(np.max(j) - j[o_idx]) if o_idx is not None else None,
            "R_P_G": float(np.max(g) - g[p_idx]) if p_idx is not None else None,
            "R_O_G": float(np.max(g) - g[o_idx]) if o_idx is not None else None,
            "Delta_J": float(j[o_idx] - j[p_idx]) if p_idx is not None and o_idx is not None else None,
            "Delta_G": float(g[o_idx] - g[p_idx]) if p_idx is not None and o_idx is not None else None,
            "abstained": p_idx is None or o_idx is None,
        },
        "top_k_normalized_regret": {},
        "uniform_expectation": {"J": float(np.mean(j)), "G": float(np.mean(g))},
        "action_only_subset": None,
    }
    for k in (1, 5, 10):
        effective_k = min(k, len(p))
        selected = np.argsort(p, kind="mergesort")[:effective_k]
        metrics["top_k_normalized_regret"][str(k)] = {
            "J": float((np.max(j) - np.max(j[selected])) / j_range) if j_range > 0 else None,
            "G": float((np.max(g) - np.max(g[selected])) / g_range) if g_range > 0 else None,
        }
    if predicted_latents is not None or realized_latents is not None:
        if predicted_latents is None or realized_latents is None:
            raise ValueError("predicted_latents and realized_latents must be supplied together")
        predicted_latents_array = _finite_array(predicted_latents, "predicted latents")
        realized_latents_array = _finite_array(realized_latents, "realized latents")
        if predicted_latents_array.shape != realized_latents_array.shape:
            raise ValueError("predicted_latents and realized_latents must have matching shapes")
        visual_prediction_mse = float(np.mean(np.square(
            predicted_latents_array.astype(np.float64) - realized_latents_array.astype(np.float64)
        )))
        proprio_prediction_mse = 0.0
        if predicted_proprio is not None or realized_proprio is not None:
            if predicted_proprio is None or realized_proprio is None:
                raise ValueError("predicted_proprio and realized_proprio must be supplied together")
            predicted_proprio_array = _finite_array(predicted_proprio, "predicted proprio")
            realized_proprio_array = _finite_array(realized_proprio, "realized proprio")
            if predicted_proprio_array.shape != realized_proprio_array.shape:
                raise ValueError("predicted_proprio and realized_proprio must have matching shapes")
            proprio_prediction_mse = float(np.mean(np.square(
                predicted_proprio_array.astype(np.float64) - realized_proprio_array.astype(np.float64)
            )))
        metrics["latent_prediction_mse"] = {
            "visual": visual_prediction_mse,
            "proprio": proprio_prediction_mse,
            "weighted_total": visual_prediction_mse + float(V12_PROTOCOL["alpha"]) * proprio_prediction_mse,
            "alpha": float(V12_PROTOCOL["alpha"]),
        }
    if normalized_actions is not None:
        normalized = _finite_array(normalized_actions, "normalized proposals")
        if normalized.shape[0] != 300:
            raise ValueError("normalized_actions must contain 300 proposals")
        support = np.max(np.abs(normalized.reshape(300, -1)), axis=1) <= float(action_only_threshold)
        if int(np.count_nonzero(support)) >= 2:
            subset = standard_outer_metrics(
                p[support], o[support], j[support], distance[support],
                action_only_threshold=action_only_threshold,
                _subset=True,
            )
            subset["subset_count"] = int(np.count_nonzero(support))
            subset["definition"] = "S = max(abs(normalized action)) <= 3"
            subset["support_mask_sha256"] = array_sha256(support.astype(np.uint8))
            metrics["action_only_subset"] = subset
        else:
            metrics["action_only_subset"] = {
                "subset_count": int(np.count_nonzero(support)),
                "definition": "S = max(abs(normalized action)) <= 3",
                "status": "insufficient_support",
            }
    return metrics


compute_standard_outer_metrics = standard_outer_metrics


def build_v12_artifact(
    *,
    model: str,
    decision_id: str,
    episode: int,
    offset: int,
    seed: int = V12_LOCKED_SEED,
    normalized_actions: Any,
    physical_actions: Any,
    predicted_visual: Any,
    realized_visual: Any,
    goal_visual: Any,
    predicted_proprio: Any,
    realized_proprio: Any,
    goal_proprio: Any,
    objective_scores: Any,
    native_returns: Any,
    native_state_distances: Any,
    final_states: Any,
    goal_state: Any,
    reward_traces: Any,
    expert_reward_trace: Any,
    noop_physical_actions: Any,
    noop_final_state: Any,
    noop_reward_trace: Any,
    raw_visual_observations: Any,
    raw_proprio_observations: Any,
    raw_initial_visual: Any,
    raw_initial_proprio: Any,
    raw_goal_visual: Any,
    raw_goal_proprio: Any,
    raw_reencoding_gate: Mapping[str, Any],
    raw_visual_digests: Any,
    raw_proprio_digests: Any,
    trajectory_id: str | None = None,
    arrays_out: Path | None = None,
    proposal_diagnostics: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None,
    planning_gate: Mapping[str, Any] | None = None,
    certificate: Mapping[str, Any] | None = None,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one v1.2 artifact from the shared 300-proposal evidence.

    The function writes only v1.2 ``initial_*`` arrays.  In particular, it
    cannot accidentally serialize a v1.1 final inventory or returned mean.
    """
    normalized = _finite_array(normalized_actions, "normalized actions", (300, 6, 10)).astype(np.float32)
    physical = _finite_array(physical_actions, "physical actions", (300, 30, 2)).astype(np.float32)
    states = _finite_array(final_states, "terminal states", (300, np.asarray(final_states).shape[-1])).astype(np.float64)
    goal = _finite_array(goal_state, "goal state", (states.shape[1],)).astype(np.float64)
    rewards = _finite_array(reward_traces, "reward traces", (300, 30)).astype(np.float32)
    expert_rewards = _finite_array(expert_reward_trace, "expert reward trace", (30,)).astype(np.float32)
    noop_physical = _finite_array(noop_physical_actions, "hold-position noop actions", (30, 2)).astype(np.float32)
    noop_state = _finite_array(noop_final_state, "hold-position noop final state").astype(np.float64)
    if noop_state.shape != goal.shape:
        raise ValueError("hold-position noop final state must match goal state shape")
    noop_rewards = _finite_array(noop_reward_trace, "hold-position noop reward trace", (30,)).astype(np.float32)
    raw_visual = np.asarray(raw_visual_observations)
    raw_proprio = _finite_array(raw_proprio_observations, "raw terminal proprio").astype(np.float32)
    initial_raw_visual = np.asarray(raw_initial_visual)
    initial_raw_proprio = _finite_array(raw_initial_proprio, "raw initial proprio").astype(np.float32)
    goal_raw_visual = np.asarray(raw_goal_visual)
    goal_raw_proprio = _finite_array(raw_goal_proprio, "raw goal proprio").astype(np.float32)
    if raw_visual.dtype != np.uint8 or raw_visual.shape != (300, 1, 3, 224, 224):
        raise ValueError("raw terminal visual observations must have uint8 shape (300, 1, 3, 224, 224)")
    if raw_proprio.shape != (300, 1, 4):
        raise ValueError("raw terminal proprio observations must have shape (300, 1, 4)")
    if initial_raw_visual.dtype != np.uint8 or initial_raw_visual.shape != (1, 3, 224, 224):
        raise ValueError("raw initial visual must have uint8 shape (1, 3, 224, 224)")
    if initial_raw_proprio.shape != (1, 4):
        raise ValueError("raw initial proprio must have shape (1, 4)")
    if goal_raw_visual.dtype != np.uint8 or goal_raw_visual.shape != (1, 3, 224, 224):
        raise ValueError("raw goal visual must have uint8 shape (1, 3, 224, 224)")
    if goal_raw_proprio.shape != (1, 4):
        raise ValueError("raw goal proprio must have shape (1, 4)")
    if raw_reencoding_gate.get("passed") is not True:
        raise ValueError("lossless raw evidence re-encoding gate did not pass")
    returns = _finite_array(native_returns, "J", (300,)).astype(np.float64)
    distances = _finite_array(native_state_distances, "G distance", (300,)).astype(np.float64)
    recomputed_returns = np.asarray([
        float(np.add.accumulate(trace, dtype=np.float32)[-1]) for trace in rewards
    ], dtype=np.float64)
    recomputed_distances = np.linalg.norm(states - goal[None, :], axis=1)
    if not np.array_equal(returns, recomputed_returns):
        raise ValueError("J does not exactly equal the float32 cumulative 30-step reward trace")
    if not np.allclose(distances, recomputed_distances, atol=1e-12, rtol=0):
        raise ValueError("G distance does not reproduce from goal and terminal states")
    scores = _finite_array(objective_scores, "captured P", (300,)).astype(np.float64)
    visual_digests = _finite_array(raw_visual_digests, "terminal visual digests").astype(np.uint8)
    proprio_digests = _finite_array(raw_proprio_digests, "terminal proprio digests").astype(np.uint8)
    if visual_digests.shape[0] != 300 or proprio_digests.shape[0] != 300:
        raise ValueError("terminal digest arrays must contain 300 proposals")
    for index in range(300):
        expected_visual = np.frombuffer(bytes.fromhex(array_sha256(raw_visual[index])), dtype=np.uint8)
        expected_proprio = np.frombuffer(bytes.fromhex(array_sha256(raw_proprio[index])), dtype=np.uint8)
        if not np.array_equal(visual_digests[index], expected_visual):
            raise ValueError(f"raw terminal visual digest mismatch at proposal {index}")
        if not np.array_equal(proprio_digests[index], expected_proprio):
            raise ValueError(f"raw terminal proprio digest mismatch at proposal {index}")
    decomposition = validate_p_o_arrays(
        predicted_visual, realized_visual, goal_visual,
        predicted_proprio=predicted_proprio, realized_proprio=realized_proprio,
        goal_proprio=goal_proprio, alpha=float(V12_PROTOCOL["alpha"]),
    )
    p, o = decomposition["P"], decomposition["O"]
    if not np.allclose(p, scores, atol=float(V12_PROTOCOL["score_reproduction_tolerance"]), rtol=0):
        raise ValueError("v1.2 captured iteration-0 scores do not reproduce P")
    summary = p_o_g_j_summary(p, o, returns, distances)
    hashes = [array_sha256(normalized[index]) for index in range(300)]
    multiplicity = {value: hashes.count(value) for value in set(hashes)}
    ids = [f"proposal_{index:04d}" for index in range(300)]
    candidate_rows = [
        {
            "candidate_id": ids[index], "action_model": normalized[index].tolist(),
            "predicted_score": float(p[index]), "source_planner_score": float(scores[index]),
            "encoded_real_score": float(o[index]), "native_return": float(returns[index]),
            "native_state_distance": float(distances[index]), "G": float(-distances[index]),
            "J": float(returns[index]), "final_state": states[index].tolist(),
            "normalized_action_sha256": hashes[index],
            "normalized_plan_multiplicity": multiplicity[hashes[index]],
            "physical_action_sha256": array_sha256(physical[index]),
            "terminal_observation_sha256": canonical_sha256({
                "visual": array_sha256(visual_digests[index]),
                "proprio": array_sha256(proprio_digests[index]),
            }),
        }
        for index in range(300)
    ]
    def candidate_id(index: int | None) -> str | None:
        return ids[index] if index is not None else None
    decision = {
        **summary,
        "decision_id": decision_id,
        "predicted_choice": candidate_id(summary["predicted_choice"]),
        "encoded_real_choice": candidate_id(summary["encoded_real_choice"]),
        "native_reward_choice": candidate_id(summary["native_reward_choice"]),
        "goal_aligned_choice": candidate_id(summary["goal_aligned_choice"]),
        "source_planner_choice": candidate_id(summary["predicted_choice"]),
        "source_recomputed_winner_agrees": True,
        "inventory": "common_initial",
    }
    common_inventory = dict(decision)
    common_inventory.update({
        "outcome_blind": True, "count": 300, "unique_plan_count": len(set(hashes)),
        "normalized_sha256": array_sha256(normalized), "physical_sha256": array_sha256(physical),
        "objective_scores_sha256": array_sha256(scores), "initial_j_sha256": array_sha256(returns),
        "initial_g_sha256": array_sha256(-distances),
        "initial_terminal_observation_sha256": canonical_sha256({
            "visual": array_sha256(realized_visual), "proprio": array_sha256(realized_proprio),
        }),
        "initial_terminal_state_sha256": array_sha256(states),
        "captured_score_source": "cem_iteration_0_objective_scores",
        "decomposition_max_abs_error": decomposition["decomposition_max_abs_error"],
    })
    outer_metrics = standard_outer_metrics(
        p, o, returns, distances, normalized_actions=normalized,
        predicted_latents=predicted_visual, realized_latents=realized_visual,
        predicted_proprio=predicted_proprio, realized_proprio=realized_proprio,
    )
    def consequential_pair(other: int | None) -> dict[str, Any] | None:
        selected = summary["predicted_choice"]
        if selected is None or other is None or selected == other:
            return None
        directional = np.asarray(decomposition["decomposition"]["directional_correction"], dtype=np.float64)
        magnitude = np.asarray(decomposition["decomposition"]["magnitude_correction"], dtype=np.float64)
        return {
            "predicted_choice_index": int(selected), "counterfactual_choice_index": int(other),
            "P_margin_selected_minus_counterfactual": float(p[selected] - p[other]),
            "O_margin_selected_minus_counterfactual": float(o[selected] - o[other]),
            "directional_margin": float(directional[selected] - directional[other]),
            "magnitude_margin": float(magnitude[selected] - magnitude[other]),
            "identity_residual": float((p[selected] - p[other]) - (o[selected] - o[other]) - (directional[selected] - directional[other]) - (magnitude[selected] - magnitude[other])),
        }
    consequential_pairs = {
        "P_vs_O_choice": consequential_pair(summary["encoded_real_choice"]),
        "P_vs_finite_menu_best_G": consequential_pair(summary["goal_aligned_choice"]),
        "P_vs_finite_menu_best_J": consequential_pair(summary["native_reward_choice"]),
    }
    if arrays_out is not None:
        arrays_out = Path(arrays_out).resolve()
        arrays_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_out,
            initial_actions_normalized=normalized,
            initial_physical_actions=physical,
            initial_predicted_visual=np.asarray(predicted_visual, dtype=np.float32),
            initial_encoded_real_visual=np.asarray(realized_visual, dtype=np.float32),
            goal_visual=np.asarray(goal_visual, dtype=np.float32),
            initial_predicted_proprio=np.asarray(predicted_proprio, dtype=np.float32),
            initial_encoded_real_proprio=np.asarray(realized_proprio, dtype=np.float32),
            goal_proprio=np.asarray(goal_proprio, dtype=np.float32),
            initial_objective_scores=scores,
            initial_native_returns=returns,
            initial_native_state_distances=distances,
            initial_final_states=states,
            goal_state=goal,
            initial_reward_traces=rewards,
            expert_reward_trace=expert_rewards,
            hold_position_noop_physical_actions=noop_physical,
            hold_position_noop_final_state=noop_state,
            hold_position_noop_reward_trace=noop_rewards,
            initial_raw_visual_observations=raw_visual,
            initial_raw_proprio_observations=raw_proprio,
            raw_initial_visual=initial_raw_visual,
            raw_initial_proprio=initial_raw_proprio,
            raw_goal_visual=goal_raw_visual,
            raw_goal_proprio=goal_raw_proprio,
            initial_raw_visual_digests=visual_digests,
            initial_raw_proprio_digests=proprio_digests,
        )
    random_index = int(hashlib.sha256(f"v1.2:{model}:{episode}:{offset}".encode("utf-8")).hexdigest(), 16) % 300
    expert_return = float(np.add.accumulate(expert_rewards, dtype=np.float32)[-1])
    noop_return = float(np.add.accumulate(noop_rewards, dtype=np.float32)[-1])
    noop_distance = float(np.linalg.norm(noop_state - goal))
    baseline_values = {
        "random": {
            "candidate_id": ids[random_index], "candidate_index": random_index,
            "J": float(returns[random_index]), "G": float(-distances[random_index]),
            "selection": "frozen SHA-256 descriptive candidate; not a primary baseline",
        },
        "P": {
            "candidate_id": candidate_id(summary["predicted_choice"]),
            "candidate_index": summary["predicted_choice"],
            "J": float(returns[summary["predicted_choice"]]) if summary["predicted_choice"] is not None else None,
            "G": float(-distances[summary["predicted_choice"]]) if summary["predicted_choice"] is not None else None,
        },
        "O": {
            "candidate_id": candidate_id(summary["encoded_real_choice"]),
            "candidate_index": summary["encoded_real_choice"],
            "J": float(returns[summary["encoded_real_choice"]]) if summary["encoded_real_choice"] is not None else None,
            "G": float(-distances[summary["encoded_real_choice"]]) if summary["encoded_real_choice"] is not None else None,
        },
        "G_oracle": {"candidate_id": candidate_id(summary["goal_aligned_choice"]), "candidate_index": summary["goal_aligned_choice"], "G": float(-np.min(distances))},
        "J_oracle": {"candidate_id": candidate_id(summary["native_reward_choice"]), "candidate_index": summary["native_reward_choice"], "J": float(np.max(returns))},
        "uniform_expectation": {"J": float(np.mean(returns)), "G": float(np.mean(-distances)), "candidate_count": 300},
        "expert_goal_defining_segment": {
            "J": expert_return, "G": -0.0,
            "native_return": expert_return, "native_state_distance": 0.0,
        },
        "hold_position_noop": {
            "J": noop_return, "G": -noop_distance,
            "native_return": noop_return, "native_state_distance": noop_distance,
            "physical_action_sha256": array_sha256(noop_physical),
        },
        "repeatability": {
            "passed": False, "repeat_count": 1,
            "state_sha256": [array_sha256(states[0])],
            "terminal_observation_sha256": [array_sha256(raw_visual[0])],
            "cumulative_reward": [float(recomputed_returns[0])],
        },
    }
    baseline_values.update(dict(baselines or {}))
    artifact: dict[str, Any] = {
        "schema_version": V12_ARTIFACT_SCHEMA_VERSION,
        "scientific_result": False,
        "inventory": "common_initial", "outcome_blind": True,
        "model": model, "decision_id": decision_id,
        "trajectory_id": trajectory_id or f"val_episode_{episode:03d}",
        "episode": int(episode), "offset": int(offset), "seed": int(seed),
        "protocol": V12_PROTOCOL, "protocol_sha256": v12_protocol_sha256(),
        "source_bundle_sha256": v12_source_bundle_sha256(),
        "frameskip": 5, "horizon": 6, "cem_iterations": 1,
        "cem_num_samples": 300, "cem_num_elites": 10,
        "model_action_dim": 10, "physical_action_dim": 2, "physical_replay_steps": 30,
        "candidate_rows": candidate_rows, "decision": decision,
        "common_inventory": common_inventory,
        "initial_inventory": {
            "inventory": "common_initial", "outcome_blind": True, "count": 300,
            "normalized_sha256": array_sha256(normalized), "physical_sha256": array_sha256(physical),
            "objective_scores_sha256": array_sha256(scores), "score_capture": "cem_iteration_0",
        },
        "baselines": baseline_values,
        "standard_outer_metrics": outer_metrics,
        "raw_evidence_reencoding_gate": dict(raw_reencoding_gate),
        "consequential_pair_decomposition": consequential_pairs,
        "proposal_support_diagnostics": dict(
            proposal_support_diagnostics(normalized, physical)
            if proposal_diagnostics is None else proposal_diagnostics
        ),
        "planning_reproducibility_gate": dict(planning_gate or {"passed": False, "repeat_count": 0}),
        "reproducibility_certificate": dict(certificate or {}),
        "P_finite": bool(np.isfinite(p).all()), "O_finite": bool(np.isfinite(o).all()),
        "score_reproduction_max_abs_error": float(np.max(np.abs(p - scores))),
        "score_reproduction_tolerance": float(V12_PROTOCOL["score_reproduction_tolerance"]),
        "decomposition_max_abs_error": decomposition["decomposition_max_abs_error"],
        "terminal_observation_sha256": canonical_sha256({
            "visual": array_sha256(realized_visual), "proprio": array_sha256(realized_proprio),
        }),
        "terminal_state_sha256": array_sha256(states),
        "array_hashes": {},
    }
    if bindings:
        artifact.update(dict(bindings))
    if arrays_out is not None:
        with np.load(arrays_out, allow_pickle=False) as archive:
            artifact["array_hashes"] = {key: array_sha256(archive[key]) for key in sorted(archive.files)}
        artifact["arrays_sidecar"] = str(arrays_out)
        artifact["arrays_sidecar_sha256"] = file_sha256(arrays_out)
    return artifact


def _dataset_lengths(dataset_root: Path) -> list[int]:
    with (dataset_root / "val" / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(value) for value in pickle.load(handle)]
    if len(lengths) != 21:
        raise ValueError("v1.2 requires 21 validation trajectories")
    return lengths


def _expert_phase(dataset_root: Path, episode: int, offset: int, length: int) -> tuple[str, int]:
    """Predeclare a contact-phase proxy from expert block motion, never model outcomes."""
    import torch
    states = torch.load(dataset_root / "val" / "states.pth", map_location="cpu", weights_only=False)
    block = np.asarray(states[episode, :length, 2:5], dtype=np.float64)
    moved = np.linalg.norm(block - block[0], axis=1) > 1e-6
    first_motion = int(np.flatnonzero(moved)[0]) if np.any(moved) else int(length)
    if offset + 30 < first_motion:
        phase = "before_expert_block_motion"
    elif offset >= first_motion:
        phase = "after_expert_block_motion_begins"
    else:
        phase = "crosses_expert_first_block_motion"
    return phase, first_motion


def make_v12_manifest(dataset_root: Path, *, seeds: tuple[int, ...] = V12_LOCKED_SEEDS) -> dict[str, Any]:
    """Build the locked 20 x 5 x 2 x 3 proposal-menu schedule."""
    if tuple(seeds) != V12_LOCKED_SEEDS:
        raise ValueError(f"v1.2 fixes menu seeds={V12_LOCKED_SEEDS}")
    lengths = _dataset_lengths(Path(dataset_root))
    rows = []
    for episode, length in enumerate(lengths):
        if episode == V12_EXCLUDED_TRAJECTORY:
            continue
        max_offset = length - 31
        if max_offset < 0:
            raise ValueError(f"validation trajectory {episode} is shorter than 31 frames")
        for k in range(5):
            offset = int((k * max_offset) // 4)
            phase, first_motion = _expert_phase(Path(dataset_root), episode, offset, length)
            for menu_seed in seeds:
                for model in V12_LOCKED_MODELS:
                    rows.append({
                        "decision_id": f"v12_{model}_pusht_val_episode_{episode:03d}_offset_{offset:03d}_seed_{menu_seed:02d}",
                        "trajectory_id": f"val_episode_{episode:03d}", "model": model,
                        "episode": episode, "offset": offset, "seed": menu_seed,
                        "expert_phase_proxy": phase,
                        "expert_first_block_motion_index": first_motion,
                        "inventory": "common_initial", "outcome_blind": True,
                    })
    return {
        "schema_version": V12_SCHEMA_VERSION,
        "scientific_result": False,
        "selection": {
            "inventory": "common_initial_only", "outcome_blind": True,
            "menu_seeds": list(seeds), "frameskip": 5, "horizon": 6, "segment_length": 31,
            "offset_rule": "floor(k * (L - 31) / 4), k=0,...,4",
            "offset_count_per_trajectory": 5, "excluded_trajectories": [7],
            "models": list(V12_LOCKED_MODELS), "dataset_split": "val",
            "phase_rule": "block state dimensions 2:5 first differ from frame 0 by L2 > 1e-6; label whether the fixed 31-frame segment is before, crosses, or starts after that index",
            "selection_source": "offsets use val/seq_lengths.pkl only; expert states label a fixed secondary phase proxy but never select or exclude rows",
        },
        "dataset_root": str(Path(dataset_root).resolve()), "dataset_split": "val",
        "protocol": V12_PROTOCOL, "protocol_sha256": v12_protocol_sha256(),
        "source_bundle": v12_source_bundle(), "source_bundle_sha256": v12_source_bundle_sha256(),
        "trajectory_count": 20, "segment_count": 300, "decision_count": len(rows), "entries": rows,
    }


def validate_v12_manifest(manifest: Mapping[str, Any], *, require_full: bool = True) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != V12_SCHEMA_VERSION:
        errors.append("manifest is not PWA-PushT-v1.2")
    if manifest.get("scientific_result") is not False:
        errors.append("v1.2 manifest must remain engineering-only")
    if manifest.get("protocol") != V12_PROTOCOL or manifest.get("protocol_sha256") != v12_protocol_sha256():
        errors.append("manifest protocol/hash is not the locked v1.2 protocol")
    if manifest.get("source_bundle_sha256") != v12_source_bundle_sha256():
        errors.append("manifest source bundle hash does not match v1.2 sources")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("inventory") != "common_initial_only":
        errors.append("manifest inventory is not common_initial_only")
    if isinstance(selection, Mapping):
        for key, expected in {
            "outcome_blind": True, "menu_seeds": list(V12_LOCKED_SEEDS), "frameskip": 5, "horizon": 6,
            "segment_length": 31, "offset_count_per_trajectory": 5,
            "excluded_trajectories": [7], "models": list(V12_LOCKED_MODELS),
            "dataset_split": "val",
        }.items():
            if selection.get(key) != expected:
                errors.append(f"manifest selection {key} is not locked for v1.2")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return errors + ["v1.2 manifest entries are not a list"]
    ids = [item.get("decision_id") for item in entries if isinstance(item, Mapping)]
    if len(ids) != len(set(ids)):
        errors.append("v1.2 manifest contains duplicate decision_id")
    keys, episodes, trajectories = [], [], []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"entry {index} is not an object")
            continue
        required = {"decision_id", "trajectory_id", "model", "episode", "offset", "seed", "expert_phase_proxy", "expert_first_block_motion_index", "inventory", "outcome_blind"}
        if not required.issubset(entry):
            errors.append(f"entry {index} is missing v1.2 fields")
            continue
        if entry.get("inventory") != "common_initial" or entry.get("outcome_blind") is not True:
            errors.append(f"entry {index} is not a common-initial outcome-blind row")
        if entry.get("episode") == 7:
            errors.append("v1.2 manifest includes excluded trajectory 7")
        if entry.get("model") not in V12_LOCKED_MODELS:
            errors.append(f"entry {index} has unsupported model")
        if entry.get("seed") not in V12_LOCKED_SEEDS:
            errors.append(f"entry {index} does not use a locked menu seed")
        keys.append((entry.get("trajectory_id"), entry.get("offset"), entry.get("seed"), entry.get("model")))
        episodes.append(entry.get("episode")); trajectories.append(entry.get("trajectory_id"))
    if len(keys) != len(set(keys)):
        errors.append("v1.2 manifest contains duplicate segment/model rows")
    if manifest.get("trajectory_count") != len(set(trajectories)):
        errors.append("v1.2 trajectory_count mismatch")
    if manifest.get("segment_count") != len({(key[0], key[1], key[2]) for key in keys}):
        errors.append("v1.2 segment_count mismatch")
    if manifest.get("decision_count") != len(entries):
        errors.append("v1.2 decision_count mismatch")
    if require_full:
        if len(entries) != 600 or len(set(trajectories)) != 20:
            errors.append("v1.2 requires exactly 600 decisions across 20 trajectories")
        for trajectory in set(trajectories):
            group = [item for item in entries if item.get("trajectory_id") == trajectory]
            if len(group) != 30 or {item.get("model") for item in group} != set(V12_LOCKED_MODELS) or {item.get("seed") for item in group} != set(V12_LOCKED_SEEDS):
                errors.append(f"v1.2 trajectory {trajectory} lacks five offsets, three menus, and both models")
        dataset_root_value = manifest.get("dataset_root")
        if not isinstance(dataset_root_value, str):
            errors.append("v1.2 manifest lacks a dataset root for canonical schedule regeneration")
        else:
            try:
                expected = make_v12_manifest(Path(dataset_root_value))
                if entries != expected["entries"]:
                    errors.append("v1.2 entries do not equal the schedule regenerated from seq_lengths")
                if selection != expected["selection"]:
                    errors.append("v1.2 selection does not equal the canonical regenerated selection")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"v1.2 canonical schedule cannot be regenerated: {exc}")
    return sorted(set(errors))


def validate_v12_artifact_identity(artifact: Mapping[str, Any], *, expected_model: str | None = None) -> list[str]:
    """Reject v1.1/final artifacts and enforce v1.2 identity markers."""
    errors: list[str] = []
    if artifact.get("schema_version") != V12_ARTIFACT_SCHEMA_VERSION:
        errors.append("artifact is not a v1.2 common-initial artifact")
    if artifact.get("scientific_result") is not False:
        errors.append("v1.2 artifact must remain engineering-only")
    if artifact.get("inventory") != "common_initial" or artifact.get("outcome_blind") is not True:
        errors.append("artifact is not marked common_initial/outcome_blind")
    if artifact.get("cem_iterations") != 1 or artifact.get("cem_num_samples") != 300:
        errors.append("v1.2 artifact must contain exactly one 300-plan iteration-0 population")
    for forbidden in ("final_inventory", "natural_final", "final_actions_normalized", "returned_mean_actions_normalized"):
        if forbidden in artifact:
            errors.append(f"v1.2 artifact cannot contain forbidden final-inventory field {forbidden}")
    if expected_model is not None and artifact.get("model") != expected_model:
        errors.append("v1.2 artifact model does not match expected model")
    return sorted(set(errors))


def cross_model_common_initial_errors(
    jepa: Mapping[str, Any], dino: Mapping[str, Any], *, array_keys: tuple[str, ...] | None = None,
) -> list[str]:
    """Require byte-exact physical candidate and terminal evidence across models."""
    keys = array_keys or (
        "normalized_actions", "physical_actions", "terminal_observation", "terminal_state",
        "J", "G", "native_returns", "native_state_distances",
    )
    errors = []
    for key in keys:
        if key not in jepa or key not in dino:
            errors.append(f"cross-model common-initial evidence missing {key}")
            continue
        try:
            if not np.array_equal(np.asarray(jepa[key]), np.asarray(dino[key])):
                errors.append(f"cross-model common-initial {key} mismatch")
        except (TypeError, ValueError):
            if jepa[key] != dino[key]:
                errors.append(f"cross-model common-initial {key} mismatch")
    return sorted(set(errors))


def validate_cross_model_common_initial_gate(gate: Mapping[str, Any]) -> list[str]:
    errors = []
    if gate.get("passed") is not True:
        errors.append("cross-model common-initial gate did not pass")
    if gate.get("inventory") != "common_initial_only":
        errors.append("cross-model gate is not bound to common_initial_only")
    if gate.get("models") != list(V12_LOCKED_MODELS):
        errors.append("cross-model gate does not bind both released models")
    return errors


__all__ = [
    "V12_SCHEMA_VERSION", "V12_ARTIFACT_SCHEMA_VERSION", "V12_PROTOCOL",
    "V12_LOCKED_MODELS", "V12_LOCKED_SEED", "V12_LOCKED_SEEDS", "V12_SOURCE_BUNDLE_VERSION", "canonical_sha256",
    "array_sha256", "v12_protocol_sha256", "v12_source_bundle",
    "v12_source_bundle_sha256", "validate_v12_args", "capture_initial_only_cem",
    "capture_cem_initial_only", "capture_common_initial_inventory", "expand_model_actions",
    "validate_p_o_arrays", "p_o_g_j_summary", "proposal_support_diagnostics",
    "compute_proposal_support_diagnostics", "diagnose_proposal_support", "make_v12_manifest",
    "validate_v12_manifest", "validate_v12_artifact_identity", "cross_model_common_initial_errors",
    "validate_cross_model_common_initial_gate", "standard_outer_metrics",
    "compute_standard_outer_metrics",
    "build_v12_artifact",
]
