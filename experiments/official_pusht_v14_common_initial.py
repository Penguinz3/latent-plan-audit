"""PWA-PushT v1.4 common-initial-only protocol.

v1.4 is a clean, versioned successor to the stopped v1.2 execution.  It
keeps the same outcome-blind iteration-zero menu and the same P/O/G/J
decomposition.  The production encoding is authoritative; re-encoding the
lossless observations is a diagnostic.  Coordinate drift is recorded but has
no latent-coordinate cutoff.  Decision-level score drift and selector gates
remain locked.

The released model/replay implementation is imported only as a code
dependency.  No v1.2 artifact, certificate, manifest, or issuance record is
read or authorized by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from experiments import official_pusht_v12_common_initial as _v12
from latent_score_decomposition import decompose_latent_scores
from experiments.official_pusht_v14_bootstrap import V14_CANONICAL_PROJECT_ROOT


V14_SCHEMA_VERSION = "PWA-PushT-v1.4"
V14_ARTIFACT_SCHEMA_VERSION = "official-pusht-decision-audit-v1.4-common-initial"
V14_ANALYSIS_VERSION = "PWA-PushT-analysis-v1.4-common-initial"
V14_CERTIFICATE_SCHEMA_VERSION = "PWA-PushT-reproducibility-certificate-v1.4-common-initial"
V14_BUNDLE_SCHEMA_VERSION = "PWA-PushT-certificate-bundle-v1.4-common-initial"
V14_SUPERVISOR_VERSION = "PWA-PushT-fresh-process-supervisor-v1.4-common-initial"
V14_SOURCE_BUNDLE_VERSION = "PWA-PushT-source-bundle-v1.4-common-initial"
V14_CERTIFICATE_ISSUANCE_SCHEMA_VERSION = "PWA-PushT-certificate-issuance-v1.4-common-initial"
V14_CERTIFICATE_ISSUANCE_RELATIVE_PATH = "experiments/official_pusht_v14_certificate_issuance.json"
V14_LOCKED_MODELS = ("official_jepa_wm", "official_dino_wm")
V14_LOCKED_SEEDS = (1, 2, 3)
V14_LOCKED_SEED = V14_LOCKED_SEEDS[0]
V14_EXCLUDED_TRAJECTORY = 7
V14_SCORE_TOLERANCE = 2e-5

V14_PROTOCOL = {
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
    "menu_seeds": list(V14_LOCKED_SEEDS),
    "score_batch_size": 8,
    "score_check_batch_size": 4,
    "score_reproduction_tolerance": V14_SCORE_TOLERANCE,
    "chunk_equivalence_tolerance": V14_SCORE_TOLERANCE,
    "decomposition_tolerance": 1e-8,
    "tie_atol": 1e-10,
    "tie_rtol": 1e-8,
    "production_pass_authoritative": True,
    "latent_coordinate_drift": "descriptive_only_no_threshold",
    "repeat_certificate": "18_fresh_processes_compared_to_repeat_0",
}

# Every execution/validation source is bound.  The v1.2 module is listed only
# because it supplies the released, pure model-independent primitives used by
# this clean v1.4 implementation; v1.2 artifacts and issuance are absent.
V14_SOURCE_BUNDLE_RELATIVE_FILES = (
    "experiments/__init__.py",
    "experiments/official_pusht_decision_audit.py",
    "experiments/official_pusht_batch_study.py",
    "experiments/official_pusht_v12_common_initial.py",
    "experiments/official_pusht_v14_common_initial.py",
    "experiments/official_pusht_v14_decision_audit.py",
    "experiments/official_pusht_v14_batch_study.py",
    "experiments/official_pusht_v14_bootstrap.py",
    "experiments/official_pusht_v14_launcher.py",
    "experiments/official_pusht_v14_certificate_supervisor.py",
    "experiments/analyze_official_pusht_v14.py",
    "latent_score_decomposition.py",
    "PWA_PUSHT_V14_PREREGISTRATION.md",
)

_BASE_ARRAY_KEYS = (
    "initial_actions_normalized", "initial_physical_actions",
    "initial_predicted_visual", "initial_encoded_real_visual", "goal_visual",
    "initial_predicted_proprio", "initial_encoded_real_proprio", "goal_proprio",
    "initial_objective_scores", "initial_native_returns", "initial_native_state_distances",
    "initial_final_states", "goal_state", "initial_reward_traces", "expert_reward_trace",
    "hold_position_noop_physical_actions", "hold_position_noop_final_state",
    "hold_position_noop_reward_trace", "initial_raw_visual_observations",
    "initial_raw_proprio_observations", "raw_initial_visual", "raw_initial_proprio",
    "raw_goal_visual", "raw_goal_proprio", "initial_raw_visual_digests",
    "initial_raw_proprio_digests",
)
V14_ARRAY_KEYS = _BASE_ARRAY_KEYS + (
    "initial_production_visual", "initial_production_proprio",
    "initial_predicted_visual_mse",
    "initial_observed_visual_mse",
    "initial_predicted_proprio_mse",
    "initial_observed_proprio_mse",
    "reencoded_chunk8_realized_visual",
    "reencoded_chunk8_realized_proprio",
    "reencoded_chunk8_goal_visual",
    "reencoded_chunk8_goal_proprio",
    "reencoded_chunk4_realized_visual",
    "reencoded_chunk4_realized_proprio",
    "reencoded_chunk4_goal_visual",
    "reencoded_chunk4_goal_proprio",
    "reencoded_chunk8_initial_visual", "reencoded_chunk8_initial_proprio",
    "reencoded_chunk4_initial_visual", "reencoded_chunk4_initial_proprio",
)

V14_CROSS_MODEL_ARRAY_KEYS = (
    "initial_actions_normalized", "initial_physical_actions", "initial_final_states",
    "goal_state", "initial_reward_traces", "initial_native_returns",
    "initial_native_state_distances", "initial_raw_visual_observations",
    "initial_raw_proprio_observations", "raw_initial_visual", "raw_initial_proprio",
    "raw_goal_visual", "raw_goal_proprio", "initial_raw_visual_digests",
    "initial_raw_proprio_digests", "expert_reward_trace",
    "hold_position_noop_physical_actions", "hold_position_noop_final_state",
    "hold_position_noop_reward_trace",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def v14_protocol_sha256() -> str:
    return canonical_sha256(V14_PROTOCOL)


def v14_source_bundle(root: Path | None = None) -> dict[str, Any]:
    project_root = Path(root).resolve() if root is not None else V14_CANONICAL_PROJECT_ROOT
    return {
        "version": V14_SOURCE_BUNDLE_VERSION,
        "files": {relative: file_sha256(project_root / relative) for relative in V14_SOURCE_BUNDLE_RELATIVE_FILES},
    }


def v14_source_bundle_sha256(root: Path | None = None) -> str:
    bundle = v14_source_bundle(root)
    missing = [name for name, digest in bundle["files"].items() if not digest]
    if missing:
        raise ValueError("v1.4 source bundle is incomplete: " + ", ".join(missing))
    return canonical_sha256(bundle)


def validate_v14_args(*, planning_repeats: int = 0, repeat_count: int = 1) -> None:
    """Reject bad invocation values before any optional model import/load."""
    if isinstance(planning_repeats, bool) or not isinstance(planning_repeats, (int, np.integer)):
        raise ValueError("planning_repeats must be an integer")
    if int(planning_repeats) < 0 or 0 < int(planning_repeats) < 3:
        raise ValueError("planning_repeats must be zero or at least three")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, (int, np.integer)):
        raise ValueError("repeat_count must be an integer")
    if int(repeat_count) < 1:
        raise ValueError("repeat_count must be positive")


@torch.no_grad()
def capture_initial_only_cem(
    unroll: Callable[[Any, Any], Any], objective: Callable[[Any, Any], Any], z_init: Any, *,
    num_samples: int = 300, horizon: int = 6, action_dim: int = 10,
    var_scale: float = 1.0, seed: int = V14_LOCKED_SEED, batch_size: int = 8,
    check_batch_size: int | None = 4, equivalence_tolerance: float = V14_SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Sample and score exactly one released iteration-zero population."""
    if (num_samples, horizon, action_dim) != (300, 6, 10):
        raise ValueError("v1.4 fixes a 300 x 6 x 10 iteration-0 population")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) not in V14_LOCKED_SEEDS:
        raise ValueError(f"v1.4 menu seed must be one of {V14_LOCKED_SEEDS}")
    if not math.isfinite(float(var_scale)) or float(var_scale) < 0:
        raise ValueError("var_scale must be finite and non-negative")
    if int(batch_size) < 1 or (check_batch_size is not None and int(check_batch_size) < 1):
        raise ValueError("score batch sizes must be positive")
    generator = torch.Generator(device=z_init.device).manual_seed(int(seed))
    mean = torch.zeros(horizon, action_dim, device=z_init.device)
    actions = mean[:, None, :] + float(var_scale) * torch.randn(
        horizon, num_samples, action_dim, device=z_init.device, generator=generator
    )
    actions[:, 0, :] = mean

    def score(population: Any, size: int) -> tuple[Any, dict[str, Any]]:
        values, terminal = [], {}
        for start in range(0, population.shape[1], size):
            batch = population[:, start:start + size]
            predicted = unroll(z_init, batch)
            value = torch.as_tensor(objective(predicted, batch)).reshape(-1)
            values.append(value)
            for key, tensor in predicted.items():
                terminal.setdefault(key, []).append(tensor[-1].detach().clone())
        return torch.cat(values), {key: torch.cat(parts, dim=0) for key, parts in terminal.items()}

    scores, terminal = score(actions, int(batch_size))
    if scores.numel() != 300 or not torch.isfinite(scores).all():
        raise ValueError("v1.4 iteration-0 proposal scores must be finite with one value per proposal")
    trace: list[dict[str, Any]] = []
    if check_batch_size is not None and int(check_batch_size) != int(batch_size):
        checked, _ = score(actions, int(check_batch_size))
        if not torch.isfinite(checked).all():
            raise ValueError("v1.4 reference scores are non-finite")
        drift = float(torch.max(torch.abs(scores - checked)).item())
        primary_selector = int(torch.argmin(scores).item())
        reference_selector = int(torch.argmin(checked).item())
        primary_ties = tie_indices(scores.detach().cpu().numpy(), lower_better=True)
        reference_ties = tie_indices(checked.detach().cpu().numpy(), lower_better=True)
        item = {
            "iteration": 0, "population_size": 300, "primary_batch_size": int(batch_size),
            "reference_batch_size": int(check_batch_size), "max_abs_score_difference": drift,
            "score_selector_agrees": primary_selector == reference_selector,
            "tie_set_agrees": primary_ties == reference_ties,
            "primary_tie_candidate_ids": [f"proposal_{index:04d}" for index in primary_ties],
            "reference_tie_candidate_ids": [f"proposal_{index:04d}" for index in reference_ties],
            "tolerance": float(equivalence_tolerance),
        }
        trace.append(item)
        if drift > float(equivalence_tolerance) or not item["score_selector_agrees"] or not item["tie_set_agrees"]:
            raise RuntimeError("v1.4 iteration-0 chunk equivalence failed: " + json.dumps(item, sort_keys=True))
    return {
        "initial_samples": actions.detach().clone(), "initial_scores": scores.detach().clone(),
        "initial_predicted": terminal, "iterations": 1, "num_samples": 300,
        "horizon": 6, "action_dim": 10, "var_scale": float(var_scale), "seed": int(seed),
        "score_batch_size": int(batch_size), "chunk_trace": trace,
        "inventory_kind": "common_initial", "outcome_blind": True,
        "rng_receipt": {
            "generator": "torch.Generator.manual_seed",
            "distribution": "standard_normal",
            "seed": int(seed), "mean": 0.0, "var_scale": float(var_scale),
            "draw_shape": [int(horizon), 300, int(action_dim)], "zero_proposal_index": 0,
            "normalized_menu_sha256": array_sha256(actions.permute(1, 0, 2).detach().cpu().numpy()),
        },
    }


def expand_model_actions(actions: Any, *, frameskip: int = 5, physical_action_dim: int = 2) -> torch.Tensor:
    tensor = torch.as_tensor(actions)
    if tensor.ndim != 3 or tensor.shape[-1] != frameskip * physical_action_dim:
        raise ValueError("expected [candidates, steps, frameskip * physical_action_dim] actions")
    return tensor.reshape(tensor.shape[0], tensor.shape[1] * frameskip, physical_action_dim)


def _finite(value: Any, name: str = "array", shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an array") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    try:
        if not np.isfinite(result.astype(np.float64, copy=False)).all():
            raise ValueError(f"{name} contains non-finite values")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(name + " contains"):
            raise
        raise ValueError(f"{name} is not finite numeric data") from exc
    return result


def validate_p_o_arrays(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the unchanged decomposition, with v1.4-owned validation errors."""
    kwargs.setdefault("alpha", float(V14_PROTOCOL["alpha"]))
    kwargs.setdefault("decomposition_tolerance", float(V14_PROTOCOL["decomposition_tolerance"]))
    return _v12.validate_p_o_arrays(*args, **kwargs)


def proposal_support_diagnostics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _v12.proposal_support_diagnostics(*args, **kwargs)


def standard_outer_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _v12.standard_outer_metrics(*args, **kwargs)


def _dataset_segment_hashes(artifact: Mapping[str, Any]) -> dict[str, str | None]:
    """Reconstruct the exact 31-state/30-action segment consumed by the runner."""
    try:
        root = Path(str(artifact["dataset_root"])) / str(artifact["dataset_split"])
        episode, offset = int(artifact["episode"]), int(artifact["offset"])
        states = torch.load(root / "states.pth", map_location="cpu", weights_only=False).float()
        velocities = torch.load(root / "velocities.pth", map_location="cpu", weights_only=False).float()
        raw_actions = torch.load(root / "rel_actions.pth", map_location="cpu", weights_only=False).float() / 100.0
        full_states = torch.cat([states, velocities], dim=-1)
        mean = torch.tensor([-0.0087, 0.0068], dtype=torch.float32)
        std = torch.tensor([0.2019, 0.2002], dtype=torch.float32)
        normalized_actions = (raw_actions - mean) / std
        return {
            "states_segment_sha256": array_sha256(full_states[episode, offset:offset + 31].numpy()),
            "actions_segment_sha256": array_sha256(normalized_actions[episode, offset:offset + 30].numpy()),
            "initial_state_sha256": array_sha256(full_states[episode, offset].numpy().astype(np.float64)),
        }
    except (OSError, RuntimeError, ValueError, IndexError, TypeError, KeyError):
        return {"states_segment_sha256": None, "actions_segment_sha256": None, "initial_state_sha256": None}


def tie_indices(values: Any, *, lower_better: bool) -> list[int]:
    values = _finite(values, "score vector").astype(np.float64, copy=False)
    if values.ndim != 1 or not len(values):
        return []
    best = float(np.min(values) if lower_better else np.max(values))
    ties = np.isclose(values, best, atol=float(V14_PROTOCOL["tie_atol"]), rtol=float(V14_PROTOCOL["tie_rtol"]))
    return [int(index) for index in np.flatnonzero(ties)]


def strict_choice(values: Any, *, lower_better: bool) -> int | None:
    ties = tie_indices(values, lower_better=lower_better)
    return ties[0] if len(ties) == 1 else None


def make_production_pass_receipt(**arrays: Any) -> dict[str, Any]:
    """Commit authoritative production arrays before serialization QC begins."""
    if not arrays:
        raise ValueError("v1.4 production pass receipt requires authoritative arrays")
    hashes: dict[str, str] = {}
    for name, value in arrays.items():
        _finite(value, f"production {name}")
        hashes[str(name)] = array_sha256(value)
    receipt = {
        "schema_version": "PWA-PushT-v1.4-production-pass-receipt",
        "authoritative": True,
        "qc_source": "production_pass_only",
        "array_hashes": hashes,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_production_pass_receipt(receipt: Mapping[str, Any], arrays: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["v1.4 production pass receipt is missing"]
    if receipt.get("schema_version") != "PWA-PushT-v1.4-production-pass-receipt" or receipt.get("authoritative") is not True:
        errors.append("v1.4 production pass receipt is not authoritative")
    base = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(base):
        errors.append("v1.4 production pass receipt hash mismatch")
    advertised = receipt.get("array_hashes")
    if not isinstance(advertised, Mapping) or set(advertised) != set(arrays):
        errors.append("v1.4 production pass receipt array allowlist mismatch")
    else:
        for name, value in arrays.items():
            if advertised.get(name) != array_sha256(value):
                errors.append(f"v1.4 production pass receipt {name} hash mismatch")
    return sorted(set(errors))


def p_o_g_j_summary(predicted_scores: Any, observed_scores: Any, native_returns: Any, native_state_distances: Any) -> dict[str, Any]:
    p = _finite(predicted_scores, "P", (300,)).astype(np.float64)
    o = _finite(observed_scores, "O", (300,)).astype(np.float64)
    j = _finite(native_returns, "J", (300,)).astype(np.float64)
    distance = _finite(native_state_distances, "G distance", (300,)).astype(np.float64)
    p_ties, o_ties = tie_indices(p, lower_better=True), tie_indices(o, lower_better=True)
    p_idx, o_idx = (p_ties[0] if len(p_ties) == 1 else None), (o_ties[0] if len(o_ties) == 1 else None)
    j_idx, g_idx = int(np.argmax(j)), int(np.argmin(distance))
    best_j, best_g = float(np.max(j)), float(np.min(distance))
    return {
        "candidate_count": 300, "predicted_choice": p_idx, "encoded_real_choice": o_idx,
        "native_reward_choice": j_idx, "goal_aligned_choice": g_idx,
        "predicted_tie_indices": p_ties, "encoded_real_tie_indices": o_ties,
        "reward_comparison_abstained": p_idx is None or o_idx is None,
        "goal_comparison_abstained": p_idx is None or o_idx is None,
        "R_P_J": best_j - float(j[p_idx]) if p_idx is not None else None,
        "R_O_J": best_j - float(j[o_idx]) if o_idx is not None else None,
        "R_P_G": float(distance[p_idx] - best_g) if p_idx is not None else None,
        "R_O_G": float(distance[o_idx] - best_g) if o_idx is not None else None,
        "Delta_J": float(j[o_idx] - j[p_idx]) if p_idx is not None and o_idx is not None else None,
        "Delta_G": float(distance[p_idx] - distance[o_idx]) if p_idx is not None and o_idx is not None else None,
        "P_finite": True, "O_finite": True,
    }


def make_reencoding_diagnostic(
    *, production_predicted_visual: Any, production_predicted_proprio: Any,
    production_realized_visual: Any, production_realized_proprio: Any,
    production_goal_visual: Any, production_goal_proprio: Any,
    chunk8_realized_visual: Any, chunk8_realized_proprio: Any,
    chunk8_goal_visual: Any, chunk8_goal_proprio: Any,
    chunk4_realized_visual: Any, chunk4_realized_proprio: Any,
    chunk4_goal_visual: Any, chunk4_goal_proprio: Any,
    production_p: Any, production_o: Any, alpha: float = 0.1,
    production_initial_visual: Any | None = None, production_initial_proprio: Any | None = None,
    chunk8_initial_visual: Any | None = None, chunk8_initial_proprio: Any | None = None,
    chunk4_initial_visual: Any | None = None, chunk4_initial_proprio: Any | None = None,
    terminal_visual_sha256: str | None = None, terminal_proprio_sha256: str | None = None,
    goal_visual_sha256: str | None = None, goal_proprio_sha256: str | None = None,
    initial_visual_sha256: str | None = None, initial_proprio_sha256: str | None = None,
    realized_visual_sha256: str | None = None, realized_proprio_sha256: str | None = None,
    encoded_goal_visual_sha256: str | None = None, encoded_goal_proprio_sha256: str | None = None,
) -> dict[str, Any]:
    """Anchor both serialization checks directly to the production pass."""
    names = {
        "production_predicted_visual": production_predicted_visual,
        "production_predicted_proprio": production_predicted_proprio,
        "production_realized_visual": production_realized_visual,
        "production_realized_proprio": production_realized_proprio,
        "production_goal_visual": production_goal_visual,
        "production_goal_proprio": production_goal_proprio,
        "chunk8_realized_visual": chunk8_realized_visual, "chunk8_realized_proprio": chunk8_realized_proprio,
        "chunk8_goal_visual": chunk8_goal_visual, "chunk8_goal_proprio": chunk8_goal_proprio,
        "chunk4_realized_visual": chunk4_realized_visual, "chunk4_realized_proprio": chunk4_realized_proprio,
        "chunk4_goal_visual": chunk4_goal_visual, "chunk4_goal_proprio": chunk4_goal_proprio,
        "production_o": production_o,
    }
    arrays = {name: _finite(value, name).astype(np.float64, copy=False) for name, value in names.items()}
    initial_values = {
        "production_initial_visual": production_initial_visual,
        "production_initial_proprio": production_initial_proprio,
        "chunk8_initial_visual": chunk8_initial_visual,
        "chunk8_initial_proprio": chunk8_initial_proprio,
        "chunk4_initial_visual": chunk4_initial_visual,
        "chunk4_initial_proprio": chunk4_initial_proprio,
    }
    initial_present = any(value is not None for value in initial_values.values())
    if initial_present and not all(value is not None for value in initial_values.values()):
        raise ValueError("v1.4 initial re-encoding diagnostic requires all production/chunk arrays")
    if initial_present:
        arrays.update({name: _finite(value, name).astype(np.float64, copy=False) for name, value in initial_values.items()})
    production_p_array = _finite(production_p, "production P").astype(np.float64)
    production_o_array = _finite(production_o, "production O").astype(np.float64)
    production_decomp = decompose_latent_scores(
        arrays["production_predicted_visual"], arrays["production_realized_visual"], arrays["production_goal_visual"],
        predicted_proprio=arrays["production_predicted_proprio"], realized_proprio=arrays["production_realized_proprio"],
        goal_proprio=arrays["production_goal_proprio"], alpha=alpha,
    )
    if not np.array_equal(np.asarray(production_decomp["predicted_score"], dtype=np.float64), production_p_array) or not np.array_equal(np.asarray(production_decomp["observed_score"], dtype=np.float64), production_o_array):
        raise ValueError("production P/O receipt does not reproduce authoritative latent arrays")

    def score(realized_visual: np.ndarray, realized_proprio: np.ndarray, goal_visual: np.ndarray, goal_proprio: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        decomp = decompose_latent_scores(
            arrays["production_predicted_visual"], realized_visual, goal_visual,
            predicted_proprio=arrays["production_predicted_proprio"], realized_proprio=realized_proprio,
            goal_proprio=goal_proprio, alpha=alpha,
        )
        return np.asarray(decomp["observed_score"], dtype=np.float64), decomp
    chunk8_o, chunk8_decomp = score(arrays["chunk8_realized_visual"], arrays["chunk8_realized_proprio"], arrays["chunk8_goal_visual"], arrays["chunk8_goal_proprio"])
    chunk4_o, chunk4_decomp = score(arrays["chunk4_realized_visual"], arrays["chunk4_realized_proprio"], arrays["chunk4_goal_visual"], arrays["chunk4_goal_proprio"])
    if production_o_array.shape != chunk8_o.shape or production_o_array.shape != chunk4_o.shape:
        raise ValueError("production and re-encoded O vectors must have identical shape")
    def coordinate_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        absolute = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).reshape(-1)
        if not np.isfinite(absolute).all():
            raise ValueError("latent coordinate drift is non-finite")
        return {
            "max_abs": float(np.max(absolute)), "rms": float(np.sqrt(np.mean(np.square(absolute)))),
            "median_abs": float(np.median(absolute)), "p95_abs": float(np.quantile(absolute, 0.95)),
        }
    coordinate_drift = {}
    for name, produced, checked in (
        ("chunk8", arrays["production_realized_visual"], arrays["chunk8_realized_visual"]),
        ("chunk4", arrays["production_realized_visual"], arrays["chunk4_realized_visual"]),
    ):
        proprio_produced = arrays["production_realized_proprio"]
        goal_produced = arrays["production_goal_visual"]
        goal_produced_p = arrays["production_goal_proprio"]
        if name == "chunk8":
            checked_proprio, checked_goal, checked_goal_p = arrays["chunk8_realized_proprio"], arrays["chunk8_goal_visual"], arrays["chunk8_goal_proprio"]
        else:
            checked_proprio, checked_goal, checked_goal_p = arrays["chunk4_realized_proprio"], arrays["chunk4_goal_visual"], arrays["chunk4_goal_proprio"]
        coordinate_drift[name] = {
            "terminal_visual": coordinate_stats(produced, checked),
            "terminal_proprio": coordinate_stats(proprio_produced, checked_proprio),
            "goal_visual": coordinate_stats(goal_produced, checked_goal),
            "goal_proprio": coordinate_stats(goal_produced_p, checked_goal_p),
        }
        if initial_present:
            coordinate_drift[name]["initial_visual"] = coordinate_stats(
                arrays["production_initial_visual"], arrays[f"{name}_initial_visual"]
            )
            coordinate_drift[name]["initial_proprio"] = coordinate_stats(
                arrays["production_initial_proprio"], arrays[f"{name}_initial_proprio"]
            )
    coordinate_drift.update({"threshold": None, "interpretation": "descriptive coordinate drift; no coordinate threshold"})
    production_ties = tie_indices(production_o_array, lower_better=True)
    chunk8_ties = tie_indices(chunk8_o, lower_better=True)
    chunk4_ties = tie_indices(chunk4_o, lower_better=True)
    production_p_ties = tie_indices(production_p_array, lower_better=True)
    chunk8_p = np.asarray(chunk8_decomp["predicted_score"], dtype=np.float64)
    chunk4_p = np.asarray(chunk4_decomp["predicted_score"], dtype=np.float64)
    chunk8_p_ties = tie_indices(chunk8_p, lower_better=True)
    chunk4_p_ties = tie_indices(chunk4_p, lower_better=True)
    component_drift = {}
    for key, source_key in (
        ("P_total", "predicted_score"), ("O_total", "observed_score"),
        ("P_visual", "predicted_visual_mse"), ("O_visual", "observed_visual_mse"),
        ("P_proprio", "predicted_proprio_mse"), ("O_proprio", "observed_proprio_mse"),
    ):
        source = np.asarray(production_decomp[source_key], dtype=np.float64)
        c8 = np.asarray(chunk8_decomp[source_key], dtype=np.float64)
        c4 = np.asarray(chunk4_decomp[source_key], dtype=np.float64)
        component_drift[key] = {
            "chunk8_max_abs_difference": float(np.max(np.abs(c8 - source))),
            "chunk4_max_abs_difference": float(np.max(np.abs(c4 - source))),
            "tolerance": V14_SCORE_TOLERANCE,
        }
    return {
        "passed": bool(
            np.isfinite(production_o_array).all()
            and all(float(item[f"{chunk}_max_abs_difference"]) <= V14_SCORE_TOLERANCE for item in component_drift.values() for chunk in ("chunk8", "chunk4"))
            and chunk8_ties == production_ties and chunk4_ties == production_ties
            and chunk8_p_ties == production_p_ties and chunk4_p_ties == production_p_ties
        ),
        "production_pass_authoritative": True,
        "initial_reencoding_present": initial_present,
        "production_chunk_size": 8,
        "cross_check_chunk_size": 4,
        "chunk8_anchor": "production_pass",
        "chunk4_anchor": "production_pass",
        "latent_coordinate_drift": coordinate_drift,
        "component_score_drift": component_drift,
        "P_score_max_abs_difference": float(np.max(np.abs(chunk8_p - production_p_array))),
        "O_score_max_abs_difference": float(np.max(np.abs(chunk8_o - production_o_array))),
        "cross_chunk_P_score_max_abs_difference": float(np.max(np.abs(chunk4_p - production_p_array))),
        "cross_chunk_O_score_max_abs_difference": float(np.max(np.abs(chunk4_o - production_o_array))),
        "O_selector_agrees": chunk8_ties == production_ties,
        "cross_chunk_O_selector_agrees": chunk4_ties == production_ties,
        "P_selector_agrees": chunk8_p_ties == production_p_ties,
        "cross_chunk_P_selector_agrees": chunk4_p_ties == production_p_ties,
        "production_P_tie_indices": production_p_ties,
        "chunk8_P_tie_indices": chunk8_p_ties,
        "chunk4_P_tie_indices": chunk4_p_ties,
        "production_O_tie_indices": production_ties,
        "chunk8_O_tie_indices": chunk8_ties,
        "chunk4_O_tie_indices": chunk4_ties,
        "production_P_tie_candidate_ids": [f"proposal_{index:04d}" for index in production_p_ties],
        "chunk8_P_tie_candidate_ids": [f"proposal_{index:04d}" for index in chunk8_p_ties],
        "chunk4_P_tie_candidate_ids": [f"proposal_{index:04d}" for index in chunk4_p_ties],
        "production_O_tie_candidate_ids": [f"proposal_{index:04d}" for index in production_ties],
        "chunk8_O_tie_candidate_ids": [f"proposal_{index:04d}" for index in chunk8_ties],
        "chunk4_O_tie_candidate_ids": [f"proposal_{index:04d}" for index in chunk4_ties],
        "score_tolerance": V14_SCORE_TOLERANCE,
        "terminal_visual_sha256": terminal_visual_sha256,
        "terminal_proprio_sha256": terminal_proprio_sha256,
        "goal_visual_sha256": goal_visual_sha256,
        "goal_proprio_sha256": goal_proprio_sha256,
        "initial_visual_sha256": initial_visual_sha256,
        "initial_proprio_sha256": initial_proprio_sha256,
        "realized_visual_sha256": realized_visual_sha256,
        "realized_proprio_sha256": realized_proprio_sha256,
        "encoded_goal_visual_sha256": encoded_goal_visual_sha256,
        "encoded_goal_proprio_sha256": encoded_goal_proprio_sha256,
    }


def _component_payload(decomp: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "initial_predicted_visual_mse": np.asarray(decomp["predicted_visual_mse"], dtype=np.float64),
        "initial_observed_visual_mse": np.asarray(decomp["observed_visual_mse"], dtype=np.float64),
        "initial_predicted_proprio_mse": np.asarray(decomp["predicted_proprio_mse"], dtype=np.float64),
        "initial_observed_proprio_mse": np.asarray(decomp["observed_proprio_mse"], dtype=np.float64),
    }


def build_v14_artifact(
    *, model: str, decision_id: str, episode: int, offset: int, seed: int = V14_LOCKED_SEED,
    normalized_actions: Any, physical_actions: Any,
    predicted_visual: Any, realized_visual: Any, goal_visual: Any,
    predicted_proprio: Any, realized_proprio: Any, goal_proprio: Any,
    objective_scores: Any, native_returns: Any, native_state_distances: Any,
    final_states: Any, goal_state: Any, reward_traces: Any,
    expert_reward_trace: Any, noop_physical_actions: Any, noop_final_state: Any,
    noop_reward_trace: Any, raw_visual_observations: Any,
    raw_proprio_observations: Any, raw_initial_visual: Any, raw_initial_proprio: Any,
    raw_goal_visual: Any, raw_goal_proprio: Any, raw_reencoding_gate: Mapping[str, Any],
    raw_visual_digests: Any, raw_proprio_digests: Any, trajectory_id: str | None = None,
    arrays_out: Path | None = None, proposal_diagnostics: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None, planning_gate: Mapping[str, Any] | None = None,
    certificate: Mapping[str, Any] | None = None, bindings: Mapping[str, Any] | None = None,
    diagnostic_arrays: Mapping[str, Any] | None = None,
    production_pass_receipt: Mapping[str, Any] | None = None,
    certificate_issuance: Mapping[str, Any] | None = None,
    study_issuance: Mapping[str, Any] | None = None,
    row_capability: Mapping[str, Any] | None = None,
    certificate_attempt_capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v1.4 artifact while retaining v1.2's exact primitive checks."""
    if raw_reencoding_gate.get("passed") is not True:
        raise ValueError("v1.4 re-encoding decision gate did not pass")
    # The v1.2 builder is used solely as a compact implementation of the
    # unchanged finite/J/G/raw/decomposition checks.  Its output is converted
    # into a new v1.4 schema and independently sealed below.
    base = _v12.build_v12_artifact(
        model=model, decision_id=decision_id, episode=episode, offset=offset,
        seed=int(seed),
        normalized_actions=normalized_actions, physical_actions=physical_actions,
        predicted_visual=predicted_visual, realized_visual=realized_visual, goal_visual=goal_visual,
        predicted_proprio=predicted_proprio, realized_proprio=realized_proprio, goal_proprio=goal_proprio,
        objective_scores=objective_scores, native_returns=native_returns,
        native_state_distances=native_state_distances, final_states=final_states,
        goal_state=goal_state, reward_traces=reward_traces,
        expert_reward_trace=expert_reward_trace, noop_physical_actions=noop_physical_actions,
        noop_final_state=noop_final_state, noop_reward_trace=noop_reward_trace,
        raw_visual_observations=raw_visual_observations, raw_proprio_observations=raw_proprio_observations,
        raw_initial_visual=raw_initial_visual, raw_initial_proprio=raw_initial_proprio,
        raw_goal_visual=raw_goal_visual, raw_goal_proprio=raw_goal_proprio,
        raw_reencoding_gate=raw_reencoding_gate, raw_visual_digests=raw_visual_digests,
        raw_proprio_digests=raw_proprio_digests, trajectory_id=trajectory_id,
        arrays_out=arrays_out, proposal_diagnostics=proposal_diagnostics, baselines=baselines,
        planning_gate=planning_gate, certificate=certificate, bindings=bindings,
    )
    if arrays_out is None or diagnostic_arrays is None:
        raise ValueError("v1.4 requires serialized chunk-8/chunk-4 diagnostic arrays")
    required_diag = (
        "initial_production_visual", "initial_production_proprio",
        "chunk8_realized_visual", "chunk8_realized_proprio", "chunk8_goal_visual", "chunk8_goal_proprio",
        "chunk4_realized_visual", "chunk4_realized_proprio", "chunk4_goal_visual", "chunk4_goal_proprio",
        "chunk8_initial_visual", "chunk8_initial_proprio",
        "chunk4_initial_visual", "chunk4_initial_proprio",
    )
    missing = [key for key in required_diag if key not in diagnostic_arrays]
    if missing:
        raise ValueError("v1.4 diagnostic arrays missing: " + ", ".join(missing))
    with np.load(arrays_out, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    receipt_keys = (
        "initial_actions_normalized", "initial_physical_actions", "initial_predicted_visual",
        "initial_encoded_real_visual", "goal_visual", "initial_predicted_proprio",
        "initial_encoded_real_proprio", "goal_proprio", "initial_objective_scores",
        "initial_native_returns", "initial_native_state_distances", "initial_final_states",
        "goal_state", "initial_reward_traces", "initial_raw_visual_observations",
        "initial_raw_proprio_observations", "raw_initial_visual", "raw_initial_proprio",
        "raw_goal_visual", "raw_goal_proprio",
        "initial_production_visual", "initial_production_proprio",
    )
    if production_pass_receipt is None:
        raise ValueError("v1.4 production pass receipt is required before diagnostic arrays")
    # Initial production encodings are diagnostic inputs but are part of the
    # pre-QC authoritative receipt, so expose them before receipt validation.
    payload["initial_production_visual"] = np.asarray(diagnostic_arrays["initial_production_visual"])
    payload["initial_production_proprio"] = np.asarray(diagnostic_arrays["initial_production_proprio"])
    receipt_errors = validate_production_pass_receipt(
        production_pass_receipt, {key: payload[key] for key in receipt_keys}
    )
    if receipt_errors:
        raise ValueError("invalid v1.4 production pass receipt: " + "; ".join(receipt_errors))
    decomp = validate_p_o_arrays(
        predicted_visual, realized_visual, goal_visual,
        predicted_proprio=predicted_proprio, realized_proprio=realized_proprio,
        goal_proprio=goal_proprio,
    )["decomposition"]
    payload.update(_component_payload(decomp))
    payload.update({
        "initial_production_visual": _finite(diagnostic_arrays["initial_production_visual"], "initial production visual").astype(np.float32),
        "initial_production_proprio": _finite(diagnostic_arrays["initial_production_proprio"], "initial production proprio").astype(np.float32),
        "reencoded_chunk8_realized_visual": _finite(diagnostic_arrays["chunk8_realized_visual"], "chunk8 realized visual").astype(np.float32),
        "reencoded_chunk8_realized_proprio": _finite(diagnostic_arrays["chunk8_realized_proprio"], "chunk8 realized proprio").astype(np.float32),
        "reencoded_chunk8_goal_visual": _finite(diagnostic_arrays["chunk8_goal_visual"], "chunk8 goal visual").astype(np.float32),
        "reencoded_chunk8_goal_proprio": _finite(diagnostic_arrays["chunk8_goal_proprio"], "chunk8 goal proprio").astype(np.float32),
        "reencoded_chunk4_realized_visual": _finite(diagnostic_arrays["chunk4_realized_visual"], "chunk4 realized visual").astype(np.float32),
        "reencoded_chunk4_realized_proprio": _finite(diagnostic_arrays["chunk4_realized_proprio"], "chunk4 realized proprio").astype(np.float32),
        "reencoded_chunk4_goal_visual": _finite(diagnostic_arrays["chunk4_goal_visual"], "chunk4 goal visual").astype(np.float32),
        "reencoded_chunk4_goal_proprio": _finite(diagnostic_arrays["chunk4_goal_proprio"], "chunk4 goal proprio").astype(np.float32),
        "reencoded_chunk8_initial_visual": _finite(diagnostic_arrays["chunk8_initial_visual"], "chunk8 initial visual").astype(np.float32),
        "reencoded_chunk8_initial_proprio": _finite(diagnostic_arrays["chunk8_initial_proprio"], "chunk8 initial proprio").astype(np.float32),
        "reencoded_chunk4_initial_visual": _finite(diagnostic_arrays["chunk4_initial_visual"], "chunk4 initial visual").astype(np.float32),
        "reencoded_chunk4_initial_proprio": _finite(diagnostic_arrays["chunk4_initial_proprio"], "chunk4 initial proprio").astype(np.float32),
    })
    np.savez_compressed(arrays_out, **payload)
    p = np.asarray(decomp["predicted_score"], dtype=np.float64)
    o = np.asarray(decomp["observed_score"], dtype=np.float64)
    summary = p_o_g_j_summary(p, o, native_returns, native_state_distances)
    ids = [f"proposal_{index:04d}" for index in range(300)]
    decision = dict(base["decision"])
    decision.update({
        "predicted_tie_candidate_ids": [ids[index] for index in summary["predicted_tie_indices"]],
        "encoded_real_tie_candidate_ids": [ids[index] for index in summary["encoded_real_tie_indices"]],
        "v14_selector_contract": "exact tied-candidate-ID set under atol=1e-10, rtol=1e-8",
    })
    base.update({
        "schema_version": V14_ARTIFACT_SCHEMA_VERSION,
        "protocol": V14_PROTOCOL,
        "protocol_sha256": v14_protocol_sha256(),
        "source_bundle_version": V14_SOURCE_BUNDLE_VERSION,
        "source_bundle_sha256": v14_source_bundle_sha256(),
        "audit_status": "v1.4_common_initial_only",
        "production_pass_authoritative": True,
        "production_pass_receipt": dict(production_pass_receipt),
        "latent_coordinate_drift": raw_reencoding_gate.get("latent_coordinate_drift"),
        "raw_evidence_reencoding_gate": dict(raw_reencoding_gate),
        "v14_reencoding_diagnostic": dict(raw_reencoding_gate),
        "decision": decision,
        "v1_2_evidence_reused": False,
        "certificate_issuance": dict(certificate_issuance) if certificate_issuance is not None else None,
        "study_issuance": dict(study_issuance) if study_issuance is not None else None,
        "row_capability": dict(row_capability) if row_capability is not None else None,
        "certificate_attempt_capability": dict(certificate_attempt_capability) if certificate_attempt_capability is not None else None,
        "proposal_rng_receipt": {
            **(dict(proposal_diagnostics.get("rng_receipt", {})) if isinstance(proposal_diagnostics, Mapping) and isinstance(proposal_diagnostics.get("rng_receipt"), Mapping) else {}),
            "generator": "torch.Generator.manual_seed", "distribution": "standard_normal",
            "seed": int(seed), "mean": 0.0, "var_scale": 1.0,
            "draw_shape": [6, 300, 10], "zero_proposal_index": 0,
            "normalized_menu_sha256": array_sha256(payload["initial_actions_normalized"]),
        },
        "menu_seed": int(seed),
        "array_hashes": {key: array_sha256(payload[key]) for key in sorted(payload)},
        "arrays_sidecar": str(Path(arrays_out).resolve()),
        "arrays_sidecar_sha256": file_sha256(Path(arrays_out)),
    })
    base.setdefault("baselines", {})["random"] = {
        "selection": "exact uniform expectation", "candidate_id": None, "candidate_index": None,
        "J": float(np.mean(np.asarray(native_returns, dtype=np.float64))),
        "G": float(np.mean(-np.asarray(native_state_distances, dtype=np.float64))),
    }
    base["initial_inventory"]["schema_version"] = V14_ARTIFACT_SCHEMA_VERSION
    base["initial_inventory"]["score_capture"] = "cem_iteration_0"
    return base


def validate_v14_sidecar(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            missing = sorted(set(V14_ARRAY_KEYS) - set(archive.files))
            unexpected = sorted(set(archive.files) - set(V14_ARRAY_KEYS))
            errors.extend(f"v1.4 sidecar missing {key}" for key in missing)
            errors.extend(f"v1.4 sidecar contains unexpected array {key}" for key in unexpected)
            if missing or unexpected:
                return sorted(set(errors))
            if archive["initial_actions_normalized"].shape != (300, 6, 10):
                errors.append("v1.4 normalized proposal array shape mismatch")
            if archive["initial_physical_actions"].shape != (300, 30, 2):
                errors.append("v1.4 physical proposal array shape mismatch")
            if not np.array_equal(np.asarray(archive["initial_actions_normalized"][0]), np.zeros((6, 10), dtype=np.float32)):
                errors.append("v1.4 proposal 0 must be the exact zero action")
            if archive["initial_raw_visual_observations"].dtype != np.uint8 or archive["initial_raw_visual_observations"].shape != (300, 1, 3, 224, 224):
                errors.append("v1.4 raw terminal visual shape/dtype mismatch")
            for key in V14_ARRAY_KEYS:
                try:
                    if not np.isfinite(np.asarray(archive[key]).astype(np.float64, copy=False)).all():
                        errors.append(f"v1.4 sidecar contains non-finite {key}")
                except (TypeError, ValueError):
                    errors.append(f"v1.4 sidecar contains non-numeric {key}")
            normalized = np.asarray(archive["initial_actions_normalized"], dtype=np.float32)
            expected_physical = normalized.reshape(300, 30, 2) * np.asarray([0.2019, 0.2002], dtype=np.float32) + np.asarray([-0.0087, 0.0068], dtype=np.float32)
            if not np.array_equal(expected_physical, np.asarray(archive["initial_physical_actions"], dtype=np.float32)):
                errors.append("v1.4 physical actions do not exactly denormalize from normalized actions")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"v1.4 sidecar unreadable: {exc}")
    return sorted(set(errors))


def validate_v14_artifact(
    artifact: Mapping[str, Any], entry: Mapping[str, Any], *, arrays_path: Path | None = None,
    expected_bindings: Mapping[str, Any] | None = None,
    expected_certificate_issuance: Mapping[str, Any] | None = None,
    expected_certificate_sha256: str | None = None,
    expected_study_issuance: Mapping[str, Any] | None = None,
    expected_row_capability: Mapping[str, Any] | None = None,
    expected_menu_sha256: str | None = None,
    allow_excluded_engineering: bool = False,
) -> list[str]:
    """Recompute the v1.4 artifact and reject all older schemas."""
    errors: list[str] = []
    schema = artifact.get("schema_version")
    if schema == _v12.V12_ARTIFACT_SCHEMA_VERSION or schema == "PWA-PushT-v1.1":
        errors.append("v1.4 validator explicitly rejects v1.1/v1.2 artifacts")
    if schema != V14_ARTIFACT_SCHEMA_VERSION:
        errors.append("artifact is not the v1.4 initial-only schema")
    if artifact.get("protocol") != V14_PROTOCOL or artifact.get("protocol_sha256") != v14_protocol_sha256():
        errors.append("v1.4 artifact protocol commitment mismatch")
    if artifact.get("source_bundle_sha256") != v14_source_bundle_sha256():
        errors.append("v1.4 artifact source bundle commitment mismatch")
    if artifact.get("inventory") != "common_initial" or artifact.get("outcome_blind") is not True:
        errors.append("v1.4 artifact is not outcome-blind common-initial inventory")
    if artifact.get("episode") == V14_EXCLUDED_TRAJECTORY and not allow_excluded_engineering:
        errors.append("v1.4 scientific artifact uses excluded trajectory 7")
    if allow_excluded_engineering and (artifact.get("episode") != V14_EXCLUDED_TRAJECTORY or entry.get("episode") != V14_EXCLUDED_TRAJECTORY):
        errors.append("v1.4 engineering exclusion override is not an exact trajectory-7 contract")
    for key in ("decision_id", "trajectory_id", "episode", "offset", "seed", "model"):
        if entry.get(key) is not None and artifact.get(key) != entry.get(key):
            errors.append(f"v1.4 artifact {key} does not match manifest")
    required_bindings = (
        "official_git_commit", "environment_seed", "menu_seed", "runtime", "hardware",
        "runtime_sha256", "hardware_sha256", "states_segment_sha256", "actions_segment_sha256",
        "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256", "goal_observation_sha256",
    )
    for key in required_bindings:
        if key not in artifact or artifact.get(key) in (None, ""):
            errors.append(f"v1.4 artifact mandatory binding {key} is missing")
    for key in ("runtime_sha256", "hardware_sha256", "states_segment_sha256", "actions_segment_sha256", "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256", "goal_observation_sha256"):
        if key in artifact and not _is_sha256(artifact.get(key)):
            errors.append(f"v1.4 artifact mandatory binding {key} is not a SHA-256 digest")
    if artifact.get("environment_seed") != 1:
        errors.append("v1.4 artifact environment_seed must be exactly 1")
    if artifact.get("menu_seed") != entry.get("seed"):
        errors.append("v1.4 artifact menu_seed does not match manifest seed")
    rng_receipt = artifact.get("proposal_rng_receipt")
    expected_rng = {
        "generator": "torch.Generator.manual_seed", "distribution": "standard_normal",
        "seed": entry.get("seed"), "mean": 0.0, "var_scale": 1.0,
        "draw_shape": [6, 300, 10], "zero_proposal_index": 0,
    }
    if not isinstance(rng_receipt, Mapping) or any(rng_receipt.get(key) != value for key, value in expected_rng.items()):
        errors.append("v1.4 proposal RNG receipt is not the locked initial Gaussian contract")
    if expected_menu_sha256 is not None and (
        not isinstance(rng_receipt, Mapping)
        or rng_receipt.get("normalized_menu_sha256") != expected_menu_sha256
    ):
        errors.append("v1.4 proposal menu does not match the certified seed inventory")
    if artifact.get("v1_2_evidence_reused") is not False:
        errors.append("v1.4 artifact v1_2_evidence_reused must be false")
    for key, expected in (expected_bindings or {}).items():
        if artifact.get(key) != expected:
            errors.append(f"v1.4 artifact binding {key} does not match certificate")
    if expected_certificate_issuance is not None and artifact.get("certificate_issuance") != dict(expected_certificate_issuance):
        errors.append("v1.4 artifact certificate-issuance binding does not match certificate")
    if expected_certificate_sha256 is not None:
        certificate_ref = artifact.get("reproducibility_certificate")
        if not isinstance(certificate_ref, Mapping) or certificate_ref.get("status") != "passed" or certificate_ref.get("sha256") != expected_certificate_sha256:
            errors.append("v1.4 artifact reproducibility-certificate binding does not match certificate")
    if expected_study_issuance is not None and artifact.get("study_issuance") != dict(expected_study_issuance):
        errors.append("v1.4 artifact study-issuance binding does not match study")
    if expected_row_capability is not None and artifact.get("row_capability") != dict(expected_row_capability):
        errors.append("v1.4 artifact row-capability binding does not match its one-shot worker authorization")
    certificate_attempt_capability = artifact.get("certificate_attempt_capability")
    if certificate_attempt_capability is not None and (
        not isinstance(certificate_attempt_capability, Mapping)
        or certificate_attempt_capability.get("status") != "consumed"
        or not _is_sha256(certificate_attempt_capability.get("sha256"))
        or not _is_sha256(certificate_attempt_capability.get("capability_sha256"))
    ):
        errors.append("v1.4 certificate-attempt capability binding is not a consumed one-shot receipt")
    for value_key, hash_key in (("runtime", "runtime_sha256"), ("hardware", "hardware_sha256")):
        value = artifact.get(value_key)
        advertised_hash = artifact.get(hash_key)
        try:
            expected_hash = canonical_sha256(value) if isinstance(value, Mapping) else None
        except (TypeError, ValueError):
            expected_hash = None
        if expected_hash is None or advertised_hash != expected_hash:
            errors.append(f"v1.4 {value_key} fingerprint hash mismatch")
    segment_hashes = _dataset_segment_hashes(artifact)
    if segment_hashes["states_segment_sha256"] is not None:
        for key, expected in segment_hashes.items():
            if artifact.get(key) != expected:
                errors.append(f"v1.4 {key} does not reproduce from the bound dataset segment")
    issuance_ref = artifact.get("certificate_issuance")
    if issuance_ref is not None:
        if not isinstance(issuance_ref, Mapping) or set(issuance_ref) != {"path", "sha256", "issuance_sha256"}:
            errors.append("v1.4 artifact certificate-issuance reference schema mismatch")
        elif not isinstance(issuance_ref.get("path"), str) or file_sha256(Path(issuance_ref["path"])) != issuance_ref.get("sha256"):
            errors.append("v1.4 artifact certificate-issuance file binding mismatch")
    study_ref = artifact.get("study_issuance")
    if study_ref is not None:
        if not isinstance(study_ref, Mapping) or set(study_ref) != {"path", "sha256", "issuance_sha256"}:
            errors.append("v1.4 artifact study-issuance reference schema mismatch")
        elif not isinstance(study_ref.get("path"), str) or file_sha256(Path(study_ref["path"])) != study_ref.get("sha256"):
            errors.append("v1.4 artifact study-issuance file binding mismatch")
    sidecar = Path(arrays_path or artifact.get("arrays_sidecar", "")) if (arrays_path or artifact.get("arrays_sidecar")) else None
    if sidecar is None:
        errors.append("v1.4 artifact has no arrays sidecar")
        return sorted(set(errors))
    errors.extend(validate_v14_sidecar(sidecar))
    if file_sha256(sidecar) != artifact.get("arrays_sidecar_sha256"):
        errors.append("v1.4 arrays sidecar hash mismatch")
    try:
        with np.load(sidecar, allow_pickle=False) as archive:
            hashes = {key: array_sha256(archive[key]) for key in sorted(archive.files)}
            if artifact.get("array_hashes") != hashes:
                errors.append("v1.4 artifact array hashes do not match sidecar")
            if isinstance(rng_receipt, Mapping) and rng_receipt.get("normalized_menu_sha256") != array_sha256(archive["initial_actions_normalized"]):
                errors.append("v1.4 proposal RNG receipt menu hash does not match NPZ")
            receipt_keys = (
                "initial_actions_normalized", "initial_physical_actions", "initial_predicted_visual",
                "initial_encoded_real_visual", "goal_visual", "initial_predicted_proprio",
                "initial_encoded_real_proprio", "goal_proprio", "initial_objective_scores",
                "initial_native_returns", "initial_native_state_distances", "initial_final_states",
                "goal_state", "initial_reward_traces", "initial_raw_visual_observations",
                "initial_raw_proprio_observations", "raw_initial_visual", "raw_initial_proprio",
                "raw_goal_visual", "raw_goal_proprio", "initial_production_visual",
                "initial_production_proprio",
            )
            errors.extend(validate_production_pass_receipt(
                artifact.get("production_pass_receipt", {}),
                {key: archive[key] for key in receipt_keys},
            ))
            decomp = validate_p_o_arrays(
                archive["initial_predicted_visual"], archive["initial_encoded_real_visual"], archive["goal_visual"],
                predicted_proprio=archive["initial_predicted_proprio"], realized_proprio=archive["initial_encoded_real_proprio"],
                goal_proprio=archive["goal_proprio"],
            )
            p, o = decomp["P"], decomp["O"]
            expected_components = _component_payload(decomp["decomposition"])
            for key, expected in expected_components.items():
                if not np.array_equal(np.asarray(archive[key]), expected):
                    errors.append(f"v1.4 {key} does not reproduce from authoritative latent arrays")
            captured = _finite(archive["initial_objective_scores"], "captured P").astype(np.float64)
            native = _finite(archive["initial_native_returns"], "J").astype(np.float64)
            distance = _finite(archive["initial_native_state_distances"], "G distance").astype(np.float64)
            replayed = np.asarray([float(np.add.accumulate(trace, dtype=np.float32)[-1]) for trace in np.asarray(archive["initial_reward_traces"], dtype=np.float32)], dtype=np.float64)
            if not np.array_equal(native, replayed):
                errors.append("v1.4 J does not reproduce from float32 reward traces")
            expected_distance = np.linalg.norm(np.asarray(archive["initial_final_states"], dtype=np.float64) - np.asarray(archive["goal_state"], dtype=np.float64)[None, :], axis=1)
            if not np.allclose(distance, expected_distance, atol=1e-12, rtol=0):
                errors.append("v1.4 G distance does not reproduce from final states and goal")
            # Controls are reconstructed from retained NPZ primitives.  The
            # expert segment ends at the retained goal by protocol, so its G
            # baseline is exactly zero; noop G comes from its retained state.
            baselines = artifact.get("baselines")
            expert_trace = np.asarray(archive["expert_reward_trace"], dtype=np.float32)
            noop_trace = np.asarray(archive["hold_position_noop_reward_trace"], dtype=np.float32)
            expert_j = float(np.add.accumulate(expert_trace, dtype=np.float32)[-1])
            noop_j = float(np.add.accumulate(noop_trace, dtype=np.float32)[-1])
            noop_distance = float(np.linalg.norm(
                np.asarray(archive["hold_position_noop_final_state"], dtype=np.float64)
                - np.asarray(archive["goal_state"], dtype=np.float64)
            ))
            controls = {
                "expert_goal_defining_segment": {"J": expert_j, "G": -0.0, "native_return": expert_j, "native_state_distance": 0.0},
                "hold_position_noop": {"J": noop_j, "G": -noop_distance, "native_return": noop_j, "native_state_distance": noop_distance},
            }
            if not isinstance(baselines, Mapping):
                errors.append("v1.4 baselines are missing")
            else:
                for name, expected in controls.items():
                    actual = baselines.get(name)
                    if not isinstance(actual, Mapping):
                        errors.append(f"v1.4 {name} baseline is missing")
                        continue
                    for field, value in expected.items():
                        try:
                            if not np.isclose(float(actual.get(field)), value, atol=1e-12, rtol=0):
                                errors.append(f"v1.4 {name} {field} does not reproduce from NPZ")
                        except (TypeError, ValueError):
                            errors.append(f"v1.4 {name} {field} is invalid")
                noop = baselines.get("hold_position_noop")
                if isinstance(noop, Mapping) and noop.get("physical_action_sha256") != array_sha256(archive["hold_position_noop_physical_actions"]):
                    errors.append("v1.4 hold_position_noop physical action hash does not reproduce from NPZ")
            expected_goal_state_hash = array_sha256(archive["goal_state"])
            expected_initial_observation_hash = canonical_sha256({
                "visual": array_sha256(archive["raw_initial_visual"]),
                "proprio": array_sha256(archive["raw_initial_proprio"]),
            })
            expected_goal_observation_hash = canonical_sha256({
                "visual": array_sha256(archive["raw_goal_visual"]),
                "proprio": array_sha256(archive["raw_goal_proprio"]),
            })
            for key, expected in (
                ("goal_state_sha256", expected_goal_state_hash),
                ("initial_observation_sha256", expected_initial_observation_hash),
                ("goal_observation_sha256", expected_goal_observation_hash),
            ):
                if artifact.get(key) != expected:
                    errors.append(f"v1.4 {key} does not reproduce from retained NPZ primitives")
            if not np.allclose(p, captured, atol=V14_SCORE_TOLERANCE, rtol=0):
                errors.append("v1.4 captured scores do not reproduce P")
            raw_visual = np.asarray(archive["initial_raw_visual_observations"])
            raw_proprio = np.asarray(archive["initial_raw_proprio_observations"])
            for index in range(300):
                expected_visual_digest = np.frombuffer(bytes.fromhex(array_sha256(raw_visual[index])), dtype=np.uint8)
                expected_proprio_digest = np.frombuffer(bytes.fromhex(array_sha256(raw_proprio[index])), dtype=np.uint8)
                if not np.array_equal(archive["initial_raw_visual_digests"][index], expected_visual_digest):
                    errors.append(f"v1.4 proposal {index} raw visual digest differs from retained observation")
                if not np.array_equal(archive["initial_raw_proprio_digests"][index], expected_proprio_digest):
                    errors.append(f"v1.4 proposal {index} raw proprio digest differs from retained observation")
            summary = p_o_g_j_summary(p, o, native, distance)
            ids = [f"proposal_{index:04d}" for index in range(300)]
            decision = artifact.get("decision", {})
            for field in ("predicted_choice", "encoded_real_choice", "native_reward_choice", "goal_aligned_choice"):
                expected_index = summary[field]
                if decision.get(field) != (ids[expected_index] if expected_index is not None else None):
                    errors.append(f"v1.4 decision {field} does not match sidecar")
            for field in ("predicted_tie_candidate_ids", "encoded_real_tie_candidate_ids"):
                expected = [ids[index] for index in summary["predicted_tie_indices" if field.startswith("predicted") else "encoded_real_tie_indices"]]
                if decision.get(field) != expected:
                    errors.append(f"v1.4 decision {field} does not match exact tie set")
            candidate_rows = artifact.get("candidate_rows")
            if not isinstance(candidate_rows, list) or len(candidate_rows) != 300:
                errors.append("v1.4 artifact must contain exactly 300 candidate rows")
            else:
                for index, row in enumerate(candidate_rows):
                    if not isinstance(row, Mapping) or row.get("candidate_id") != ids[index]:
                        errors.append(f"v1.4 proposal {index} has a non-canonical ID")
                        continue
                    if not np.array_equal(np.asarray(row.get("action_model"), dtype=np.float32), archive["initial_actions_normalized"][index]):
                        errors.append(f"v1.4 proposal {index} action differs from NPZ")
                    for field, expected in (("predicted_score", p[index]), ("encoded_real_score", o[index]), ("source_planner_score", captured[index]), ("native_return", native[index]), ("native_state_distance", distance[index]), ("J", native[index]), ("G", -distance[index])):
                        try:
                            if not np.isclose(float(row.get(field)), float(expected), atol=1e-12, rtol=0):
                                errors.append(f"v1.4 proposal {index} {field} differs from NPZ")
                        except (TypeError, ValueError):
                            errors.append(f"v1.4 proposal {index} {field} is invalid")
                    if row.get("normalized_action_sha256") != array_sha256(archive["initial_actions_normalized"][index]) or row.get("physical_action_sha256") != array_sha256(archive["initial_physical_actions"][index]):
                        errors.append(f"v1.4 proposal {index} action hash differs from NPZ")
                    if not np.array_equal(np.asarray(row.get("final_state"), dtype=np.float64), archive["initial_final_states"][index]):
                        errors.append(f"v1.4 proposal {index} terminal state differs from NPZ")
            gate = artifact.get("v14_reencoding_diagnostic", artifact.get("raw_evidence_reencoding_gate"))
            if not isinstance(gate, Mapping) or gate.get("passed") is not True:
                errors.append("v1.4 re-encoding decision gate did not pass")
            else:
                try:
                    recomputed_gate = make_reencoding_diagnostic(
                        production_predicted_visual=archive["initial_predicted_visual"],
                        production_predicted_proprio=archive["initial_predicted_proprio"],
                        production_realized_visual=archive["initial_encoded_real_visual"],
                        production_realized_proprio=archive["initial_encoded_real_proprio"],
                        production_goal_visual=archive["goal_visual"],
                        production_goal_proprio=archive["goal_proprio"],
                        chunk8_realized_visual=archive["reencoded_chunk8_realized_visual"],
                        chunk8_realized_proprio=archive["reencoded_chunk8_realized_proprio"],
                        chunk8_goal_visual=archive["reencoded_chunk8_goal_visual"],
                        chunk8_goal_proprio=archive["reencoded_chunk8_goal_proprio"],
                        chunk4_realized_visual=archive["reencoded_chunk4_realized_visual"],
                        chunk4_realized_proprio=archive["reencoded_chunk4_realized_proprio"],
                        chunk4_goal_visual=archive["reencoded_chunk4_goal_visual"],
                        chunk4_goal_proprio=archive["reencoded_chunk4_goal_proprio"],
                        production_p=p, production_o=o,
                        production_initial_visual=archive["initial_production_visual"],
                        production_initial_proprio=archive["initial_production_proprio"],
                        chunk8_initial_visual=archive["reencoded_chunk8_initial_visual"],
                        chunk8_initial_proprio=archive["reencoded_chunk8_initial_proprio"],
                        chunk4_initial_visual=archive["reencoded_chunk4_initial_visual"],
                        chunk4_initial_proprio=archive["reencoded_chunk4_initial_proprio"],
                    )
                    for field in (
                        "passed", "production_pass_authoritative", "initial_reencoding_present", "production_chunk_size", "cross_check_chunk_size",
                        "chunk8_anchor", "chunk4_anchor", "P_score_max_abs_difference",
                        "O_score_max_abs_difference", "cross_chunk_P_score_max_abs_difference",
                        "cross_chunk_O_score_max_abs_difference", "O_selector_agrees",
                        "cross_chunk_O_selector_agrees", "P_selector_agrees",
                        "cross_chunk_P_selector_agrees", "production_P_tie_indices",
                        "production_O_tie_indices", "chunk8_P_tie_indices", "chunk8_O_tie_indices",
                        "chunk4_P_tie_indices", "chunk4_O_tie_indices", "latent_coordinate_drift",
                        "component_score_drift", "production_P_tie_candidate_ids",
                        "chunk8_P_tie_candidate_ids", "chunk4_P_tie_candidate_ids",
                        "production_O_tie_candidate_ids", "chunk8_O_tie_candidate_ids",
                        "chunk4_O_tie_candidate_ids",
                    ):
                        if field == "latent_coordinate_drift" and isinstance(gate.get(field), Mapping) and isinstance(recomputed_gate.get(field), Mapping):
                            drift_mismatch = any(
                                canonical_sha256(gate[field].get(section, {}).get(key))
                                != canonical_sha256(recomputed_gate[field].get(section, {}).get(key))
                                for section in ("chunk8", "chunk4")
                                for key in ("terminal_visual", "terminal_proprio", "goal_visual", "goal_proprio", "initial_visual", "initial_proprio")
                            )
                        else:
                            drift_mismatch = canonical_sha256(gate.get(field)) != canonical_sha256(recomputed_gate.get(field))
                        if drift_mismatch:
                            errors.append(f"v1.4 re-encoding diagnostic {field} does not recompute from NPZ")
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    errors.append(f"v1.4 re-encoding diagnostic cannot be recomputed: {exc}")
                if gate.get("production_pass_authoritative") is not True or gate.get("chunk8_anchor") != "production_pass" or gate.get("chunk4_anchor") != "production_pass":
                    errors.append("v1.4 re-encoding checks are not anchored to production pass")
                for field in ("P_score_max_abs_difference", "O_score_max_abs_difference", "cross_chunk_P_score_max_abs_difference", "cross_chunk_O_score_max_abs_difference"):
                    if not _finite(gate.get(field), field).size or float(gate[field]) > V14_SCORE_TOLERANCE:
                        errors.append(f"v1.4 {field} exceeds score tolerance")
                expected_o_ties = [summary["encoded_real_tie_indices"]]
                if gate.get("production_O_tie_indices") != expected_o_ties[0] or gate.get("chunk8_O_tie_indices") != expected_o_ties[0] or gate.get("chunk4_O_tie_indices") != expected_o_ties[0]:
                    errors.append("v1.4 O tied-candidate-ID set gate does not match authoritative O")
                production_p_ties = tie_indices(p, lower_better=True)
                if gate.get("production_P_tie_indices") != production_p_ties or gate.get("chunk8_P_tie_indices") != production_p_ties or gate.get("chunk4_P_tie_indices") != production_p_ties:
                    errors.append("v1.4 P tied-candidate-ID set gate does not match authoritative P")
                expected_p_ids = [f"proposal_{index:04d}" for index in production_p_ties]
                expected_o_ids = [f"proposal_{index:04d}" for index in summary["encoded_real_tie_indices"]]
                for key in ("production_P_tie_candidate_ids", "chunk8_P_tie_candidate_ids", "chunk4_P_tie_candidate_ids"):
                    if gate.get(key) != expected_p_ids:
                        errors.append("v1.4 P tied candidate-ID set is not exact")
                for key in ("production_O_tie_candidate_ids", "chunk8_O_tie_candidate_ids", "chunk4_O_tie_candidate_ids"):
                    if gate.get(key) != expected_o_ids:
                        errors.append("v1.4 O tied candidate-ID set is not exact")
                drift = gate.get("latent_coordinate_drift")
                if not isinstance(drift, Mapping):
                    errors.append("v1.4 coordinate drift diagnostic is missing")
                else:
                    for section in ("chunk8", "chunk4"):
                        section_values = drift.get(section)
                        if not isinstance(section_values, Mapping):
                            errors.append("v1.4 coordinate drift diagnostic is non-finite")
                            continue
                        for field in ("terminal_visual", "terminal_proprio", "goal_visual", "goal_proprio", "initial_visual", "initial_proprio"):
                            stats = section_values.get(field)
                            if not isinstance(stats, Mapping) or any(not math.isfinite(float(stats.get(key, math.nan))) for key in ("max_abs", "rms", "median_abs", "p95_abs")):
                                errors.append("v1.4 coordinate drift diagnostic is non-finite")
            expected_metrics = standard_outer_metrics(
                p, o, native, distance, normalized_actions=archive["initial_actions_normalized"],
                predicted_latents=archive["initial_predicted_visual"], realized_latents=archive["initial_encoded_real_visual"],
                predicted_proprio=archive["initial_predicted_proprio"], realized_proprio=archive["initial_encoded_real_proprio"],
            )
            if canonical_sha256(artifact.get("standard_outer_metrics", {})) != canonical_sha256(expected_metrics):
                errors.append("v1.4 standard outer metrics do not match sidecar")
            uniform = artifact.get("baselines", {}).get("random") if isinstance(artifact.get("baselines"), Mapping) else None
            expected_uniform = {"selection": "exact uniform expectation", "candidate_id": None, "candidate_index": None, "J": float(np.mean(native)), "G": float(np.mean(-distance))}
            if uniform != expected_uniform:
                errors.append("v1.4 exact uniform baseline does not match NPZ")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        errors.append(f"v1.4 sidecar recomputation failed: {exc}")
    return sorted(set(errors))


def compare_v14_repeats(records: Sequence[Mapping[str, Any]], *, tolerance: float = V14_SCORE_TOLERANCE) -> list[str]:
    """Compare fresh repeats to repeat 0 with exact primitives and bounded scores."""
    if len(records) != 3:
        return ["v1.4 repeat comparison requires exactly three fresh records"]
    errors: list[str] = []
    arrays: list[dict[str, np.ndarray]] = []
    for index, record in enumerate(records):
        candidate = record.get("arrays") if isinstance(record.get("arrays"), Mapping) else None
        if candidate is None and isinstance(record.get("arrays_path"), (str, Path)):
            try:
                with np.load(record["arrays_path"], allow_pickle=False) as archive:
                    candidate = {key: np.asarray(archive[key]) for key in archive.files}
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"repeat {index} arrays unreadable: {exc}")
        if candidate is None:
            errors.append(f"repeat {index} lacks arrays")
        else:
            arrays.append({key: np.asarray(value) for key, value in candidate.items()})
    if errors:
        return sorted(set(errors))
    first = arrays[0]
    for index, current in enumerate(arrays[1:], 1):
        for key in V14_CROSS_MODEL_ARRAY_KEYS:
            if key not in first or key not in current or not np.array_equal(first[key], current[key]):
                errors.append(f"repeat {index} {key} differs exactly from repeat 0")
        for key in ("initial_objective_scores", "initial_predicted_visual_mse", "initial_observed_visual_mse", "initial_predicted_proprio_mse", "initial_observed_proprio_mse"):
            if key not in first or key not in current:
                errors.append(f"repeat {index} lacks score component {key}")
    if not errors:
        stack_keys = (
            "initial_objective_scores", "initial_predicted_visual_mse", "initial_observed_visual_mse",
            "initial_predicted_proprio_mse", "initial_observed_proprio_mse",
        )
        for key in stack_keys:
            stack = np.stack([current[key].astype(np.float64) for current in arrays])
            elementwise_range = np.max(stack, axis=0) - np.min(stack, axis=0)
            if not np.isfinite(stack).all() or float(np.max(elementwise_range)) > float(tolerance):
                errors.append(f"repeat score component {key} range exceeds tolerance")
        # O total and component vectors are recomputed from the authoritative
        # production arrays, not from a JSON score table.
        predicted_scores = []
        observed_scores = []
        for current in arrays:
            decomp = validate_p_o_arrays(
                current["initial_predicted_visual"], current["initial_encoded_real_visual"], current["goal_visual"],
                predicted_proprio=current["initial_predicted_proprio"], realized_proprio=current["initial_encoded_real_proprio"], goal_proprio=current["goal_proprio"],
            )
            predicted_scores.append(decomp["P"])
            observed_scores.append(decomp["O"])
        p_stack = np.stack(predicted_scores).astype(np.float64)
        if not np.isfinite(p_stack).all() or float(np.max(np.max(p_stack, axis=0) - np.min(p_stack, axis=0))) > float(tolerance):
            errors.append("repeat recomputed P score vector range exceeds tolerance")
        o_stack = np.stack(observed_scores).astype(np.float64)
        if not np.isfinite(o_stack).all() or float(np.max(np.max(o_stack, axis=0) - np.min(o_stack, axis=0))) > float(tolerance):
            errors.append("repeat observed O score vector range exceeds tolerance")
    decisions = [record.get("artifact", record).get("decision", {}) for record in records]
    for field in ("predicted_tie_candidate_ids", "encoded_real_tie_candidate_ids", "predicted_choice", "encoded_real_choice"):
        if any(decision.get(field) != decisions[0].get(field) for decision in decisions[1:]):
            errors.append(f"repeat {field} differs from repeat 0")
    binding_fields = (
        "protocol_sha256", "source_bundle_sha256", "checkpoint_sha256", "config_sha256",
        "official_git_commit", "official_root", "dataset_root", "dataset_split",
        "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256",
        "states_segment_sha256", "actions_segment_sha256", "environment_seed",
        "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256",
        "goal_observation_sha256", "dinov2", "runtime", "hardware", "runtime_sha256", "hardware_sha256",
    )
    binding_records = [record.get("artifact", record) for record in records]
    for field in binding_fields:
        if any(item.get(field) != binding_records[0].get(field) for item in binding_records[1:]):
            errors.append(f"repeat binding {field} differs from repeat 0")
    for field in ("production_pass_authoritative", "v14_reencoding_diagnostic"):
        if field == "production_pass_authoritative" and any(record.get("artifact", record).get(field) is not True for record in records):
            errors.append("production-pass authority is not asserted for every repeat")
    for index, record in enumerate(records):
        diagnostic = record.get("artifact", record).get("v14_reencoding_diagnostic", {})
        drift = diagnostic.get("latent_coordinate_drift") if isinstance(diagnostic, Mapping) else None
        if not isinstance(drift, Mapping):
            errors.append(f"repeat {index} lacks latent coordinate drift diagnostics")
            continue
        for section in ("chunk8", "chunk4"):
            values = drift.get(section) if isinstance(drift.get(section), Mapping) else {}
            for key in ("terminal_visual", "terminal_proprio", "goal_visual", "goal_proprio", "initial_visual", "initial_proprio"):
                try:
                    stats = values.get(key) if isinstance(values, Mapping) else None
                    if not isinstance(stats, Mapping) or any(not np.isfinite(float(stats.get(statistic))) for statistic in ("max_abs", "rms", "median_abs", "p95_abs")):
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"repeat {index} coordinate drift {section}.{key} is non-finite")
    return sorted(set(errors))


def make_v14_manifest(dataset_root: Path, *, seeds: tuple[int, ...] = V14_LOCKED_SEEDS) -> dict[str, Any]:
    if tuple(seeds) != V14_LOCKED_SEEDS:
        raise ValueError(f"v1.4 fixes menu seeds={V14_LOCKED_SEEDS}")
    base = _v12.make_v12_manifest(dataset_root, seeds=V14_LOCKED_SEEDS)
    entries = []
    for entry in base["entries"]:
        copied = dict(entry)
        copied["decision_id"] = str(copied["decision_id"]).replace("v12_", "v14_", 1)
        entries.append(copied)
    manifest = dict(base)
    manifest.update({
        "schema_version": V14_SCHEMA_VERSION, "protocol": V14_PROTOCOL,
        "protocol_sha256": v14_protocol_sha256(), "source_bundle": v14_source_bundle(),
        "source_bundle_sha256": v14_source_bundle_sha256(), "entries": entries,
    })
    return manifest


def validate_v14_manifest(manifest: Mapping[str, Any], *, require_full: bool = True) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != V14_SCHEMA_VERSION:
        errors.append("manifest is not PWA-PushT-v1.4")
    if manifest.get("scientific_result") is not False:
        errors.append("v1.4 manifest must remain engineering-only")
    if manifest.get("protocol") != V14_PROTOCOL or manifest.get("protocol_sha256") != v14_protocol_sha256():
        errors.append("manifest protocol/hash is not the v1.4 protocol")
    if manifest.get("source_bundle_sha256") != v14_source_bundle_sha256():
        errors.append("manifest source bundle hash mismatch")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("inventory") != "common_initial_only":
        errors.append("manifest inventory is not common_initial_only")
    elif any(selection.get(key) != expected for key, expected in {
        "outcome_blind": True, "menu_seeds": list(V14_LOCKED_SEEDS), "frameskip": 5,
        "horizon": 6, "segment_length": 31, "offset_count_per_trajectory": 5,
        "excluded_trajectories": [V14_EXCLUDED_TRAJECTORY], "models": list(V14_LOCKED_MODELS),
        "dataset_split": "val",
    }.items()):
        errors.append("v1.4 manifest selection is not locked")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return errors + ["v1.4 manifest entries are not a list"]
    keys, trajectories = [], []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"entry {index} is not an object")
            continue
        required = {"decision_id", "trajectory_id", "model", "episode", "offset", "seed", "expert_phase_proxy", "expert_first_block_motion_index", "inventory", "outcome_blind"}
        if not required.issubset(entry):
            errors.append(f"entry {index} is missing v1.4 schedule fields")
            continue
        if not str(entry.get("decision_id", "")).startswith("v14_"):
            errors.append(f"entry {index} is not a v1.4 decision")
        if entry.get("episode") == V14_EXCLUDED_TRAJECTORY:
            errors.append("v1.4 manifest includes excluded trajectory 7")
        if entry.get("model") not in V14_LOCKED_MODELS or entry.get("seed") not in V14_LOCKED_SEEDS:
            errors.append(f"entry {index} has unsupported model/menu seed")
        if entry.get("inventory") != "common_initial" or entry.get("outcome_blind") is not True:
            errors.append(f"entry {index} is not outcome-blind common-initial")
        keys.append((entry.get("trajectory_id"), entry.get("offset"), entry.get("seed"), entry.get("model")))
        trajectories.append(entry.get("trajectory_id"))
    if len(keys) != len(set(keys)):
        errors.append("v1.4 manifest contains duplicate segment/model rows")
    if manifest.get("decision_count") != len(entries):
        errors.append("v1.4 decision_count mismatch")
    if require_full and (len(entries) != 600 or len(set(trajectories)) != 20):
        errors.append("v1.4 requires exactly 600 decisions across 20 trajectories")
    if require_full and isinstance(manifest.get("dataset_root"), str):
        try:
            expected = make_v14_manifest(Path(manifest["dataset_root"]))
            if manifest.get("entries") != expected.get("entries"):
                errors.append("v1.4 manifest entries do not match the canonical regenerated schedule")
            for key in ("selection", "trajectory_count", "segment_count", "decision_count", "dataset_root", "dataset_split"):
                if manifest.get(key) != expected.get(key):
                    errors.append(f"v1.4 manifest {key} does not match canonical regeneration")
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            errors.append(f"v1.4 canonical schedule cannot be regenerated: {exc}")
    return sorted(set(errors))


__all__ = [
    "V14_SCHEMA_VERSION", "V14_ARTIFACT_SCHEMA_VERSION", "V14_ANALYSIS_VERSION",
    "V14_CERTIFICATE_SCHEMA_VERSION", "V14_BUNDLE_SCHEMA_VERSION", "V14_SUPERVISOR_VERSION",
    "V14_SOURCE_BUNDLE_VERSION", "V14_CERTIFICATE_ISSUANCE_SCHEMA_VERSION",
    "V14_CERTIFICATE_ISSUANCE_RELATIVE_PATH", "V14_LOCKED_MODELS", "V14_LOCKED_SEEDS",
    "V14_EXCLUDED_TRAJECTORY", "V14_PROTOCOL", "V14_ARRAY_KEYS", "V14_CROSS_MODEL_ARRAY_KEYS",
    "canonical_sha256", "array_sha256", "file_sha256", "v14_protocol_sha256",
    "v14_source_bundle", "v14_source_bundle_sha256", "validate_v14_args", "capture_initial_only_cem",
    "expand_model_actions", "validate_p_o_arrays", "proposal_support_diagnostics",
    "standard_outer_metrics", "tie_indices", "strict_choice", "p_o_g_j_summary",
    "make_reencoding_diagnostic", "build_v14_artifact", "validate_v14_sidecar",
    "validate_v14_artifact", "compare_v14_repeats", "make_v14_manifest", "validate_v14_manifest",
]
