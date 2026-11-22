"""Sealed trajectory-clustered analysis for PWA-PushT v1.4.

The analyzer is deliberately a one-shot consumer of the complete 600-row
manifest. Primary quantities are recomputed from each NPZ sidecar; JSON
decision values are never an analysis input.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.official_pusht_v14_batch_study import (
    LEDGER_NAME, V14_STUDY_ISSUANCE_PATH, V14_STUDY_LEDGER_PATH, _certified_menu_hashes, _row_capability_path,
    _row_capability_state, completed_receipt_errors,
    failed_receipt_errors, paired_v14_inventory_errors, read_v14_ledger,
    validate_v14_study_issuance,
)
from experiments.official_pusht_v14_certificate_supervisor import V14_CERTIFICATE_ISSUANCE_PATH, validate_v14_bundle
from experiments.official_pusht_v14_common_initial import (
    V14_ANALYSIS_VERSION, V14_LOCKED_MODELS, V14_PROTOCOL, V14_SCHEMA_VERSION,
    _dataset_segment_hashes, array_sha256, canonical_sha256, file_sha256, make_v14_manifest, p_o_g_j_summary,
    standard_outer_metrics, validate_p_o_arrays, validate_v14_artifact,
    validate_v14_manifest, v14_protocol_sha256, v14_source_bundle_sha256,
)

BOOTSTRAP_REPLICATES = 10_000
SUPPORT_MIN_CANDIDATES = 200
MIN_VALID_ROWS = 540
MIN_ELIGIBLE_TRAJECTORIES = 18
MIN_MATCHED_CELLS_PER_TRAJECTORY = 12
FIXED_CELLS_PER_TRAJECTORY = 15

# Kept as a descriptive-only compatibility entry point for callers that want
# row-level summaries. The sealed analysis below never consumes these values
# for a primary endpoint; it recomputes metrics from NPZ first.
def aggregate_standard_metrics(artifacts: list[Mapping[str, Any]]) -> dict[str, Any]:
    paths = (
        ("score_error_mse",), ("absolute_P_minus_O", "mean"), ("absolute_P_minus_O", "median"),
        ("latent_prediction_mse", "visual"), ("latent_prediction_mse", "proprio"),
        ("latent_prediction_mse", "weighted_total"),
        ("spearman", "P_vs_J"), ("spearman", "P_vs_G"), ("spearman", "O_vs_J"), ("spearman", "O_vs_G"),
        ("pairwise_ranking_accuracy", "P_vs_J"), ("pairwise_ranking_accuracy", "P_vs_G"),
        ("pairwise_ranking_accuracy", "O_vs_J"), ("pairwise_ranking_accuracy", "O_vs_G"),
        ("normalized_regret_at_1", "P_J"), ("normalized_regret_at_1", "P_G"),
        ("normalized_regret_at_1", "O_J"), ("normalized_regret_at_1", "O_G"),
        ("uniform_expectation", "J"), ("uniform_expectation", "G"),
    )
    result: dict[str, Any] = {}
    for path in paths:
        values = []
        for artifact in artifacts:
            current: Any = artifact.get("standard_outer_metrics", {})
            for key in path:
                current = current.get(key) if isinstance(current, Mapping) else None
            if _finite(current):
                values.append(float(current))
        result[".".join(path)] = {"mean": float(np.mean(values)) if values else None, "count": len(values)}
    for k in ("1", "5", "10"):
        for outcome in ("J", "G"):
            values = []
            for artifact in artifacts:
                metric = artifact.get("standard_outer_metrics", {}).get("top_k_normalized_regret", {}).get(k, {})
                if isinstance(metric, Mapping) and _finite(metric.get(outcome)):
                    values.append(float(metric[outcome]))
            result[f"top_{k}_normalized_regret.{outcome}"] = {"mean": float(np.mean(values)) if values else None, "count": len(values)}
    subset_values: dict[str, list[float]] = {}
    for artifact in artifacts:
        subset = artifact.get("standard_outer_metrics", {}).get("action_only_subset")
        if not isinstance(subset, Mapping) or subset.get("status") == "insufficient_support":
            continue
        for key, value in subset.items():
            if key in {"proposal_count", "subset_count", "definition", "support_mask_sha256", "action_only_subset"}:
                continue
            if isinstance(value, Mapping):
                for subkey, subvalue in value.items():
                    if _finite(subvalue):
                        subset_values.setdefault(f"{key}.{subkey}", []).append(float(subvalue))
            elif _finite(value):
                subset_values.setdefault(key, []).append(float(value))
    result["magnitude_bounded_standard_metric_means"] = {key: {"mean": float(np.mean(values)), "count": len(values)} for key, values in sorted(subset_values.items())}
    return result




def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _sign_flip(values: list[float]) -> float | None:
    data = np.asarray([value for value in values if _finite(value)], dtype=np.float64)
    if not len(data):
        return None
    observed, total, exceed = abs(float(np.mean(data))), 1 << len(data), 0
    bits = np.arange(len(data), dtype=np.uint64)
    for start in range(0, total, 65_536):
        masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bits) & 1).astype(np.float64)
        exceed += int(np.count_nonzero(np.abs((signs @ data) / len(data)) >= observed - 1e-15))
    return float(exceed / total)


def _bootstrap_ci(values: list[float], label: str) -> list[float] | None:
    data = np.asarray([value for value in values if _finite(value)], dtype=np.float64)
    if not len(data):
        return None
    seed = int(hashlib.sha256(f"PWA-PushT-v1.4:{label}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(BOOTSTRAP_REPLICATES, len(data)))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _stats(values: list[float], label: str) -> dict[str, Any]:
    finite = [float(value) for value in values if _finite(value)]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "median": float(np.median(finite)) if finite else None,
        "trajectory_count": len(finite),
        "trajectory_bootstrap_95_ci": _bootstrap_ci(finite, label),
        "two_sided_exact_sign_flip_p": _sign_flip(finite),
        "trajectory_values": finite,
    }


def _holm(p_values: Mapping[str, Any]) -> dict[str, float | None]:
    valid = sorted((float(value), key) for key, value in p_values.items() if _finite(value))
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    for rank, (value, key) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        adjusted[key] = running
    return adjusted


def _zero_row(entry: Mapping[str, Any], reason: str) -> dict[str, Any]:
    base = {key: entry.get(key) for key in ("decision_id", "trajectory_id", "model", "episode", "offset", "seed", "expert_phase_proxy")}
    row = {**base, "raw_valid": False, "support_valid": False, "usefulness_valid": False,
           "standard_outer_metrics": {}, "consequential_pair_decomposition": {},
           "mechanism_by_encoding": {}, "descriptive_controls": {}, "recomputed_baselines": {}, "invalid_reason": reason}
    for mode in ("raw", "support"):
        for label in ("Delta_J", "Delta_G", "R_P_J", "R_O_J", "R_P_G", "R_O_G"):
            row[f"{mode}_{label}"] = 0.0
    row.update({"usefulness_Delta_J": 0.0, "usefulness_Delta_G": 0.0})
    return row


def _retained_npz_bindings(arrays_path: Path) -> dict[str, str]:
    """Derive row self-bindings from this row's NPZ, not a repeat certificate."""
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            return {
                "goal_state_sha256": array_sha256(archive["goal_state"]),
                "initial_observation_sha256": canonical_sha256({
                    "visual": array_sha256(archive["raw_initial_visual"]),
                    "proprio": array_sha256(archive["raw_initial_proprio"]),
                }),
                "goal_observation_sha256": canonical_sha256({
                    "visual": array_sha256(archive["raw_goal_visual"]),
                    "proprio": array_sha256(archive["raw_goal_proprio"]),
                }),
            }
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def _pair_map(p: np.ndarray, o: np.ndarray, decomposition: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    direction = np.asarray(decomposition.get("directional_correction"), dtype=np.float64)
    magnitude = np.asarray(decomposition.get("magnitude_correction"), dtype=np.float64)
    selected = summary.get("predicted_choice")
    result: dict[str, Any] = {}
    for name, other_key, tie_key in (("P_vs_O_choice", "encoded_real_choice", "encoded_real_tie_indices"),
                                     ("P_vs_finite_menu_best_G", "goal_aligned_choice", "goal_aligned_tie_indices"),
                                     ("P_vs_finite_menu_best_J", "native_reward_choice", "native_reward_tie_indices")):
        other = summary.get(other_key)
        tie_indices = summary.get(tie_key, [])
        tie_indices = list(tie_indices) if isinstance(tie_indices, (list, tuple)) else []
        finite_menu_tie = name != "P_vs_O_choice" and len(tie_indices) > 1
        abstention = None
        if selected is None:
            abstention = "predicted_choice_tie"
        elif other is None:
            abstention = "counterfactual_choice_tie"
        elif finite_menu_tie:
            # The primary regret remains defined by the finite-menu maximum or
            # minimum.  Mechanism attribution abstains because first-index
            # tie-breaking must never create a directional explanation.
            abstention = "finite_menu_tie_set"
        pair: dict[str, Any] = {
            "predicted_choice_index": int(selected) if selected is not None else None,
            "counterfactual_choice_index": int(other) if other is not None and not finite_menu_tie else None,
            "counterfactual_tie_indices": [int(index) for index in tie_indices],
            "selection_disagreement": bool(selected is not None and other is not None and selected != other and not finite_menu_tie),
            "same_choice": bool(selected is not None and other is not None and selected == other and not finite_menu_tie),
            "latent_ranking_reversal": False,
            "mechanism_only_abstention": abstention,
            "predicted_margin": None, "realized_latent_margin": None,
            "P_margin_selected_minus_counterfactual": None,
            "O_margin_selected_minus_counterfactual": None,
            "directional_margin": None, "magnitude_margin": None,
            "directional_vs_magnitude_dominance": "indeterminate",
            "identity_residual": None,
        }
        if selected is not None and other is not None and not finite_menu_tie:
            p_margin = float(p[selected] - p[other])
            o_margin = float(o[selected] - o[other])
            directional_margin = float(direction[selected] - direction[other])
            magnitude_margin = float(magnitude[selected] - magnitude[other])
            pair.update({
                "predicted_margin": p_margin,
                "realized_latent_margin": o_margin,
                "P_margin_selected_minus_counterfactual": p_margin,
                "O_margin_selected_minus_counterfactual": o_margin,
                "directional_margin": directional_margin,
                "magnitude_margin": magnitude_margin,
                "directional_vs_magnitude_dominance": (
                    "directional" if abs(directional_margin) > abs(magnitude_margin)
                    else "magnitude" if abs(magnitude_margin) > abs(directional_margin)
                    else "balanced"
                ),
                "identity_residual": float(
                    p_margin - o_margin - directional_margin - magnitude_margin
                ),
                "latent_ranking_reversal": bool(p_margin < 0 < o_margin),
            })
        result[name] = pair
    return result


def _mechanism_summary(p: np.ndarray, o: np.ndarray, native: np.ndarray, distance: np.ndarray) -> dict[str, Any]:
    """Add physical-oracle tie sets without changing primary regret maxima."""
    summary = dict(p_o_g_j_summary(p, o, native, distance))
    best_j, best_distance = float(np.max(native)), float(np.min(distance))
    summary["native_reward_tie_indices"] = [
        int(index) for index in np.flatnonzero(np.isclose(native, best_j, atol=1e-10, rtol=1e-8))
    ]
    summary["goal_aligned_tie_indices"] = [
        int(index) for index in np.flatnonzero(np.isclose(distance, best_distance, atol=1e-10, rtol=1e-8))
    ]
    return summary


def _coordinate_drift_from_archive(archive: Any) -> dict[str, Any]:
    def stats(left: Any, right: Any) -> dict[str, float]:
        absolute = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).reshape(-1)
        return {"max_abs": float(np.max(absolute)), "rms": float(np.sqrt(np.mean(np.square(absolute)))),
                "median_abs": float(np.median(absolute)), "p95_abs": float(np.quantile(absolute, 0.95))}
    result: dict[str, Any] = {"threshold": None, "interpretation": "descriptive coordinate drift; no coordinate threshold"}
    for encoding in ("chunk8", "chunk4"):
        result[encoding] = {
            "initial_visual": stats(archive["initial_production_visual"], archive[f"reencoded_{encoding}_initial_visual"]),
            "initial_proprio": stats(archive["initial_production_proprio"], archive[f"reencoded_{encoding}_initial_proprio"]),
            "terminal_visual": stats(archive["initial_encoded_real_visual"], archive[f"reencoded_{encoding}_realized_visual"]),
            "terminal_proprio": stats(archive["initial_encoded_real_proprio"], archive[f"reencoded_{encoding}_realized_proprio"]),
            "goal_visual": stats(archive["goal_visual"], archive[f"reencoded_{encoding}_goal_visual"]),
            "goal_proprio": stats(archive["goal_proprio"], archive[f"reencoded_{encoding}_goal_proprio"]),
        }
    return result


def _recomputed_row(artifact: Mapping[str, Any], entry: Mapping[str, Any], arrays_path: Path) -> dict[str, Any]:
    with np.load(arrays_path, allow_pickle=False) as archive:
        decomposition = validate_p_o_arrays(
            archive["initial_predicted_visual"], archive["initial_encoded_real_visual"], archive["goal_visual"],
            predicted_proprio=archive["initial_predicted_proprio"],
            realized_proprio=archive["initial_encoded_real_proprio"], goal_proprio=archive["goal_proprio"],
        )
        p, o = np.asarray(decomposition["P"], dtype=np.float64), np.asarray(decomposition["O"], dtype=np.float64)
        # Recompute physical outcomes from retained primitives as well; the
        # advertised J/G vectors are validation commitments, not inputs.
        native = np.asarray([
            float(np.add.accumulate(trace, dtype=np.float32)[-1])
            for trace in np.asarray(archive["initial_reward_traces"], dtype=np.float32)
        ], dtype=np.float64)
        distance = np.linalg.norm(
            np.asarray(archive["initial_final_states"], dtype=np.float64)
            - np.asarray(archive["goal_state"], dtype=np.float64)[None, :], axis=1,
        )
        summary = _mechanism_summary(p, o, native, distance)
        metrics = standard_outer_metrics(
            p, o, native, distance,
            normalized_actions=archive["initial_actions_normalized"],
            predicted_latents=archive["initial_predicted_visual"],
            realized_latents=archive["initial_encoded_real_visual"],
            predicted_proprio=archive["initial_predicted_proprio"],
            realized_proprio=archive["initial_encoded_real_proprio"],
        )
        encodings: dict[str, dict[str, Any]] = {}
        for name, suffix in (("production", ""), ("chunk8", "8"), ("chunk4", "4")):
            if not suffix:
                current = decomposition
            else:
                current = validate_p_o_arrays(
                    archive["initial_predicted_visual"], archive[f"reencoded_chunk{suffix}_realized_visual"],
                    archive[f"reencoded_chunk{suffix}_goal_visual"],
                    predicted_proprio=archive["initial_predicted_proprio"],
                    realized_proprio=archive[f"reencoded_chunk{suffix}_realized_proprio"],
                    goal_proprio=archive[f"reencoded_chunk{suffix}_goal_proprio"],
                )
            current_p = np.asarray(current["P"], dtype=np.float64)
            current_o = np.asarray(current["O"], dtype=np.float64)
            encodings[name] = _pair_map(current_p, current_o, current["decomposition"], _mechanism_summary(current_p, current_o, native, distance))
        subset = metrics.get("action_only_subset")
        subset_recovery = subset.get("matched_recovery", {}) if isinstance(subset, Mapping) else {}
        raw_valid = summary.get("predicted_choice") is not None and summary.get("encoded_real_choice") is not None
        support_valid = bool(isinstance(subset, Mapping) and int(subset.get("subset_count", 0)) >= SUPPORT_MIN_CANDIDATES and subset_recovery.get("abstained") is False)
        raw_values = {"Delta_J": summary.get("Delta_J"), "Delta_G": summary.get("Delta_G"), "R_P_J": summary.get("R_P_J"), "R_O_J": summary.get("R_O_J"), "R_P_G": summary.get("R_P_G"), "R_O_G": summary.get("R_O_G")}
        support_values = {label: subset_recovery.get(label) for label in ("Delta_J", "Delta_G", "R_P_J", "R_O_J", "R_P_G", "R_O_G")}
        p_index = summary.get("predicted_choice")
        o_index = summary.get("encoded_real_choice")
        uniform_j, uniform_g = float(np.mean(native)), float(np.mean(-distance))
        # Controls are reconstructed from the immutable NPZ primitives.  The
        # artifact's JSON baselines are intentionally never consulted: they
        # are a reporting surface and may be tampered with independently of
        # the retained replay evidence.
        expert_trace = np.asarray(archive["expert_reward_trace"], dtype=np.float32)
        noop_actions = np.asarray(archive["hold_position_noop_physical_actions"], dtype=np.float32)
        noop_state = np.asarray(archive["hold_position_noop_final_state"], dtype=np.float64)
        noop_trace = np.asarray(archive["hold_position_noop_reward_trace"], dtype=np.float32)
        expert_j = float(np.add.accumulate(expert_trace, dtype=np.float32)[-1])
        noop_j = float(np.add.accumulate(noop_trace, dtype=np.float32)[-1])
        noop_distance = float(np.linalg.norm(noop_state - np.asarray(archive["goal_state"], dtype=np.float64)))
        controls: dict[str, dict[str, Any]] = {
            "random": {"J": uniform_j, "G": uniform_g, "selection": "exact uniform expectation"},
            "expert_goal_defining_segment": {
                "J": expert_j, "G": -0.0, "native_return": expert_j,
                "native_state_distance": 0.0,
                "reward_trace_sha256": array_sha256(expert_trace),
                "expert_reward_trace_sha256": array_sha256(expert_trace),
                "goal_state_sha256": array_sha256(archive["goal_state"]),
            },
            "hold_position_noop": {
                "J": noop_j, "G": -noop_distance, "native_return": noop_j,
                "native_state_distance": noop_distance,
                "reward_trace_sha256": array_sha256(noop_trace),
                "noop_reward_trace_sha256": array_sha256(noop_trace),
                "physical_action_sha256": array_sha256(noop_actions),
                "noop_physical_actions_sha256": array_sha256(noop_actions),
                "final_state_sha256": array_sha256(noop_state),
                "noop_final_state_sha256": array_sha256(noop_state),
                "goal_state_sha256": array_sha256(archive["goal_state"]),
            },
        }
        row = {
            **{key: entry.get(key) for key in ("decision_id", "trajectory_id", "model", "episode", "offset", "seed", "expert_phase_proxy")},
            "raw_valid": raw_valid, "support_valid": support_valid, "usefulness_valid": raw_valid,
            "standard_outer_metrics": metrics, "consequential_pair_decomposition": encodings["production"],
            "mechanism_by_encoding": encodings, "descriptive_controls": controls,
            "recomputed_controls": controls,
            "coordinate_drift": _coordinate_drift_from_archive(archive),
            "recomputed_baselines": {
                "uniform_expectation": {"J": uniform_j, "G": uniform_g},
                "P": {"J": float(native[p_index]), "G": float(-distance[p_index])} if p_index is not None else {},
                "O": {"J": float(native[o_index]), "G": float(-distance[o_index])} if o_index is not None else {},
                "G_oracle": {"G": float(np.max(-distance))}, "J_oracle": {"J": float(np.max(native))},
            },
            "raw_recomputed_summary": summary, "support_recomputed_summary": subset_recovery,
        }
        for label, value in raw_values.items():
            row[f"raw_{label}"] = float(value) if raw_valid and _finite(value) else 0.0
        for label, value in support_values.items():
            row[f"support_{label}"] = float(value) if support_valid and _finite(value) else 0.0
        row["usefulness_Delta_J"] = float(native[p_index] - uniform_j) if raw_valid and p_index is not None else 0.0
        row["usefulness_Delta_G"] = float(-distance[p_index] - uniform_g) if raw_valid and p_index is not None else 0.0
        return row


def _read_rows(
    study_dir: Path, manifest: Mapping[str, Any], certificate_hashes: Mapping[str, str] | None = None,
    certificate_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    terminal_failed: set[str] | None = None, strict_completed: bool = False,
    certificate_issuance: Mapping[str, Any] | None = None,
    study_issuance: Mapping[str, Any] | None = None,
    certified_menu_hashes: Mapping[int, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, invalid = [], []
    failed = terminal_failed or set()
    for entry in manifest["entries"]:
        decision_id = str(entry["decision_id"])
        if decision_id in failed:
            rows.append(_zero_row(entry, "terminal_failed_attempt_preregistered_zero"))
            invalid.append({"decision_id": decision_id, "reason": "terminal_failed_attempt_preregistered_zero"})
            continue
        artifact_path, arrays_path = study_dir / f"{decision_id}.json", study_dir / f"{decision_id}.npz"
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            advertised_sidecar = artifact.get("arrays_sidecar")
            if not isinstance(advertised_sidecar, str) or Path(advertised_sidecar).resolve() != arrays_path.resolve():
                raise ValueError("v1.4 artifact arrays sidecar path is not the canonical manifest row sidecar")
            expected_certificate_hash = (certificate_hashes or {}).get(str(entry.get("model")))
            row_bindings = dict(
                (certificate_bindings or {}).get(decision_id)
                or (certificate_bindings or {}).get(str(entry.get("model")))
                or {}
            )
            # Goal/observation self-bindings are recomputed from the row's
            # NPZ.  They must never be borrowed from the excluded trajectory
            # supervisor signatures.
            row_bindings.update(_retained_npz_bindings(arrays_path))
            validation_kwargs: dict[str, Any] = {
                "arrays_path": arrays_path,
                "expected_bindings": row_bindings,
                "expected_certificate_issuance": certificate_issuance,
                "expected_certificate_sha256": expected_certificate_hash,
                "expected_menu_sha256": (certified_menu_hashes or {}).get(int(entry["seed"])),
            }
            expected_row_bindings = validation_kwargs.get("expected_bindings")
            if strict_completed:
                missing_bindings = _missing_full_bindings(expected_row_bindings if isinstance(expected_row_bindings, Mapping) else {})
                if missing_bindings:
                    raise ValueError("sealed v1.4 row lacks complete expected bindings: " + ", ".join(missing_bindings))
            # The study issuance key is intentionally forwarded when the
            # installed v1.4 validator exposes it.  This keeps the analyzer
            # compatible with the current validator while making the
            # output-independent study tombstone a first-class seal as soon
            # as that validator lands.
            validator_parameters = inspect.signature(validate_v14_artifact).parameters
            if study_issuance is not None and (
                "expected_study_issuance" in validator_parameters
                or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in validator_parameters.values())
            ):
                validation_kwargs["expected_study_issuance"] = study_issuance
            errors = validate_v14_artifact(artifact, entry, **validation_kwargs)
            if artifact.get("v1_2_evidence_reused") is not False:
                errors.append("v1.4 artifact must explicitly disclose no v1.2 evidence reuse")
            if expected_certificate_hash is not None:
                certificate_ref = artifact.get("reproducibility_certificate")
                if not isinstance(certificate_ref, Mapping) or certificate_ref.get("status") != "passed" or certificate_ref.get("sha256") != expected_certificate_hash:
                    errors.append("v1.4 artifact reproducibility certificate binding mismatch")
            if errors:
                raise ValueError("; ".join(errors))
            row = _recomputed_row(artifact, entry, arrays_path)
            rows.append(row)
            if not row["raw_valid"] or not row["support_valid"]:
                invalid.append({"decision_id": decision_id, "raw_valid": row["raw_valid"], "support_valid": row["support_valid"]})
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            if strict_completed:
                raise ValueError(f"completed v1.4 row {decision_id} is missing, corrupt, or invalid: {exc}") from exc
            rows.append(_zero_row(entry, f"invalid_artifact:{exc}"))
            invalid.append({"decision_id": decision_id, "reason": f"invalid_artifact:{exc}"})
    return rows, invalid


def _summarize(rows: list[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    labels = ("Delta_J", "Delta_G") if mode == "usefulness" else ("Delta_J", "Delta_G", "R_P_J", "R_O_J", "R_P_G", "R_O_G")
    index = {(str(row.get("trajectory_id")), int(row.get("offset", 0)), int(row.get("seed", 0)), str(row.get("model"))): row for row in rows}
    trajectories = sorted({str(row.get("trajectory_id")) for row in rows})
    clusters: dict[str, dict[str, float]] = {}
    model_clusters = {model: {label: [] for label in labels} for model in V14_LOCKED_MODELS}
    matched_counts: dict[str, int] = {}
    for trajectory in trajectories:
        cells = sorted({(int(row["offset"]), int(row["seed"])) for row in rows if str(row.get("trajectory_id")) == trajectory})
        accum = {model: {label: [] for label in labels} for model in V14_LOCKED_MODELS}
        matched = 0
        for offset, seed in cells:
            pair = [index.get((trajectory, offset, seed, model)) for model in V14_LOCKED_MODELS]
            valid = all(item is not None and bool(item.get(f"{mode}_valid")) for item in pair)
            matched += int(valid)
            for model, item in zip(V14_LOCKED_MODELS, pair):
                for label in labels:
                    value = item.get(f"{mode}_{label}", 0.0) if valid and item is not None else 0.0
                    accum[model][label].append(float(value) if _finite(value) else 0.0)
        for _ in range(max(0, FIXED_CELLS_PER_TRAJECTORY - len(cells))):
            for model in V14_LOCKED_MODELS:
                for label in labels:
                    accum[model][label].append(0.0)
        per_model = {model: {label: float(np.mean(values)) if values else 0.0 for label, values in accum[model].items()} for model in V14_LOCKED_MODELS}
        for model in V14_LOCKED_MODELS:
            for label in labels:
                model_clusters[model][label].append(per_model[model][label])
        clusters[trajectory] = {label: float(np.mean([per_model[model][label] for model in V14_LOCKED_MODELS])) for label in labels}
        matched_counts[trajectory] = matched
    result: dict[str, Any] = {
        "invalid_policy": "invalid, tied, failed, or unmatched cells contribute zero to both models under one matched mask",
        "matched_cell_counts": matched_counts, "trajectory_clusters": clusters,
        "model_specific": {model: {label: _stats(values, f"{mode}:{model}:{label}") for label, values in model_clusters[model].items()} for model in V14_LOCKED_MODELS},
    }
    for label in labels:
        result[label] = _stats([cluster[label] for cluster in clusters.values()], f"{mode}:pooled:{label}")
    result["holm_adjusted_p"] = _holm({label: result[label]["two_sided_exact_sign_flip_p"] for label in ("Delta_G", "Delta_J")})
    matched_rows = 2 * sum(matched_counts.values())
    eligible = sum(count >= MIN_MATCHED_CELLS_PER_TRAJECTORY for count in matched_counts.values())
    result["validity_gate"] = {
        "valid_rows": matched_rows, "required_valid_rows": MIN_VALID_ROWS,
        "eligible_trajectories": eligible, "required_eligible_trajectories": MIN_ELIGIBLE_TRAJECTORIES,
        "matched_cells_required_per_trajectory": MIN_MATCHED_CELLS_PER_TRAJECTORY,
        "fixed_cells_per_trajectory": FIXED_CELLS_PER_TRAJECTORY,
        "passed": matched_rows >= MIN_VALID_ROWS and eligible >= MIN_ELIGIBLE_TRAJECTORIES,
    }
    return result


def _positive_with_ci(summary: Mapping[str, Any], label: str) -> bool:
    stats = summary.get(label, {})
    ci = stats.get("trajectory_bootstrap_95_ci") if isinstance(stats, Mapping) else None
    return bool(_finite(stats.get("mean")) and float(stats["mean"]) > 0 and isinstance(ci, list) and ci[0] > 0)


def _matched_descriptive_metric_means(
    rows: list[Mapping[str, Any]],
    metric_rows: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    labels: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Aggregate S metrics using the locked cell, trajectory, and model units."""
    protocol_rows = [
        row for row in rows
        if all(key in row for key in ("trajectory_id", "offset", "seed", "model"))
    ]
    if not protocol_rows:
        return {}, {}, {}
    index = {
        (str(row["trajectory_id"]), int(row["offset"]), int(row["seed"]), str(row["model"])): row
        for row in protocol_rows
    }
    trajectories = sorted({str(row["trajectory_id"]) for row in protocol_rows})
    model_trajectory_values = {model: {label: [] for label in labels} for model in V14_LOCKED_MODELS}
    trajectory_clusters: dict[str, dict[str, float]] = {}
    for trajectory in trajectories:
        cells = sorted({
            (int(row["offset"]), int(row["seed"]))
            for row in protocol_rows if str(row["trajectory_id"]) == trajectory
        })
        accum = {model: {label: [] for label in labels} for model in V14_LOCKED_MODELS}
        for offset, seed in cells:
            pair = [index.get((trajectory, offset, seed, model)) for model in V14_LOCKED_MODELS]
            matched = all(row is not None and bool(row.get("support_valid")) for row in pair)
            for model, row in zip(V14_LOCKED_MODELS, pair):
                values = metric_rows.get((trajectory, offset, seed, model), {}) if matched else {}
                for label in labels:
                    value = values.get(label) if isinstance(values, Mapping) else None
                    accum[model][label].append(float(value) if _finite(value) else 0.0)
        # A complete manifest has exactly 15 cells. Preserve that fixed
        # denominator if a synthetic or damaged input omits both model rows.
        for _ in range(max(0, FIXED_CELLS_PER_TRAJECTORY - len(cells))):
            for model in V14_LOCKED_MODELS:
                for label in labels:
                    accum[model][label].append(0.0)
        per_model = {
            model: {label: float(np.mean(values)) for label, values in accum[model].items()}
            for model in V14_LOCKED_MODELS
        }
        for model in V14_LOCKED_MODELS:
            for label in labels:
                model_trajectory_values[model][label].append(per_model[model][label])
        trajectory_clusters[trajectory] = {
            label: float(np.mean([per_model[model][label] for model in V14_LOCKED_MODELS]))
            for label in labels
        }
    pooled = {
        label: float(np.mean([cluster[label] for cluster in trajectory_clusters.values()]))
        for label in labels
    }
    model_means = {
        model: {
            label: float(np.mean(values)) if values else 0.0
            for label, values in model_trajectory_values[model].items()
        }
        for model in V14_LOCKED_MODELS
    }
    return pooled, model_means, trajectory_clusters




def _descriptive_summaries_base(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, list[float]]] = {}
    metric_values: dict[str, list[float]] = {
        "score_error_mse": [], "visual_latent_prediction_mse": [],
        "proprio_latent_prediction_mse": [], "weighted_total_latent_prediction_mse": [],
        "absolute_P_minus_O_mean": [], "absolute_P_minus_O_median": [],
        "absolute_P_minus_O_max": [],
        "spearman_P_vs_J": [], "spearman_P_vs_G": [],
        "spearman_O_vs_J": [], "spearman_O_vs_G": [],
        "pairwise_accuracy_P_vs_J": [], "pairwise_accuracy_P_vs_G": [],
        "pairwise_accuracy_O_vs_J": [], "pairwise_accuracy_O_vs_G": [],
        "normalized_regret_at_1_P_J": [], "normalized_regret_at_1_P_G": [],
        "normalized_regret_at_1_O_J": [], "normalized_regret_at_1_O_G": [],
        "top_1_normalized_regret_J": [], "top_1_normalized_regret_G": [],
        "top_5_normalized_regret_J": [], "top_5_normalized_regret_G": [],
        "top_10_normalized_regret_J": [], "top_10_normalized_regret_G": [],
        "uniform_expectation_J": [], "uniform_expectation_G": [],
    }
    magnitude_bounded_metric_values: dict[str, list[float]] = {
        key: [] for key in (
            "score_error_mse", "absolute_P_minus_O_mean", "absolute_P_minus_O_median",
            "absolute_P_minus_O_max", "spearman_P_vs_J", "spearman_P_vs_G",
            "spearman_O_vs_J", "spearman_O_vs_G", "pairwise_accuracy_P_vs_J",
            "pairwise_accuracy_P_vs_G", "pairwise_accuracy_O_vs_J",
            "pairwise_accuracy_O_vs_G", "normalized_regret_at_1_P_J",
            "normalized_regret_at_1_P_G", "normalized_regret_at_1_O_J",
            "normalized_regret_at_1_O_G", "top_1_normalized_regret_J",
            "top_1_normalized_regret_G", "top_5_normalized_regret_J",
            "top_5_normalized_regret_G", "top_10_normalized_regret_J",
            "top_10_normalized_regret_G", "uniform_expectation_J",
            "uniform_expectation_G",
        )
    }
    magnitude_bounded_metric_rows: dict[tuple[str, int, int, str], dict[str, float]] = {}
    control_values: dict[str, dict[str, list[float]]] = {
        name: {outcome: [] for outcome in ("J", "G")}
        for name in ("random", "hold_position_noop", "expert_goal_defining_segment")
    }
    pair_names = ("P_vs_O_choice", "P_vs_finite_menu_best_G", "P_vs_finite_menu_best_J")
    pairs: dict[str, dict[str, Any]] = {
        name: {"eligible": 0, "distinct": [], "same": [], "abstained": []}
        for name in pair_names
    }
    for row in rows:
        phase = str(row.get("expert_phase_proxy", "unknown"))
        phases.setdefault(phase, {"Delta_J": [], "Delta_G": []})
        if row.get("raw_valid"):
            for label in ("Delta_J", "Delta_G"):
                phases[phase][label].append(float(row[f"raw_{label}"]))
        metrics = row.get("standard_outer_metrics")
        if isinstance(metrics, Mapping):
            latent = metrics.get("latent_prediction_mse")
            candidates = {
                "score_error_mse": metrics.get("score_error_mse"),
                "visual_latent_prediction_mse": latent.get("visual") if isinstance(latent, Mapping) else None,
                "proprio_latent_prediction_mse": latent.get("proprio") if isinstance(latent, Mapping) else None,
                "weighted_total_latent_prediction_mse": latent.get("weighted_total") if isinstance(latent, Mapping) else None,
                "absolute_P_minus_O_mean": metrics.get("absolute_P_minus_O", {}).get("mean"),
                "absolute_P_minus_O_median": metrics.get("absolute_P_minus_O", {}).get("median"),
                "absolute_P_minus_O_max": metrics.get("absolute_P_minus_O", {}).get("max"),
                "spearman_P_vs_J": metrics.get("spearman", {}).get("P_vs_J"),
                "spearman_P_vs_G": metrics.get("spearman", {}).get("P_vs_G"),
                "spearman_O_vs_J": metrics.get("spearman", {}).get("O_vs_J"),
                "spearman_O_vs_G": metrics.get("spearman", {}).get("O_vs_G"),
                "pairwise_accuracy_P_vs_J": metrics.get("pairwise_ranking_accuracy", {}).get("P_vs_J"),
                "pairwise_accuracy_P_vs_G": metrics.get("pairwise_ranking_accuracy", {}).get("P_vs_G"),
                "pairwise_accuracy_O_vs_J": metrics.get("pairwise_ranking_accuracy", {}).get("O_vs_J"),
                "pairwise_accuracy_O_vs_G": metrics.get("pairwise_ranking_accuracy", {}).get("O_vs_G"),
                "normalized_regret_at_1_P_J": metrics.get("normalized_regret_at_1", {}).get("P_J"),
                "normalized_regret_at_1_P_G": metrics.get("normalized_regret_at_1", {}).get("P_G"),
                "normalized_regret_at_1_O_J": metrics.get("normalized_regret_at_1", {}).get("O_J"),
                "normalized_regret_at_1_O_G": metrics.get("normalized_regret_at_1", {}).get("O_G"),
                "uniform_expectation_J": metrics.get("uniform_expectation", {}).get("J"),
                "uniform_expectation_G": metrics.get("uniform_expectation", {}).get("G"),
            }
            top_k = metrics.get("top_k_normalized_regret", {})
            for k in (1, 5, 10):
                top = top_k.get(str(k), {}) if isinstance(top_k, Mapping) else {}
                candidates[f"top_{k}_normalized_regret_J"] = top.get("J")
                candidates[f"top_{k}_normalized_regret_G"] = top.get("G")
            for key, value in candidates.items():
                if _finite(value):
                    metric_values[key].append(float(value))
            subset = metrics.get("action_only_subset")
            if isinstance(subset, Mapping):
                subset_candidates = {
                    "score_error_mse": subset.get("score_error_mse"),
                    "absolute_P_minus_O_mean": subset.get("absolute_P_minus_O", {}).get("mean"),
                    "absolute_P_minus_O_median": subset.get("absolute_P_minus_O", {}).get("median"),
                    "absolute_P_minus_O_max": subset.get("absolute_P_minus_O", {}).get("max"),
                    "spearman_P_vs_J": subset.get("spearman", {}).get("P_vs_J"),
                    "spearman_P_vs_G": subset.get("spearman", {}).get("P_vs_G"),
                    "spearman_O_vs_J": subset.get("spearman", {}).get("O_vs_J"),
                    "spearman_O_vs_G": subset.get("spearman", {}).get("O_vs_G"),
                    "pairwise_accuracy_P_vs_J": subset.get("pairwise_ranking_accuracy", {}).get("P_vs_J"),
                    "pairwise_accuracy_P_vs_G": subset.get("pairwise_ranking_accuracy", {}).get("P_vs_G"),
                    "pairwise_accuracy_O_vs_J": subset.get("pairwise_ranking_accuracy", {}).get("O_vs_J"),
                    "pairwise_accuracy_O_vs_G": subset.get("pairwise_ranking_accuracy", {}).get("O_vs_G"),
                    "normalized_regret_at_1_P_J": subset.get("normalized_regret_at_1", {}).get("P_J"),
                    "normalized_regret_at_1_P_G": subset.get("normalized_regret_at_1", {}).get("P_G"),
                    "normalized_regret_at_1_O_J": subset.get("normalized_regret_at_1", {}).get("O_J"),
                    "normalized_regret_at_1_O_G": subset.get("normalized_regret_at_1", {}).get("O_G"),
                    "uniform_expectation_J": subset.get("uniform_expectation", {}).get("J"),
                    "uniform_expectation_G": subset.get("uniform_expectation", {}).get("G"),
                }
                subset_top_k = subset.get("top_k_normalized_regret", {})
                for k in (1, 5, 10):
                    top = subset_top_k.get(str(k), {}) if isinstance(subset_top_k, Mapping) else {}
                    subset_candidates[f"top_{k}_normalized_regret_J"] = top.get("J")
                    subset_candidates[f"top_{k}_normalized_regret_G"] = top.get("G")
                if all(key in row for key in ("trajectory_id", "offset", "seed", "model")):
                    cell_key = (
                        str(row["trajectory_id"]), int(row["offset"]),
                        int(row["seed"]), str(row["model"]),
                    )
                    magnitude_bounded_metric_rows[cell_key] = {
                        key: float(value) for key, value in subset_candidates.items() if _finite(value)
                    }
                for key, value in subset_candidates.items():
                    if _finite(value):
                        magnitude_bounded_metric_values[key].append(float(value))
        controls = row.get("descriptive_controls")
        if isinstance(controls, Mapping):
            for name, outcomes in control_values.items():
                recorded = controls.get(name)
                if isinstance(recorded, Mapping):
                    for outcome in outcomes:
                        if _finite(recorded.get(outcome)):
                            outcomes[outcome].append(float(recorded[outcome]))
        pair_map = row.get("consequential_pair_decomposition")
        if row.get("raw_valid"):
            for name in pair_names:
                pairs[name]["eligible"] += 1
                pair = pair_map.get(name) if isinstance(pair_map, Mapping) else None
                if isinstance(pair, Mapping):
                    if pair.get("mechanism_only_abstention") is not None:
                        pairs[name]["abstained"].append(pair)
                    elif pair.get("selection_disagreement") is True:
                        pairs[name]["distinct"].append(pair)
                    elif pair.get("same_choice") is True:
                        pairs[name]["same"].append(pair)
                    else:
                        pairs[name]["abstained"].append(pair)
    standard_means = {key: float(np.mean(values)) if values else None for key, values in metric_values.items()}
    standard_means.update({
        "latent_prediction_mse": {
            "visual": standard_means["visual_latent_prediction_mse"],
            "proprio": standard_means["proprio_latent_prediction_mse"],
            "weighted_total": standard_means["weighted_total_latent_prediction_mse"],
        },
        "absolute_P_minus_O": {
            "mean": standard_means["absolute_P_minus_O_mean"],
            "median": standard_means["absolute_P_minus_O_median"],
            "max": standard_means["absolute_P_minus_O_max"],
        },
        "spearman": {
            key.removeprefix("spearman_"): standard_means[key]
            for key in ("spearman_P_vs_J", "spearman_P_vs_G", "spearman_O_vs_J", "spearman_O_vs_G")
        },
        "pairwise_ranking_accuracy": {
            key.removeprefix("pairwise_accuracy_"): standard_means[key]
            for key in ("pairwise_accuracy_P_vs_J", "pairwise_accuracy_P_vs_G", "pairwise_accuracy_O_vs_J", "pairwise_accuracy_O_vs_G")
        },
        "normalized_regret_at_1": {
            key.removeprefix("normalized_regret_at_1_"): standard_means[key]
            for key in ("normalized_regret_at_1_P_J", "normalized_regret_at_1_P_G", "normalized_regret_at_1_O_J", "normalized_regret_at_1_O_G")
        },
        "top_k_normalized_regret": {
            str(k): {
                "J": standard_means[f"top_{k}_normalized_regret_J"],
                "G": standard_means[f"top_{k}_normalized_regret_G"],
            }
            for k in (1, 5, 10)
        },
        "uniform_expectation": {
            "J": standard_means["uniform_expectation_J"],
            "G": standard_means["uniform_expectation_G"],
        },
    })
    (
        magnitude_bounded_means,
        magnitude_bounded_model_means,
        magnitude_bounded_trajectory_clusters,
    ) = _matched_descriptive_metric_means(
        rows, magnitude_bounded_metric_rows, tuple(magnitude_bounded_metric_values)
    )
    if not magnitude_bounded_means:
        magnitude_bounded_means = {
            key: float(np.mean(values)) if values else None
            for key, values in magnitude_bounded_metric_values.items()
        }
    magnitude_bounded_means.update({
        "absolute_P_minus_O": {
            "mean": magnitude_bounded_means["absolute_P_minus_O_mean"],
            "median": magnitude_bounded_means["absolute_P_minus_O_median"],
            "max": magnitude_bounded_means["absolute_P_minus_O_max"],
        },
        "spearman": {
            key.removeprefix("spearman_"): magnitude_bounded_means[key]
            for key in ("spearman_P_vs_J", "spearman_P_vs_G", "spearman_O_vs_J", "spearman_O_vs_G")
        },
        "pairwise_ranking_accuracy": {
            key.removeprefix("pairwise_accuracy_"): magnitude_bounded_means[key]
            for key in ("pairwise_accuracy_P_vs_J", "pairwise_accuracy_P_vs_G", "pairwise_accuracy_O_vs_J", "pairwise_accuracy_O_vs_G")
        },
        "normalized_regret_at_1": {
            key.removeprefix("normalized_regret_at_1_"): magnitude_bounded_means[key]
            for key in ("normalized_regret_at_1_P_J", "normalized_regret_at_1_P_G", "normalized_regret_at_1_O_J", "normalized_regret_at_1_O_G")
        },
        "top_k_normalized_regret": {
            str(k): {
                "J": magnitude_bounded_means[f"top_{k}_normalized_regret_J"],
                "G": magnitude_bounded_means[f"top_{k}_normalized_regret_G"],
            }
            for k in (1, 5, 10)
        },
        "uniform_expectation": {
            "J": magnitude_bounded_means["uniform_expectation_J"],
            "G": magnitude_bounded_means["uniform_expectation_G"],
        },
    })
    controls = {
        name: {outcome: float(np.mean(values)) if values else None for outcome, values in outcomes.items()}
        for name, outcomes in control_values.items()
    }
    missing_controls = [
        f"{name}.{outcome}"
        for name, outcomes in control_values.items()
        for outcome, values in outcomes.items()
        if not values
    ]
    control_blockers = [
        f"{name}.{outcome}: not recoverable from retained NPZ control primitives"
        for name, outcomes in control_values.items()
        for outcome, values in outcomes.items()
        if not values
    ]
    return {
        "expert_block_motion_phase_proxy": {
            phase: {label: {"mean": float(np.mean(values)) if values else None, "row_count": len(values)} for label, values in labels.items()}
            for phase, labels in phases.items()
        },
        "standard_metric_means": standard_means,
        "magnitude_bounded_standard_metric_means": magnitude_bounded_means,
        "magnitude_bounded_standard_metric_model_specific_means": magnitude_bounded_model_means,
        "magnitude_bounded_standard_metric_trajectory_clusters": magnitude_bounded_trajectory_clusters,
        "descriptive_controls": controls,
        "descriptive_controls_missing": missing_controls,
        "descriptive_control_blockers": control_blockers,
        "consequential_pair_mechanisms": {
            name: {
                "eligible_row_count": group["eligible"],
                "distinct_pair_count": len(group["distinct"]),
                "same_choice_count": len(group["same"]),
                "mechanism_abstention_count": len(group["abstained"]),
                **({
                    "P_O_selection_disagreement_fraction": (
                        float(len(group["distinct"]) / group["eligible"])
                        if group["eligible"] else None
                    )
                } if name == "P_vs_O_choice" else {
                    "recovery_aligned_ranking_reversal_fraction_all_eligible_rows": (
                        float(sum(pair["P_margin_selected_minus_counterfactual"] < 0 < pair["O_margin_selected_minus_counterfactual"] for pair in group["distinct"]) / group["eligible"])
                        if group["eligible"] else None
                    )
                }),
                "directional_dominant_fraction_conditional_on_distinct_pair": (
                    float(np.mean([abs(pair["directional_margin"]) > abs(pair["magnitude_margin"]) for pair in group["distinct"]]))
                    if group["distinct"] else None
                ),
                "mean_directional_margin_conditional_on_distinct_pair": float(np.mean([pair["directional_margin"] for pair in group["distinct"]])) if group["distinct"] else None,
                "mean_magnitude_margin_conditional_on_distinct_pair": float(np.mean([pair["magnitude_margin"] for pair in group["distinct"]])) if group["distinct"] else None,
                "max_abs_identity_residual": float(np.max(np.abs([pair["identity_residual"] for pair in group["distinct"]]))) if group["distinct"] else 0.0,
            }
            for name, group in pairs.items()
        },
        "status": "secondary descriptive; cannot change primary claim gates",
    }




def _descriptive_summaries(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = _descriptive_summaries_base(rows)
    result["mechanism_sensitivity"] = _mechanism_sensitivity(rows)
    baseline_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        baselines = row.get("recomputed_baselines", {})
        if not isinstance(baselines, Mapping):
            continue
        for name, outcomes in baselines.items():
            if not isinstance(outcomes, Mapping):
                continue
            for outcome, value in outcomes.items():
                if _finite(value):
                    baseline_values.setdefault(str(name), {}).setdefault(str(outcome), []).append(float(value))
    result["selection_baseline_means"] = {
        name: {outcome: float(np.mean(values)) for outcome, values in outcomes.items()}
        for name, outcomes in baseline_values.items()
    }
    drift_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        drift = row.get("coordinate_drift", {})
        if not isinstance(drift, Mapping):
            continue
        for encoding in ("chunk8", "chunk4"):
            sections = drift.get(encoding, {})
            if not isinstance(sections, Mapping):
                continue
            for component, values in sections.items():
                if not isinstance(values, Mapping):
                    continue
                for statistic in ("max_abs", "rms", "median_abs", "p95_abs"):
                    if _finite(values.get(statistic)):
                        drift_values.setdefault(encoding, {}).setdefault(component, {}).setdefault(statistic, []).append(float(values[statistic]))
    result["coordinate_drift_descriptive"] = {
        encoding: {component: {statistic: float(np.mean(values)) for statistic, values in statistics.items()}
                   for component, statistics in components.items()}
        for encoding, components in drift_values.items()
    }
    return result


def _mechanism_signature(pair: Mapping[str, Any]) -> tuple[str, int, int] | None:
    if not isinstance(pair, Mapping):
        return None
    if isinstance(pair.get("label"), str):
        try:
            return str(pair["label"]), int(np.sign(float(pair.get("directional_margin", 0)))), int(np.sign(float(pair.get("magnitude_margin", 0))))
        except (TypeError, ValueError):
            return None
    if not _finite(pair.get("directional_margin")) or not _finite(pair.get("magnitude_margin")):
        return None
    directional, magnitude = float(pair["directional_margin"]), float(pair["magnitude_margin"])
    dominant = "directional" if abs(directional) > abs(magnitude) else "magnitude" if abs(magnitude) > abs(directional) else "balanced"
    return dominant, int(np.sign(directional)), int(np.sign(magnitude))


def _mechanism_sensitivity(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("P_vs_O_choice", "P_vs_finite_menu_best_G", "P_vs_finite_menu_best_J")
    result: dict[str, Any] = {}
    for name in names:
        encodings = ("production", "chunk8", "chunk4")
        by_encoding: dict[str, list[tuple[str, int, int]]] = {key: [] for key in encodings}
        row_records: list[dict[str, Any]] = []
        for row in rows:
            maps = row.get("mechanism_by_encoding", {})
            if not isinstance(maps, Mapping):
                continue
            per_row: dict[str, Any] = {}
            signatures: list[tuple[str, int, int]] = []
            for encoding in encodings:
                values = maps.get(encoding)
                pair = values.get(name) if isinstance(values, Mapping) else None
                if isinstance(pair, Mapping):
                    per_row[encoding] = dict(pair)
                signature = _mechanism_signature(pair)
                if signature is not None:
                    by_encoding[encoding].append(signature)
                    signatures.append(signature)
                    per_row[encoding].update({
                        "directional_margin": float(pair["directional_margin"]),
                        "magnitude_margin": float(pair["magnitude_margin"]),
                        "dominance": signature[0], "directional_sign": signature[1],
                        "magnitude_sign": signature[2], "signature": list(signature),
                    })
            stable = bool(len(signatures) == len(encodings) and len(set(signatures)) == 1)
            label = signatures[0][0] if stable else "indeterminate"
            row_records.append({
                "decision_id": row.get("decision_id"), "trajectory_id": row.get("trajectory_id"),
                "model": row.get("model"), "offset": row.get("offset"), "seed": row.get("seed"),
                "stable": stable, "mechanism_label": label, "per_encoding": per_row,
            })
        per_encoding: dict[str, Any] = {}
        for encoding, values in by_encoding.items():
            encoded = [
                record for record in row_records
                if isinstance(record.get("per_encoding", {}).get(encoding), Mapping)
                and "signature" in record["per_encoding"][encoding]
            ]
            labels = [record["per_encoding"][encoding].get("dominance") for record in encoded]
            per_encoding[encoding] = {
                "row_count": len(values), "signatures": [list(item) for item in values],
                "dominance": labels[0] if labels and len(set(labels)) == 1 else "indeterminate",
                "sign_stable": bool(values and len({item[1:] for item in values}) == 1),
                # Keep each observed margin tied to its encoding and row;
                # never collapse heterogeneous rows to one signature.
                "margin_summaries": [
                    {key: record[key] for key in ("decision_id", "trajectory_id", "model", "offset", "seed")}
                    | {key: record["per_encoding"][encoding][key] for key in ("directional_margin", "magnitude_margin", "dominance", "directional_sign", "magnitude_sign")}
                    for record in encoded
                ],
            }
        stable_rows = [record for record in row_records if record["stable"]]
        label_counts = {label: sum(record["mechanism_label"] == label for record in row_records) for label in ("directional", "magnitude", "balanced", "indeterminate")}
        denominator = len(row_records)
        label_fractions = {label: (count / denominator if denominator else None) for label, count in label_counts.items()}
        stable_labels = {record["mechanism_label"] for record in stable_rows}
        result[name] = {
            "per_encoding": per_encoding, "rows": row_records,
            "row_count": len(row_records), "stable_row_count": len(stable_rows),
            "row_label_counts": label_counts, "row_label_fractions": label_fractions,
            "stable": bool(row_records and len(stable_rows) == len(row_records)),
            "mechanism_label": next(iter(stable_labels)) if len(stable_labels) == 1 and len(stable_rows) == len(row_records) else "indeterminate",
        }
    return result


def _issuance_reference(bundle: Mapping[str, Any] | None, *keys: str) -> Mapping[str, Any] | None:
    if not isinstance(bundle, Mapping):
        return None
    for key in keys:
        value = bundle.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _issuance_file_errors(label: str, reference: Mapping[str, Any] | None) -> list[str]:
    """Check an immutable issuance reference without trusting its JSON body."""
    if reference is None:
        return [f"sealed study lacks {label} issuance reference"]
    errors: list[str] = []
    path_value, file_hash, commitment = reference.get("path"), reference.get("sha256"), reference.get("issuance_sha256")
    if not isinstance(path_value, str) or not isinstance(file_hash, str) or not isinstance(commitment, str):
        return [f"sealed study {label} issuance reference is incomplete"]
    path = Path(path_value)
    if not path.is_file():
        return [f"sealed study {label} issuance file is missing"]
    if file_sha256(path) != file_hash:
        errors.append(f"sealed study {label} issuance file hash changed")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"sealed study {label} issuance cannot be read: {exc}")
    else:
        if not isinstance(record, Mapping):
            errors.append(f"sealed study {label} issuance is not an object")
        elif record.get("issuance_sha256") != commitment:
            errors.append(f"sealed study {label} issuance commitment changed")
        elif record.get("issuance_sha256") != canonical_sha256({key: value for key, value in record.items() if key != "issuance_sha256"}):
            errors.append(f"sealed study {label} issuance hash is not self-consistent")
        if isinstance(record, Mapping):
            if record.get("status") != "consumed" or record.get("one_shot") is not True:
                errors.append(f"sealed study {label} issuance is not a consumed one-shot record")
            if record.get("protocol_sha256") != v14_protocol_sha256() or record.get("source_bundle_sha256") != v14_source_bundle_sha256():
                errors.append(f"sealed study {label} issuance protocol/source binding mismatch")
    return errors


def _canonical_study_issuance(
    study_dir: Path, manifest: Mapping[str, Any], certificate_bundle: Mapping[str, Any],
    certificate_issuance: Mapping[str, Any] | None,
) -> tuple[list[str], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Load the writer's canonical STUDY record without mutating the bundle."""
    path = Path(V14_STUDY_ISSUANCE_PATH).resolve()
    if not path.is_file():
        return ["sealed study lacks its canonical STUDY issuance file"], None, None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return [f"canonical STUDY issuance cannot be read: {exc}"], None, None
    if not isinstance(record, Mapping):
        return ["canonical STUDY issuance is not an object"], None, None
    dataset = certificate_bundle.get("dataset_root")
    official = certificate_bundle.get("official_root")
    dataset_split = (certificate_bundle.get("dataset_bindings") or {}).get("dataset_split") if isinstance(certificate_bundle.get("dataset_bindings"), Mapping) else None
    expected = {
        "manifest_sha256": canonical_sha256(manifest),
        "certificate_bundle_sha256": certificate_bundle.get("bundle_sha256"),
        "certificate_issuance_sha256": certificate_issuance.get("sha256") if isinstance(certificate_issuance, Mapping) else None,
        "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
        "output_dir": str(Path(study_dir).resolve()),
        "official_root": str(Path(str(official)).resolve()) if isinstance(official, str) else official,
        "dataset_root": str(Path(str(dataset)).resolve()) if isinstance(dataset, str) else dataset,
        "dataset_split": dataset_split or "val", "study_ledger_path": str(V14_STUDY_LEDGER_PATH),
        "dinov2": certificate_bundle.get("dinov2"),
        "runtime": certificate_bundle.get("runtime"), "hardware": certificate_bundle.get("hardware"),
    }
    errors = validate_v14_study_issuance(record, expected_bindings=expected)
    # The writer's STUDY record carries the exact model specification used by
    # the run, not the certificate-hash-only map from the immutable bundle.
    # Validate the mapping independently so checkpoint/config paths cannot be
    # swapped while leaving the record's top-level commitment unchanged.
    certificates = certificate_bundle.get("certificates") if isinstance(certificate_bundle, Mapping) else None
    advertised_hashes = certificate_bundle.get("certificate_sha256", {}) if isinstance(certificate_bundle, Mapping) else {}
    models = record.get("models")
    if not isinstance(models, Mapping) or set(models) != set(V14_LOCKED_MODELS):
        errors.append("canonical STUDY issuance must bind exactly the two locked models")
    elif not isinstance(certificates, Mapping) or not isinstance(advertised_hashes, Mapping):
        errors.append("canonical STUDY issuance cannot be checked against model certificates")
    else:
        for model in V14_LOCKED_MODELS:
            spec = models.get(model)
            certificate = certificates.get(model)
            if not isinstance(spec, Mapping) or set(spec) != {"certificate_sha256", "checkpoint", "config"}:
                errors.append(f"canonical STUDY issuance {model} model specification is incomplete")
                continue
            if spec.get("certificate_sha256") != advertised_hashes.get(model):
                errors.append(f"canonical STUDY issuance {model} certificate hash differs from bundle")
            if not isinstance(certificate, Mapping):
                errors.append(f"canonical STUDY issuance {model} lacks its certificate")
                continue
            for field, digest_key in (("checkpoint", "checkpoint_sha256"), ("config", "config_sha256")):
                value = spec.get(field)
                if not isinstance(value, str):
                    errors.append(f"canonical STUDY issuance {model} {field} path is missing")
                    continue
                path_value = Path(value)
                if not path_value.is_absolute():
                    errors.append(f"canonical STUDY issuance {model} {field} path is not absolute")
                    continue
                if not path_value.is_file():
                    errors.append(f"canonical STUDY issuance {model} {field} file is missing")
                    continue
                expected_hash = certificate.get(digest_key)
                if not isinstance(expected_hash, str) or file_sha256(path_value) != expected_hash:
                    errors.append(f"canonical STUDY issuance {model} {field} file hash differs from certificate")
    reference = {
        "path": str(path), "sha256": file_sha256(path), "issuance_sha256": record.get("issuance_sha256"),
    }
    errors.extend(_issuance_file_errors("study", reference))
    if record.get("output_dir") != str(Path(study_dir).resolve()):
        errors.append("canonical STUDY issuance output_dir does not equal analyzed study directory")
    return sorted(set(errors)), reference, record


def _receipt_commitment_errors(
    decision_id: str, terminal: Mapping[str, Any], *, actual_files: Mapping[str, Any],
    expected: Mapping[str, Any], failed: bool = False,
) -> list[str]:
    """Validate all immutable receipt fields, including failed attempts."""
    errors: list[str] = []
    required = ["artifact_exists", "arrays_exists", "artifact_sha256", "arrays_sha256"]
    required.extend((
        "manifest_sha256", "certificate_bundle_sha256", "certificate_sha256",
        "certificate_issuance_sha256", "study_issuance", "protocol_sha256",
        "source_bundle_sha256",
    ))
    for key in required:
        value = actual_files.get(key, expected.get(key)) if key in {"artifact_exists", "arrays_exists", "artifact_sha256", "arrays_sha256"} else expected.get(key)
        if terminal.get(key) != value:
            errors.append(f"sealed ledger {decision_id} {'failed ' if failed else ''}receipt {key} changed or is missing")
    return errors


def _sealed_ledger_status(
    study_dir: Path, manifest: Mapping[str, Any], certificate_bundle: Mapping[str, Any] | None = None,
    study_issuance: Mapping[str, Any] | None = None,
    study_record: Mapping[str, Any] | None = None,
) -> tuple[list[str], set[str]]:
    ledger_path = V14_STUDY_LEDGER_PATH if isinstance(study_record, Mapping) else study_dir / LEDGER_NAME
    if not ledger_path.is_file():
        return ["sealed study has no append-only attempt ledger"], set()
    manifest_sha256 = canonical_sha256(manifest)
    bundle_sha256 = str(certificate_bundle.get("bundle_sha256")) if isinstance(certificate_bundle, Mapping) else None
    commitments = {"schema_version": manifest.get("schema_version"), "manifest_sha256": manifest_sha256,
                   "certificate_bundle_sha256": bundle_sha256, "protocol_sha256": v14_protocol_sha256(),
                   "source_bundle_sha256": v14_source_bundle_sha256()}
    if study_issuance is not None:
        commitments["study_issuance"] = dict(study_issuance)
    _header, events, errors, _last = read_v14_ledger(ledger_path, expected_commitments=commitments)
    strict_study_record = isinstance(study_record, Mapping)
    if strict_study_record and study_record.get("study_ledger_path") != str(V14_STUDY_LEDGER_PATH):
        errors.append("canonical STUDY issuance does not bind the authoritative external ledger")
    if _header is not None and strict_study_record and _header.get("run_id") != study_record.get("run_id"):
        errors.append("v1.4 ledger run_id differs from canonical STUDY issuance")
    if _header is not None:
        header_commitments = {
            "certificate_issuance_sha256": (certificate_bundle or {}).get("certificate_issuance", {}).get("sha256")
            if isinstance(certificate_bundle, Mapping) and isinstance(certificate_bundle.get("certificate_issuance"), Mapping) else None,
            "study_issuance": dict(study_issuance) if isinstance(study_issuance, Mapping) else None,
        }
        for key, value in header_commitments.items():
            if key in _header and _header.get(key) != value:
                errors.append(f"v1.4 ledger header {key} commitment mismatch")
    if _header is not None and study_issuance is None and "study_issuance" in _header:
        errors.append("sealed ledger carries a study-issuance commitment but the bundle does not")
    entry_items = [(str(entry["decision_id"]), entry) for entry in manifest["entries"]]
    entries = dict(entry_items)
    failed: set[str] = set()
    if set(events) != set(entries):
        errors.append("sealed ledger decision IDs do not exactly match manifest")
    certificate_hashes = certificate_bundle.get("certificate_sha256", {}) if isinstance(certificate_bundle, Mapping) else {}
    issuance = certificate_bundle.get("certificate_issuance") if isinstance(certificate_bundle, Mapping) else None
    issuance_sha256 = issuance.get("sha256") if isinstance(issuance, Mapping) else None
    expected_commitments = {
        "manifest_sha256": manifest_sha256, "certificate_bundle_sha256": bundle_sha256,
        "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
        "certificate_issuance_sha256": issuance_sha256,
        "study_issuance": dict(study_issuance) if isinstance(study_issuance, Mapping) else None,
    }
    certified_menus: dict[int, str] = {}
    if strict_study_record:
        try:
            certified_menus = _certified_menu_hashes(certificate_bundle or {})
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"sealed ledger certified proposal menus cannot be reconstructed: {exc}")
    certificates = certificate_bundle.get("certificates", {}) if isinstance(certificate_bundle, Mapping) else {}
    issuance_record = _load_issuance_record(issuance if isinstance(issuance, Mapping) else None)
    for ordinal, (decision_id, entry) in enumerate(entry_items):
        history = events.get(decision_id, [])
        statuses = [event.get("status") for event in history]
        artifact_path, arrays_path = study_dir / f"{decision_id}.json", study_dir / f"{decision_id}.npz"
        actual_files = {
            "artifact_exists": artifact_path.is_file(), "arrays_exists": arrays_path.is_file(),
            "artifact_sha256": file_sha256(artifact_path), "arrays_sha256": file_sha256(arrays_path),
        }
        expected_receipt = {
            **actual_files,
            **expected_commitments,
            "certificate_sha256": certificate_hashes.get(str(entry.get("model"))),
        }
        model = str(entry.get("model"))
        row_commitments: dict[str, Any] = {}
        if strict_study_record:
            row_bindings = _expected_bindings_for_entry(
                certificate_bundle or {}, model, entry,
                certificates.get(model) if isinstance(certificates, Mapping) else None,
                issuance_record, manifest_sha256,
            )
            row_commitments = {
                "ordinal": int(ordinal), "model": model,
                "episode": int(entry["episode"]), "offset": int(entry["offset"]),
                "menu_seed": int(entry["seed"]), "environment_seed": 1,
                "artifact_path": str(artifact_path.resolve()), "arrays_path": str(arrays_path.resolve()),
                "row_bindings": row_bindings,
                "certified_menu_sha256": certified_menus.get(int(entry["seed"])),
                "row_capability_path": str(_row_capability_path(study_dir, ordinal, decision_id)),
            }
            for event_index, event in enumerate(history):
                for key, value in row_commitments.items():
                    if key not in event or event.get(key) != value:
                        errors.append(f"sealed ledger {decision_id} event {event_index} {key} row commitment changed or is missing")
            expected_receipt.update(row_commitments)
            terminal_row_commitments = {
                **row_commitments,
                "row_capability": _row_capability_state(study_dir, ordinal, decision_id),
            }
            expected_receipt.update(terminal_row_commitments)
        else:
            terminal_row_commitments = row_commitments
        if statuses == ["started", "completed"]:
            receipt_kwargs = {
                "manifest_sha256": manifest_sha256,
                "certificate_bundle_sha256": bundle_sha256,
                "certificate_sha256": certificate_hashes.get(str(entry.get("model"))),
                "certificate_issuance_sha256": issuance_sha256,
                "study_issuance": dict(study_issuance) if isinstance(study_issuance, Mapping) else None,
                "row_commitments": terminal_row_commitments,
            }
            if "study_issuance" in inspect.signature(completed_receipt_errors).parameters:
                receipt_kwargs["study_issuance"] = dict(study_issuance) if isinstance(study_issuance, Mapping) else None
            errors.extend(f"sealed ledger {decision_id} {error}" for error in completed_receipt_errors(
                artifact_path, arrays_path, history[-1], **receipt_kwargs,
            ))
            errors.extend(_receipt_commitment_errors(decision_id, history[-1], actual_files=actual_files, expected=expected_receipt))
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                pass
            else:
                if strict_study_record and artifact.get("row_capability") != terminal_row_commitments.get("row_capability"):
                    errors.append(f"sealed ledger {decision_id} artifact row capability differs from its consumed receipt")
        elif statuses == ["started", "failed"]:
            terminal = history[-1]
            if terminal.get("failure_kind") not in {"scientific_integrity", "supervisor_interrupted"} or not terminal.get("error"):
                errors.append(f"sealed ledger {decision_id} failed receipt lacks failure evidence")
            errors.extend(f"sealed ledger {decision_id} {error}" for error in failed_receipt_errors(
                artifact_path, arrays_path, terminal,
                manifest_sha256=manifest_sha256,
                certificate_bundle_sha256=bundle_sha256,
                certificate_sha256=certificate_hashes.get(model),
                certificate_issuance_sha256=issuance_sha256,
                study_issuance=dict(study_issuance) if isinstance(study_issuance, Mapping) else None,
                row_commitments=terminal_row_commitments,
            ))
            errors.extend(_receipt_commitment_errors(decision_id, terminal, actual_files=actual_files, expected=expected_receipt, failed=True))
            failed.add(decision_id)
        else:
            errors.append(f"sealed ledger {decision_id} is not one legal terminal attempt")
    return errors, failed


def _canonical_manifest_errors(manifest: Mapping[str, Any], *, sealed: bool) -> list[str]:
    entries = manifest.get("entries")
    errors: list[str] = []
    if isinstance(entries, list):
        trajectories = {entry.get("trajectory_id") for entry in entries if isinstance(entry, Mapping)}
        cells = {(entry.get("trajectory_id"), entry.get("offset"), entry.get("seed")) for entry in entries if isinstance(entry, Mapping)}
        for key, expected in (("trajectory_count", len(trajectories)), ("segment_count", len(cells)), ("decision_count", len(entries))):
            if manifest.get(key) != expected:
                errors.append(f"v1.4 manifest {key} does not match its entries")
    if not sealed:
        return errors
    root = manifest.get("dataset_root")
    if not isinstance(root, str):
        return errors + ["v1.4 manifest lacks dataset_root for canonical schedule regeneration"]
    try:
        expected = make_v14_manifest(Path(root))
    except (OSError, ValueError, TypeError) as exc:
        return errors + [f"v1.4 canonical schedule cannot be regenerated: {exc}"]
    if manifest.get("entries") != expected.get("entries"):
        errors.append("v1.4 entries do not equal the canonical schedule regenerated from seq_lengths")
    if manifest.get("selection") != expected.get("selection"):
        errors.append("v1.4 selection does not equal the canonical regenerated selection")
    return errors


_FULL_BINDING_KEYS = (
    "model", "protocol_sha256", "source_bundle_sha256", "manifest_sha256", "checkpoint_sha256", "config_sha256",
    "official_git_commit", "official_root", "dataset_root", "dataset_split",
    "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256",
    "states_segment_sha256", "actions_segment_sha256", "environment_seed",
    "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256",
    "goal_observation_sha256", "dinov2", "runtime", "hardware", "runtime_sha256",
    "hardware_sha256", "menu_seed",
)


def _load_issuance_record(reference: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
        return None
    try:
        value = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, Mapping) else None


def _binding_candidates(
    bundle: Mapping[str, Any], model: str, entry: Mapping[str, Any],
    certificate: Mapping[str, Any] | None, issuance_record: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    """Return global, model, and row binding maps in decreasing specificity."""
    candidates: list[Mapping[str, Any]] = []
    candidates.append(bundle)
    for source in (entry.get("bindings"), entry.get("expected_bindings"), bundle.get("study_bindings"), bundle.get("expected_bindings"), bundle.get("bindings")):
        if not isinstance(source, Mapping):
            continue
        row_id = str(entry.get("decision_id"))
        model_source = source.get(model)
        row_source = source.get(row_id)
        if isinstance(row_source, Mapping):
            candidates.append(row_source)
        if isinstance(model_source, Mapping):
            row_model = model_source.get(row_id)
            if isinstance(row_model, Mapping):
                candidates.append(row_model)
            candidates.append(model_source)
        candidates.append(source)
    if isinstance(certificate, Mapping):
        candidates.append(certificate)
    if isinstance(issuance_record, Mapping):
        candidates.append(issuance_record)
        official = issuance_record.get("official_data_bindings")
        if isinstance(official, Mapping):
            candidates.append(official)
    return candidates


def _expected_bindings_for_entry(
    bundle: Mapping[str, Any], model: str, entry: Mapping[str, Any],
    certificate: Mapping[str, Any] | None, issuance_record: Mapping[str, Any] | None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the complete per-row map expected by the artifact validator."""
    dataset = bundle.get("dataset_bindings") if isinstance(bundle.get("dataset_bindings"), Mapping) else {}
    defaults: dict[str, Any] = {
        "model": model, "protocol_sha256": v14_protocol_sha256(),
        "source_bundle_sha256": v14_source_bundle_sha256(),
        "manifest_sha256": manifest_sha256 or bundle.get("manifest_sha256"),
        "checkpoint_sha256": certificate.get("checkpoint_sha256") if isinstance(certificate, Mapping) else None,
        "config_sha256": certificate.get("config_sha256") if isinstance(certificate, Mapping) else None,
        "official_root": bundle.get("official_root"), "dataset_root": bundle.get("dataset_root"),
        "dataset_split": dataset.get("dataset_split"), "dataset_archive_sha256": dataset.get("dataset_archive_sha256"),
        "seq_lengths_sha256": dataset.get("seq_lengths_sha256"), "dataset_states_sha256": dataset.get("dataset_states_sha256"),
        "environment_seed": 1, "menu_seed": entry.get("menu_seed", entry.get("seed")), "dinov2": bundle.get("dinov2"),
    }
    result: dict[str, Any] = {}
    candidates = _binding_candidates(bundle, model, entry, certificate, issuance_record)
    for key in _FULL_BINDING_KEYS:
        value = defaults.get(key)
        if key not in {
            "protocol_sha256", "source_bundle_sha256", "manifest_sha256", "environment_seed", "menu_seed", "model",
            # These are regenerated below from the scientific row's bound
            # dataset or retained NPZ, never inherited from certificate
            # repeat signatures (which are trajectory 7 only).
            "states_segment_sha256", "actions_segment_sha256", "initial_state_sha256",
            "goal_state_sha256", "initial_observation_sha256", "goal_observation_sha256",
        }:
            for candidate in candidates:
                if key in candidate and candidate.get(key) is not None:
                    value = candidate.get(key)
                    break
        if value is not None:
            result[key] = value
    if result.get("dataset_root") and result.get("dataset_split") and entry.get("episode") is not None and entry.get("offset") is not None:
        segment_values = _dataset_segment_hashes({
            **result, "episode": int(entry["episode"]), "offset": int(entry["offset"]),
        })
        result.update({key: value for key, value in segment_values.items() if value is not None})
    return result


def _missing_full_bindings(bindings: Mapping[str, Any]) -> list[str]:
    # A sealed run must carry all cross-run and per-row provenance.  Keeping
    # this check here (in addition to the common validator) prevents a
    # certificate that only binds checkpoint/dataset files from silently
    # dropping state, action, environment, or hardware identity.
    return [key for key in _FULL_BINDING_KEYS if key not in bindings]


def _result_interpretation(raw: Mapping[str, Any], support: Mapping[str, Any], *, claimable: bool = True) -> dict[str, str]:
    def classify(summary: Mapping[str, Any], label: str) -> str:
        stats, ci = summary.get(label, {}), summary.get(label, {}).get("trajectory_bootstrap_95_ci")
        adjusted_p = summary.get("holm_adjusted_p", {}).get(label)
        if not claimable or summary.get("validity_gate", {}).get("passed") is not True:
            return "unstable"
        if not _finite(stats.get("mean")) or not isinstance(ci, list) or len(ci) != 2 or not _finite(adjusted_p):
            return "unstable"
        if ci[0] > 0 and float(stats["mean"]) > 0 and float(adjusted_p) < 0.05:
            return "positive"
        if ci[1] < 0 and float(stats["mean"]) < 0 and float(adjusted_p) < 0.05:
            return "negative"
        if ci[0] <= 0 <= ci[1]:
            return "null"
        return "unstable"
    result = {f"raw_{label}": classify(raw, label) for label in ("Delta_G", "Delta_J")} | {f"S_{label}": classify(support, label) for label in ("Delta_G", "Delta_J")}
    statuses = list(result.values())
    result["overall"] = "unstable" if "unstable" in statuses else "mixed" if len(set(statuses)) > 1 else statuses[0]
    return result


def analyze_v14(study_dir: Path, manifest: Mapping[str, Any], certificate_bundle: Mapping[str, Any] | None = None, *, sealed: bool = True) -> dict[str, Any]:
    if manifest.get("schema_version") in {"PWA-PushT-v1.1", "PWA-PushT-v1.2"}:
        raise ValueError("v1.1/v1.2 evidence cannot enter the v1.4 analysis")
    errors = validate_v14_manifest(manifest, require_full=True)
    if not isinstance(manifest.get("entries"), list) or len(manifest["entries"]) != 600:
        errors.append("v1.4 analysis requires exactly 600 manifest entries")
    errors.extend(_canonical_manifest_errors(manifest, sealed=sealed))
    if errors:
        raise ValueError("invalid v1.4 manifest: " + "; ".join(sorted(set(errors))))
    certificate_hashes: Mapping[str, str] | None = None
    certificate_bindings: dict[str, dict[str, Any]] = {}
    terminal_failed: set[str] = set()
    certificate_issuance: Mapping[str, Any] | None = None
    study_issuance: Mapping[str, Any] | None = None
    if sealed:
        if not isinstance(certificate_bundle, Mapping):
            raise ValueError("sealed v1.4 analysis requires a v1.4 certificate bundle")
        bundle_errors = validate_v14_bundle(certificate_bundle)
        unsigned = {key: value for key, value in certificate_bundle.items() if key != "bundle_sha256"}
        if certificate_bundle.get("bundle_sha256") != canonical_sha256(unsigned):
            bundle_errors.append("v1.4 certificate bundle hash mismatch")
        if bundle_errors:
            raise ValueError("invalid v1.4 certificate bundle: " + "; ".join(sorted(set(bundle_errors))))
        certificate_hashes = certificate_bundle["certificate_sha256"]
        certificates = certificate_bundle.get("certificates", {})
        certificate_issuance = _issuance_reference(certificate_bundle, "certificate_issuance")
        bundle_errors.extend(_issuance_file_errors("certificate", certificate_issuance))
        study_errors, study_issuance, study_record = _canonical_study_issuance(
            Path(study_dir), manifest, certificate_bundle, certificate_issuance,
        )
        bundle_errors.extend(study_errors)
        if bundle_errors:
            raise ValueError("invalid v1.4 certificate bundle: " + "; ".join(sorted(set(bundle_errors))))
        issuance_record = _load_issuance_record(certificate_issuance)
        certificate_bindings = {
            model: _expected_bindings_for_entry(
                certificate_bundle, model,
                {"decision_id": "__global__", "seed": None},
                certificates.get(model) if isinstance(certificates, Mapping) else None,
                issuance_record,
                canonical_sha256(manifest),
            )
            for model in V14_LOCKED_MODELS
        }
        # Row-specific bindings (notably state/action segment and observation
        # hashes) are supplied through the bundle's expected study map.  The
        # common global values above remain a useful fallback for every row.
        ledger_errors, terminal_failed = _sealed_ledger_status(
            Path(study_dir), manifest, certificate_bundle, study_issuance, study_record,
        )
        ledger_errors.extend(paired_v14_inventory_errors(Path(study_dir), [entry for entry in manifest["entries"] if str(entry["decision_id"]) not in terminal_failed]))
        if ledger_errors:
            raise ValueError("unsealed v1.4 evidence: " + "; ".join(sorted(set(ledger_errors))))
    # Rebuild the complete map per manifest row, preserving the exact menu
    # seed and any row-specific segment/state/observation commitments.
    if sealed and isinstance(certificate_bundle, Mapping):
        certificates = certificate_bundle.get("certificates", {})
        issuance_record = _load_issuance_record(certificate_issuance)
        per_row_bindings: dict[str, dict[str, Any]] = {}
        for entry in manifest["entries"]:
            model = str(entry.get("model"))
            per_row_bindings[str(entry["decision_id"])] = _expected_bindings_for_entry(
                certificate_bundle, model, entry,
                certificates.get(model) if isinstance(certificates, Mapping) else None,
                issuance_record,
                canonical_sha256(manifest),
            )
        certificate_bindings = per_row_bindings
    certified_menu_hashes = _certified_menu_hashes(certificate_bundle) if sealed and isinstance(certificate_bundle, Mapping) else {}
    rows, invalid = _read_rows(Path(study_dir), manifest, certificate_hashes, certificate_bindings,
                               terminal_failed=terminal_failed, strict_completed=sealed,
                               certificate_issuance=certificate_issuance, study_issuance=study_issuance,
                               certified_menu_hashes=certified_menu_hashes)
    raw, support, usefulness = (_summarize(rows, mode) for mode in ("raw", "support", "usefulness"))
    conflicts = {label: bool(_finite(raw[label]["mean"]) and _finite(support[label]["mean"]) and np.sign(raw[label]["mean"]) != np.sign(support[label]["mean"])) for label in ("Delta_G", "Delta_J")}
    repeat_gate = {"passed": True, "models": list(V14_LOCKED_MODELS), "contracts_per_model": 3, "repeats_per_contract": 3, "fresh_process": True, "comparison_anchor": "repeat_0", "score_tolerance": 2e-5, "component_score_vectors_elementwise": True, "tie_set_stability": True}
    if isinstance(certificate_bundle, Mapping):
        certs = certificate_bundle.get("certificates", {})
        repeat_gate["passed"] = all(isinstance(certs.get(model), Mapping) and certs[model].get("supervisor_status") == "passed" and certs[model].get("repeatability", {}).get("passed") is True and certs[model].get("repeatability", {}).get("repeat_count") == 3 for model in V14_LOCKED_MODELS)
    sealed_bundle_passed = bool(sealed and isinstance(certificate_bundle, Mapping))
    claimable = bool(sealed_bundle_passed and repeat_gate["passed"] and raw["validity_gate"]["passed"] and support["validity_gate"]["passed"])
    broad = bool(claimable and not any(conflicts.values())
                 and all(raw["holm_adjusted_p"].get(label) is not None and raw["holm_adjusted_p"][label] < 0.05 for label in ("Delta_G", "Delta_J"))
                 and all(support["holm_adjusted_p"].get(label) is not None and support["holm_adjusted_p"][label] < 0.05 for label in ("Delta_G", "Delta_J"))
                 and all(_positive_with_ci(summary, label) for summary in (raw, support) for label in ("Delta_G", "Delta_J"))
                 and all(_positive_with_ci(summary["model_specific"][model], label) for summary in (raw, support) for model in V14_LOCKED_MODELS for label in ("Delta_G", "Delta_J")))
    useful = bool(claimable and usefulness["validity_gate"]["passed"] and all(usefulness["holm_adjusted_p"].get(label) is not None and usefulness["holm_adjusted_p"][label] < 0.05 for label in ("Delta_G", "Delta_J")) and all(_positive_with_ci(usefulness, label) for label in ("Delta_G", "Delta_J")))
    interpretation = _result_interpretation(raw, support, claimable=claimable)
    descriptive = _descriptive_summaries(rows)
    result: dict[str, Any] = {
        "analysis_version": V14_ANALYSIS_VERSION, "schema_version": V14_SCHEMA_VERSION,
        "protocol": V14_PROTOCOL, "protocol_sha256": v14_protocol_sha256(), "sealed": bool(sealed), "scientific_result": False,
        "row_count": len(rows), "valid_row_count": raw["validity_gate"]["valid_rows"],
        "inventory": "seeded_gaussian_proposal_menus_matching_released_cem_initialization",
        "source_bundle_sha256": v14_source_bundle_sha256(), "manifest_sha256": canonical_sha256(manifest),
        "certificate_bundle_sha256": certificate_bundle.get("bundle_sha256") if isinstance(certificate_bundle, Mapping) else None,
        "certificate_issuance": dict(certificate_issuance) if isinstance(certificate_issuance, Mapping) else None,
        "study_issuance": dict(study_issuance) if isinstance(study_issuance, Mapping) else None,
        "raw_inventory": raw, "magnitude_bounded_subset": support, "P_vs_uniform_usefulness": usefulness,
        "descriptive_diagnostics": descriptive,
        "mechanism_sensitivity": descriptive["mechanism_sensitivity"],
        "coordinate_drift_descriptive": descriptive["coordinate_drift_descriptive"],
        "invalid_rows": invalid,
        "integrity_gate": {"passed": sealed_bundle_passed, "sealed_only": True},
        "raw_S_conflict_gate": {"conflicts": conflicts, "passed": not any(conflicts.values())}, "repeat_decision_stability_gate": repeat_gate,
        "claimable": claimable,
        "claim_gates": {"broad_predictive_ranking_recovery": broad, "P_is_useful_vs_uniform_menu_expectation": useful, "isolated_findings_cannot_be_promoted": True},
        "result_interpretation": interpretation, "v1_2_evidence_reused": False,
    }
    result["analysis_commitment_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--certificate-bundle", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    study_dir = args.study_dir.resolve()
    protected = {args.manifest.resolve(), args.certificate_bundle.resolve(), V14_STUDY_ISSUANCE_PATH.resolve(), V14_STUDY_LEDGER_PATH.resolve(), V14_CERTIFICATE_ISSUANCE_PATH.resolve()}
    if output.suffix.lower() != ".json":
        raise ValueError("analysis output must be a new .json file")
    if output.exists():
        raise FileExistsError("analysis output already exists; refusing to overwrite any prior file")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundle = json.loads(args.certificate_bundle.read_text(encoding="utf-8"))
    protected_roots = [study_dir]
    for value in (bundle.get("official_root"), bundle.get("dataset_root")):
        if isinstance(value, str):
            protected_roots.append(Path(value).resolve())
    dinov2 = bundle.get("dinov2")
    if isinstance(dinov2, Mapping):
        for key in ("source_root", "torch_home"):
            if isinstance(dinov2.get(key), str):
                protected_roots.append(Path(str(dinov2[key])).resolve())
    if output in protected or any(output == root or root in output.parents for root in protected_roots):
        raise ValueError("analysis output must be outside the sealed study and its immutable inputs")
    result = analyze_v14(study_dir, manifest, bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
