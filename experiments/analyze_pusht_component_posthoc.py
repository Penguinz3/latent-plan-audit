from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    return float(np.dot(x_centered, y_centered) / denom) if denom > 0 else math.nan


def circular_error(theta: float, goal: float) -> float:
    return abs(math.atan2(math.sin(theta - goal), math.cos(theta - goal)))


def percentile_bootstrap(values: np.ndarray, label: str, replicates: int) -> list[float]:
    seed = int(hashlib.sha256(f"PWA-PushT-v1.4-component:{label}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    total = 1 << len(values)
    bits = np.arange(len(values), dtype=np.uint64)
    exceed = 0
    for start in range(0, total, 65_536):
        masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bits) & 1).astype(np.float64)
        means = (signs @ values) / len(values)
        exceed += int(np.count_nonzero(np.abs(means) >= observed - 1e-15))
    return float(exceed / total)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted((value, key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[key] = running
    return adjusted


def stats(values: list[float], label: str, replicates: int) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite or empty trajectory values for {label}")
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "trajectory_count": int(len(data)),
        "trajectory_bootstrap_95_ci": percentile_bootstrap(data, label, replicates),
        "two_sided_exact_sign_flip_p": exact_sign_flip(data),
        "trajectory_values": [float(x) for x in data],
    }


def selected_utilities(states: np.ndarray, goal: np.ndarray, rewards: np.ndarray, index: int) -> dict[str, float]:
    state = np.asarray(states[index], dtype=np.float64)
    return {
        "agent_position": -float(np.linalg.norm(state[0:2] - goal[0:2])),
        "block_position": -float(np.linalg.norm(state[2:4] - goal[2:4])),
        "block_angle_circular": -circular_error(float(state[4]), float(goal[4])),
        "agent_velocity": -float(np.linalg.norm(state[5:7] - goal[5:7])),
        "terminal_coverage": float(rewards[index, -1]),
        "native_G": -float(np.linalg.norm(state - goal)),
        "cumulative_J": float(np.asarray(rewards[index], dtype=np.float64).sum()),
    }


def squared_contributions(states: np.ndarray, goal: np.ndarray, p_index: int, o_index: int) -> dict[str, float]:
    p_diff = np.asarray(states[p_index], dtype=np.float64) - goal
    o_diff = np.asarray(states[o_index], dtype=np.float64) - goal
    return {
        "agent_position_sq": float(np.dot(p_diff[0:2], p_diff[0:2]) - np.dot(o_diff[0:2], o_diff[0:2])),
        "block_position_sq": float(np.dot(p_diff[2:4], p_diff[2:4]) - np.dot(o_diff[2:4], o_diff[2:4])),
        "block_angle_raw_sq": float(p_diff[4] ** 2 - o_diff[4] ** 2),
        "agent_velocity_sq": float(np.dot(p_diff[5:7], p_diff[5:7]) - np.dot(o_diff[5:7], o_diff[5:7])),
    }


def row_record(merge_row: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(merge_row["artifact_path"])
    arrays_path = Path(merge_row["arrays_path"])
    artifact = load_json(artifact_path)
    recovery = artifact["standard_outer_metrics"]["matched_recovery"]
    if recovery.get("abstained"):
        raise ValueError(f"Unexpected abstention in valid matched row: {artifact['decision_id']}")
    p_index = int(recovery["predicted_choice_index"])
    o_index = int(recovery["realized_latent_choice_index"])
    alpha = float(artifact["standard_outer_metrics"]["latent_prediction_mse"]["alpha"])

    with np.load(arrays_path, allow_pickle=False) as archive:
        states = np.asarray(archive["initial_final_states"], dtype=np.float64)
        goal = np.asarray(archive["goal_state"], dtype=np.float64)
        rewards = np.asarray(archive["initial_reward_traces"], dtype=np.float64)
        p_scores = np.asarray(archive["initial_objective_scores"], dtype=np.float64)
        o_scores = (
            np.asarray(archive["initial_observed_visual_mse"], dtype=np.float64)
            + alpha * np.asarray(archive["initial_observed_proprio_mse"], dtype=np.float64)
        )
        native_distances = np.asarray(archive["initial_native_state_distances"], dtype=np.float64)

    if states.shape != (300, 7) or goal.shape != (7,) or rewards.shape != (300, 30):
        raise ValueError(f"Unexpected array shape in {arrays_path}")
    if not np.all(np.isfinite(p_scores)) or not np.all(np.isfinite(o_scores)):
        raise ValueError(f"Non-finite score in {arrays_path}")
    if abs(float(native_distances[p_index]) - float(np.linalg.norm(states[p_index] - goal))) > 1e-7:
        raise ValueError(f"Native P distance mismatch in {arrays_path}")
    if abs(float(native_distances[o_index]) - float(np.linalg.norm(states[o_index] - goal))) > 1e-7:
        raise ValueError(f"Native O distance mismatch in {arrays_path}")
    if p_scores[p_index] > np.min(p_scores) + 2e-5 or o_scores[o_index] > np.min(o_scores) + 2e-5:
        raise ValueError(f"Saved selector does not minimize score in {arrays_path}")

    p_utility = selected_utilities(states, goal, rewards, p_index)
    o_utility = selected_utilities(states, goal, rewards, o_index)
    effects = {key: o_utility[key] - p_utility[key] for key in p_utility}
    sq = squared_contributions(states, goal, p_index, o_index)
    identity_target = float(native_distances[p_index] ** 2 - native_distances[o_index] ** 2)
    identity_residual = float(sum(sq.values()) - identity_target)

    abs_error = np.abs(p_scores - o_scores)
    iqr_p = float(np.quantile(p_scores, 0.75) - np.quantile(p_scores, 0.25))
    iqr_o = float(np.quantile(o_scores, 0.75) - np.quantile(o_scores, 0.25))
    median_abs = float(np.median(abs_error))
    score_scale = {
        "spearman_P_O": spearman(p_scores, o_scores),
        "median_abs_P_minus_O": median_abs,
        "IQR_P": iqr_p,
        "IQR_O": iqr_o,
        "median_abs_P_minus_O_over_IQR_P": median_abs / max(iqr_p, 1e-12),
        "median_abs_P_minus_O_over_IQR_O": median_abs / max(iqr_o, 1e-12),
    }

    return {
        "decision_id": artifact["decision_id"],
        "trajectory_id": artifact["trajectory_id"],
        "offset": int(artifact["offset"]),
        "menu_seed": int(artifact["menu_seed"]),
        "model": artifact["model"],
        "p_index": p_index,
        "o_index": o_index,
        "switched": p_index != o_index,
        "effects": effects,
        "squared_contributions": sq,
        "squared_identity_residual": identity_residual,
        "score_scale": score_scale,
        "evidence_source": merge_row["evidence_source"],
    }


def matched_cells(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["trajectory_id"], row["offset"], row["menu_seed"])
        grouped.setdefault(key, []).append(row)
    cells: list[list[dict[str, Any]]] = []
    expected_models = {"official_jepa_wm", "official_dino_wm"}
    for key in sorted(grouped):
        group = grouped[key]
        if {row["model"] for row in group} == expected_models and len(group) == 2:
            cells.append(group)
    return cells


def aggregate_fixed_denominator(cells: list[list[dict[str, Any]]], field: str, keys: list[str]) -> dict[str, list[float]]:
    trajectories = sorted({cell[0]["trajectory_id"] for cell in cells})
    output = {key: [] for key in keys}
    for trajectory in trajectories:
        trajectory_cells = [cell for cell in cells if cell[0]["trajectory_id"] == trajectory]
        for key in keys:
            cell_means = [float(np.mean([row[field][key] for row in cell])) for cell in trajectory_cells]
            output[key].append(float(np.sum(cell_means) / 15.0))
    return output


def aggregate_score_scale(cells: list[list[dict[str, Any]]], keys: list[str]) -> dict[str, list[float]]:
    trajectories = sorted({cell[0]["trajectory_id"] for cell in cells})
    output = {key: [] for key in keys}
    for trajectory in trajectories:
        trajectory_rows = list(itertools.chain.from_iterable(cell for cell in cells if cell[0]["trajectory_id"] == trajectory))
        for key in keys:
            output[key].append(float(np.mean([row["score_scale"][key] for row in trajectory_rows])))
    return output


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Push-T post-hoc physical-component and score-scale analysis",
        "",
        "This analysis was frozen before inspecting component outcomes and is separate from the preregistered PWA-PushT-v1.4 confirmatory analysis.",
        "",
        "## Evidence accounting",
        "",
        f"- Valid merge rows inspected: {result['counts']['valid_merge_rows']}",
        f"- Model-matched rows: {result['counts']['model_matched_rows']}",
        f"- Matched cells: {result['counts']['matched_cells']}",
        f"- Trajectory clusters: {result['counts']['trajectory_clusters']}",
        f"- P/O selector switches on matched rows: {result['counts']['switches_matched_rows']}",
        "",
        "## Component effects",
        "",
        "Positive values favor the action selected using realized-endpoint scores. Intervals are unadjusted trajectory-bootstrap intervals; p_H is Holm-adjusted across the five declared components.",
        "",
        "| Component utility | Mean O-P | 95% CI | exact p | p_H |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "agent_position": "Agent-position utility",
        "block_position": "Block-position utility",
        "block_angle_circular": "Circular block-angle utility",
        "agent_velocity": "Agent-velocity utility",
        "terminal_coverage": "Terminal coverage reward",
    }
    for key, label in labels.items():
        item = result["component_effects"][key]
        ci = item["trajectory_bootstrap_95_ci"]
        lines.append(f"| {label} | {item['mean']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {item['two_sided_exact_sign_flip_p']:.6g} | {item['holm_adjusted_p']:.6g} |")
    lines.extend([
        "",
        "## Exact native squared-distance decomposition",
        "",
        "These descriptive terms sum exactly to the selected P-minus-O change in squared native state distance.",
        "",
        "| Squared-error contribution | Mean P-O |",
        "|---|---:|",
    ])
    for key, item in result["native_squared_decomposition"].items():
        if key != "max_abs_identity_residual":
            lines.append(f"| {key} | {item['mean']:.6g} |")
    lines.append(f"\nMaximum row-level identity residual: {result['native_squared_decomposition']['max_abs_identity_residual']:.3g}.")
    lines.extend([
        "",
        "## Score-scale diagnostics",
        "",
        "Values below are trajectory-weighted descriptive means across matched rows; ratios compare typical absolute score error with within-menu score spread.",
        "",
        "| Quantity | Mean | 95% trajectory-bootstrap CI |",
        "|---|---:|---:|",
    ])
    for key, item in result["score_scale_diagnostics"].items():
        ci = item["trajectory_bootstrap_95_ci"]
        lines.append(f"| {key} | {item['mean']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        result["interpretation_boundary"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "experiments/official_pusht_v14_component_posthoc_spec_20260914.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "output/pusht_component_posthoc_20260914.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "output/pusht_component_posthoc_20260914.md")
    args = parser.parse_args()

    spec = load_json(args.spec)
    bindings = spec["source_bindings"]
    for binding in bindings.values():
        path = ROOT / binding["path"]
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise RuntimeError(f"Source binding mismatch for {path}: {actual}")

    merge = load_json(ROOT / bindings["recovery_merge_index"]["path"])
    valid_merge_rows = [row for row in merge["rows"] if row.get("valid")]
    rows = [row_record(row) for row in valid_merge_rows]
    cells = matched_cells(rows)
    matched = list(itertools.chain.from_iterable(cells))
    trajectories = sorted({row["trajectory_id"] for row in matched})

    expected = spec["analysis_population"]
    observed = (len(matched), len(cells), len(trajectories))
    required = (expected["expected_model_rows"], expected["expected_cells"], expected["expected_trajectory_clusters"])
    if observed != required:
        raise RuntimeError(f"Matched population mismatch: observed={observed}, required={required}")

    component_keys = list(spec["component_outcomes"])
    all_effect_keys = component_keys + ["native_G", "cumulative_J"]
    effect_values = aggregate_fixed_denominator(cells, "effects", all_effect_keys)
    component_stats = {key: stats(effect_values[key], key, spec["inference"]["bootstrap_replicates"]) for key in component_keys}
    adjusted = holm({key: item["two_sided_exact_sign_flip_p"] for key, item in component_stats.items()})
    for key in component_keys:
        component_stats[key]["holm_adjusted_p"] = adjusted[key]

    sealed = load_json(ROOT / bindings["sealed_analysis"]["path"])
    replication = {
        "native_G": stats(effect_values["native_G"], "native_G_replication", spec["inference"]["bootstrap_replicates"]),
        "cumulative_J": stats(effect_values["cumulative_J"], "cumulative_J_replication", spec["inference"]["bootstrap_replicates"]),
        "sealed_native_G_mean": float(sealed["raw_inventory"]["Delta_G"]["mean"]),
        "sealed_cumulative_J_mean": float(sealed["raw_inventory"]["Delta_J"]["mean"]),
    }
    if abs(replication["native_G"]["mean"] - replication["sealed_native_G_mean"]) > 1e-7:
        raise RuntimeError("Failed to reproduce sealed Delta_G")
    if abs(replication["cumulative_J"]["mean"] - replication["sealed_cumulative_J_mean"]) > 1e-5:
        raise RuntimeError("Failed to reproduce sealed Delta_J")

    sq_keys = list(spec["exact_native_squared_decomposition"]["terms"])
    sq_field_keys = ["agent_position_sq", "block_position_sq", "block_angle_raw_sq", "agent_velocity_sq"]
    del sq_keys
    sq_values = aggregate_fixed_denominator(cells, "squared_contributions", sq_field_keys)
    sq_result = {key: stats(sq_values[key], f"sq_{key}", spec["inference"]["bootstrap_replicates"]) for key in sq_field_keys}
    sq_result["max_abs_identity_residual"] = max(abs(row["squared_identity_residual"]) for row in matched)

    score_keys = list(spec["score_scale_diagnostics"]["quantities"])
    score_keys = [
        "spearman_P_O",
        "median_abs_P_minus_O_over_IQR_P",
        "median_abs_P_minus_O_over_IQR_O",
        "IQR_P",
        "IQR_O",
        "median_abs_P_minus_O",
    ]
    score_values = aggregate_score_scale(cells, score_keys)
    score_result = {key: stats(score_values[key], f"score_{key}", spec["inference"]["bootstrap_replicates"]) for key in score_keys}

    disagreement = [row for row in matched if row["switched"]]
    switch_component_direction = {
        key: {
            "O_better": sum(row["effects"][key] > 0 for row in disagreement),
            "equal": sum(row["effects"][key] == 0 for row in disagreement),
            "O_worse": sum(row["effects"][key] < 0 for row in disagreement),
        }
        for key in component_keys
    }

    result = {
        "schema_version": "pusht-component-posthoc-result-v1",
        "status": "complete_posthoc",
        "spec_path": str(args.spec.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": sha256_file(args.spec),
        "source_bindings": bindings,
        "counts": {
            "valid_merge_rows": len(valid_merge_rows),
            "model_matched_rows": len(matched),
            "matched_cells": len(cells),
            "trajectory_clusters": len(trajectories),
            "switches_matched_rows": sum(row["switched"] for row in matched),
        },
        "replication_checks": replication,
        "component_effects": component_stats,
        "switch_component_direction_counts": switch_component_direction,
        "native_squared_decomposition": sq_result,
        "score_scale_diagnostics": score_result,
        "sealed_latent_prediction_mse": sealed["descriptive_diagnostics"]["standard_metric_means"]["latent_prediction_mse"],
        "interpretation_boundary": "These post-hoc component estimates identify which stored state terms differ between the P- and O-selected actions. They do not causally identify the encoder, objective, predictor, or controller as the source of a difference. The score-scale summaries describe perturbation size relative to menu spread; they are not uncertainty estimates.",
    }
    dump_json(args.output_json, result)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(result), encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
