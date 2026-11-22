"""Additional post-hoc context, not new inference or a change to the frozen study.

Run from the repository root with: python -m experiments.analyze_pusht_component_context
The original analysis and its outputs are read-only inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.analyze_pusht_component_posthoc import (
    ROOT, aggregate_fixed_denominator, dump_json, load_json, matched_cells,
    row_record, selected_utilities, sha256_file,
)


def summarize(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Expected at least two finite trajectory contrasts")
    loo = (values.sum() - values) / (len(values) - 1)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "positive": int(np.count_nonzero(values > 0)),
        "zero": int(np.count_nonzero(values == 0)),
        "negative": int(np.count_nonzero(values < 0)),
        "leave_one_out_mean_range": [float(loo.min()), float(loo.max())],
        "leave_one_out_means": loo.tolist(), "trajectory_values": values.tolist(),
    }


def absolute_context(cells: list, keys: list[str]) -> dict:
    """Equal weight to trajectories; missing physical outcomes are NOT zero."""
    trajectories = sorted({cell[0]["trajectory_id"] for cell in cells})
    results = {}
    for key in keys:
        p_values, o_values = [], []
        for trajectory in trajectories:
            observed = [cell for cell in cells if cell[0]["trajectory_id"] == trajectory]
            for selector, target in (("P", p_values), ("O", o_values)):
                cell_means = [np.mean([row["absolute"][selector][key] for row in cell]) for cell in observed]
                target.append(float(np.mean(cell_means)))
        # Distances are positive errors; rewards are higher-is-better outcomes.
        direction = 1 if key in ("terminal_coverage", "cumulative_J") else -1
        differences = direction * (np.asarray(o_values) - np.asarray(p_values))
        results[key] = {
            "scale": "higher_is_better_reward" if direction == 1 else "positive_goal_error",
            "quantity": "native_distance" if key == "native_G" else key,
            "P_mean": float(np.mean(p_values)), "O_mean": float(np.mean(o_values)),
            "P_trajectory_means": p_values, "O_trajectory_means": o_values,
            "available_cell_utility_contrast": summarize(differences.tolist()),
        }
    return results


def self_check() -> None:
    summary = summarize([1., 2., -3., 0.])
    assert (summary["positive"], summary["zero"], summary["negative"]) == (2, 1, 1)
    np.testing.assert_allclose(summary["leave_one_out_means"], [-1/3, -2/3, 1., 0.])
    for invalid in ([], [1.], [1., float("nan")]):
        try:
            summarize(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    def cell(trajectory: str, p: float, o: float) -> list:
        return [{"trajectory_id": trajectory, "absolute": {"P": {"block_position": p}, "O": {"block_position": o}}} for _ in range(2)]
    context = absolute_context([cell("a", 10, 8), cell("a", 20, 16), cell("b", 30, 27)], ["block_position"])["block_position"]
    assert context["P_mean"] == 22.5 and context["O_mean"] == 19.5
    assert context["available_cell_utility_contrast"]["mean"] == 3.


def markdown_report(result: dict) -> str:
    counts = result["counts"]
    support = result["available_cells_per_trajectory"]
    lines = ["# Additional post-hoc component context", "", result["scope"], "",
             "## Evidence accounting", "",
             f"{counts['valid_rows']} valid rows; {counts['matched_rows']} matched model rows; {counts['matched_cells']}/300 matched cells; {counts['trajectories']} trajectory clusters. Available matched-cell counts range from {min(support)} to {max(support)}. The available-cell levels weight all trajectories equally, including the 9-cell trajectory; informative missingness remains possible.", "",
             "## Trajectory consistency", "", result["contrast_estimand"], "",
             "Positive means realized-endpoint selection is better. All 20 leave-one-out means are retained in JSON in the same order as its trajectory IDs; the table reports their range, not a confidence interval or a new test.", "",
             "| Utility | Mean | Median | Positive / zero / negative trajectories | Leave-one-out mean range |",
             "|---|---:|---:|---:|---:|"]
    for key, item in result["fixed_denominator_consistency"].items():
        low, high = item["leave_one_out_mean_range"]
        lines.append(f"| {key} | {item['mean']:.6g} | {item['median']:.6g} | {item['positive']} / {item['zero']} / {item['negative']} | [{low:.6g}, {high:.6g}] |")
    lines += ["", "## Absolute scale on available matched cells", "", result["absolute_estimand"], "",
              "Positions/velocity/native distance are positive error magnitudes in native simulator units; angle is radians. Coverage and cumulative J retain their higher-is-better reward scales.", "",
              "| Quantity | P mean | O mean | Available-cell utility contrast |",
              "|---|---:|---:|---:|"]
    for key, item in result["absolute_context"].items():
        label = "native_distance" if key == "native_G" else key
        lines.append(f"| {label} | {item['P_mean']:.6g} | {item['O_mean']:.6g} | {item['available_cell_utility_contrast']['mean']:.6g} |")
    lines += ["", "## Interpretation", "", result["interpretation"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("Self-check passed")
        return

    original_path = ROOT / "output/pusht_component_posthoc_20260914.json"
    original = load_json(original_path)
    for binding in original["source_bindings"].values():
        if sha256_file(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {binding['path']}")
    merge_path = ROOT / original["source_bindings"]["recovery_merge_index"]["path"]
    merge = load_json(merge_path)
    valid = [row for row in merge["rows"] if row.get("valid")]
    rows = []
    for index, item in enumerate(valid, 1):
        for name in ("arrays", "artifact"):
            if sha256_file(Path(item[f"{name}_path"])) != item[f"{name}_sha256"]:
                raise ValueError(f"Evidence hash mismatch: {item['decision_id']} {name}")
        record = row_record(item)
        with np.load(item["arrays_path"], allow_pickle=False) as archive:
            states, goal, rewards = (archive[key] for key in ("initial_final_states", "goal_state", "initial_reward_traces"))
            if not all(np.all(np.isfinite(value)) for value in (states, goal, rewards)):
                raise ValueError(f"Non-finite physical evidence: {item['decision_id']}")
            record["absolute"] = {}
            for selector, choice in (("P", record["p_index"]), ("O", record["o_index"])):
                utility = selected_utilities(states, goal, rewards, choice)
                record["absolute"][selector] = {key: value if key in ("terminal_coverage", "cumulative_J") else -value for key, value in utility.items()}
        rows.append(record)
        if index % 100 == 0:
            print(f"Verified {index}/{len(valid)} saved rows", flush=True)

    cells = matched_cells(rows)
    trajectories = sorted({cell[0]["trajectory_id"] for cell in cells})
    if (len(valid), len(cells), len(trajectories)) != (577, 281, 20):
        raise ValueError("Unexpected analysis population")
    keys = list(original["component_effects"]) + ["native_G", "cumulative_J"]
    fixed = aggregate_fixed_denominator(cells, "effects", keys)
    for key, values in fixed.items():
        previous = original["component_effects"].get(key, original["replication_checks"].get(key))
        np.testing.assert_allclose(values, previous["trajectory_values"], rtol=0, atol=1e-12)
    observed_counts = [sum(cell[0]["trajectory_id"] == trajectory for cell in cells) for trajectory in trajectories]
    absolute = absolute_context(cells, keys)
    for key in keys:
        available = absolute[key]["available_cell_utility_contrast"]["trajectory_values"]
        np.testing.assert_allclose(np.asarray(available) * np.asarray(observed_counts) / 15., fixed[key], rtol=0, atol=1e-12)

    result = {
        "status": "complete_additional_posthoc_descriptive", "analysis_date": "2026-09-28",
        "scope": "Additional post-hoc descriptive analysis requested after the component results were known. No new simulation, hypothesis test, holdout, or alteration of the frozen analysis. All five components and original G/J are reported regardless of direction.",
        "contrast_estimand": "Original fixed-15-cell contrasts: average the two model rows within each available trajectory-offset-seed cell; sum cell contrasts and divide by 15; weight all 20 trajectories equally. Missing cells contribute zero contrast evidence.",
        "absolute_estimand": "Descriptive absolute outcomes: average the two models within each available matched cell, average available cells within each trajectory, then weight the 20 trajectories equally. Do NOT impute missing physical outcomes. These differences are a different estimand from the fixed-15 contrasts and do not inherit their p-values or confidence intervals.",
        "interpretation": "Leave-one-trajectory-out means only measure single-cluster influence on the observed mean sign; they do not establish leave-one-out statistical significance, practical adequacy, new-task generalization, or robustness to missingness. Absolute errors contextualize pose-error differences, not completion improvement, a deployable correction, or causal encoder attribution. Available-cell levels give every trajectory equal weight despite 9-15 observed cells; informative missingness remains possible.",
        "counts": {"valid_rows": len(valid), "matched_rows": 2 * len(cells), "matched_cells": len(cells), "trajectories": len(trajectories), "switches_matched": sum(row["switched"] for cell in cells for row in cell)},
        "trajectories": trajectories, "available_cells_per_trajectory": observed_counts,
        "source_bindings": {"original_result": {"path": str(original_path.relative_to(ROOT)), "sha256": sha256_file(original_path)}, "original_spec": {"path": original["spec_path"], "sha256": original["spec_sha256"]}, "analysis_script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__))}, **original["source_bindings"]},
        "fixed_denominator_consistency": {key: summarize(values) for key, values in fixed.items()},
        "absolute_context": absolute,
    }
    stem = ROOT / "output/pusht_component_context_20260928"
    dump_json(stem.with_suffix(".json"), result)
    stem.with_suffix(".md").write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))


if __name__ == "__main__":
    main()
