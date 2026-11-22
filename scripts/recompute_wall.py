"""CPU-only verifier for the compact Wall recomputation artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _frozen_probability(frozen: dict[str, Any], name: str, model: str, feature: float) -> float:
    spec = frozen["probability_models"][name]
    standardized = (feature - float(spec["center"])) / float(spec["scale"])
    is_dino = 1.0 if model == "dino" else 0.0
    intercept, dino_coef, feature_coef = (float(value) for value in spec["coefficients"])
    return _sigmoid(intercept + dino_coef * is_dino + feature_coef * standardized)


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    pairs = sorted((float(score), bool(label)) for score, label in zip(scores, labels) if math.isfinite(float(score)))
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        rank_sum += ((index + 1 + end) / 2.0) * sum(label for _, label in pairs[index:end])
        index = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def load_candidates(bundle: Path) -> dict[tuple[str, int, int, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = {}
    with (bundle / "candidate_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["case_id"]), int(row["case_seed"]), int(row["menu_seed"]), str(row["model"]))
            groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate"]))
        if [int(row["candidate"]) for row in rows] != list(range(300)):
            raise ValueError(f"candidate indices are not exactly 0..299 for {key}")
        if len({(row["predicted_choice_P"], row["encoded_real_choice_O"]) for row in rows}) != 1:
            raise ValueError(f"choice metadata differs within {key}")
        p_costs = np.asarray([float(row["predicted_cost_P"]) for row in rows], dtype=np.float64)
        o_costs = np.asarray([float(row["encoded_real_cost_O"]) for row in rows], dtype=np.float64)
        distances = np.asarray([float(row["native_terminal_distance"]) for row in rows], dtype=np.float64)
        if not np.isfinite(p_costs).all() or not np.isfinite(o_costs).all() or not np.isfinite(distances).all():
            raise ValueError(f"non-finite exported vector for {key}")
        if not all(isinstance(row["native_terminal_success"], bool) for row in rows):
            raise ValueError(f"native success vector is not boolean for {key}")
        stored_p = int(rows[0]["predicted_choice_P"])
        stored_o = int(rows[0]["encoded_real_choice_O"])
        if stored_p != int(np.argmin(p_costs)) or stored_o != int(np.argmin(o_costs)):
            raise ValueError(f"stored P/O choices are not first argmins of the exported vectors for {key}")
    return groups


def derive_record(rows: list[dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    first = rows[0]
    model = str(first["model"])
    p_index = int(first["predicted_choice_P"])
    o_index = int(first["encoded_real_choice_O"])
    distances = np.asarray([float(row["native_terminal_distance"]) for row in rows], dtype=np.float64)
    successes = np.asarray([float(row["native_terminal_success"]) for row in rows], dtype=np.float64)
    unique: dict[str, float] = {}
    for row in rows:
        key = str(row["actions_physical_sha256"])
        score = float(row["predicted_cost_P"])
        unique[key] = min(unique.get(key, float("inf")), score)
    unique_scores = np.sort(np.asarray(list(unique.values()), dtype=np.float64))
    raw_margin = float(unique_scores[1] - unique_scores[0]) if len(unique_scores) >= 2 else None
    iqr = float(np.subtract(*np.quantile(unique_scores, [0.75, 0.25]))) if len(unique_scores) else None
    normalized_margin = raw_margin / max(iqr, 1e-12) if raw_margin is not None and iqr is not None else None
    near_tie_score = -math.log10(max(normalized_margin, 1e-12)) if normalized_margin is not None else None
    p_dist, o_dist = float(distances[p_index]), float(distances[o_index])
    p_success, o_success = float(successes[p_index]), float(successes[o_index])
    uniform_utility = float(np.mean(-distances))
    uniform_success = float(np.mean(successes))
    minimum, maximum = float(np.min(distances)), float(np.max(distances))
    distance_range = maximum - minimum
    p_gap, o_gap = minimum - p_dist, minimum - o_dist
    return {
        "case_id": first["case_id"], "case_seed": int(first["case_seed"]), "menu_seed": int(first["menu_seed"]), "model": model, "decision_id": first["decision_id"],
        "initial_state": None, "goal_state": None, "p_index": p_index, "o_index": o_index, "p_distance": p_dist, "o_distance": o_dist,
        "p_success": p_success, "o_success": o_success, "o_minus_p_native_utility": p_dist - o_dist, "o_minus_p_binary_success": o_success - p_success,
        "p_minus_uniform_native_utility": -p_dist - uniform_utility, "p_minus_uniform_binary_success": p_success - uniform_success,
        "p_signed_native_menu_gap": p_gap, "o_signed_native_menu_gap": o_gap,
        "p_normalized_native_menu_gap": p_gap / distance_range if distance_range > 0 else None, "o_normalized_native_menu_gap": o_gap / distance_range if distance_range > 0 else None,
        "menu_min_distance": minimum, "menu_max_distance": maximum, "menu_distance_range": distance_range, "menu_constant_distance": distance_range == 0.0,
        "p_choice_o_choice_switch": p_index != o_index, "raw_margin": raw_margin, "unique_physical_action_count": len(unique), "physical_action_duplicate_count": 300 - len(unique),
        "raw_margin_iqr_unique_P": iqr, "normalized_margin": normalized_margin, "near_tie_score": near_tie_score,
        "near_tie_threshold": float(frozen["thresholds"][model]["near_tie_score_greater_equal"]),
        "near_tie_flagged": bool(near_tie_score is not None and near_tie_score >= float(frozen["thresholds"][model]["near_tie_score_greater_equal"])),
        "frozen_probability_near_tie": _frozen_probability(frozen, "near_tie_score", model, near_tie_score) if near_tie_score is not None else None,
        "frozen_probability_raw_margin": _frozen_probability(frozen, "raw_margin", model, -math.log10(max(raw_margin, 1e-12))) if raw_margin is not None else None,
    }


def compare_number(actual: Any, expected: Any, label: str, differences: list[dict[str, Any]]) -> None:
    if actual is None or expected is None:
        if actual != expected:
            differences.append({"field": label, "actual": actual, "expected": expected})
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        differences.append({"field": label, "actual": actual, "expected": expected})


def verify(bundle: Path) -> dict[str, Any]:
    reference = json.loads((bundle / "metadata" / "analysis_reference.json").read_text(encoding="utf-8"))
    frozen = json.loads((bundle / "metadata" / "frozen_near_tie.json").read_text(encoding="utf-8"))
    groups = load_candidates(bundle)
    manifest = json.loads((bundle / "metadata" / "manifest.json").read_text(encoding="utf-8"))
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != 180:
        raise ValueError("exported manifest does not contain 180 rows")
    manifest_keys = {(row["case_id"], int(row["case_seed"]), int(row["menu_seed"]), row["model"]) for row in manifest_rows}
    group_keys = set(groups)
    failure = json.loads((bundle / "metadata" / "failure_ledger.json").read_text(encoding="utf-8"))
    failure_keys = {(row["item"]["case_id"], int(row["item"]["case_seed"]), int(row["item"]["menu_seed"]), row["item"]["model"]) for row in failure.get("rows", [])}
    if manifest_keys - group_keys != failure_keys or group_keys - manifest_keys or len(failure_keys) != 1:
        raise ValueError("candidate coverage does not match the manifest and one-row failure ledger")
    records = [derive_record(rows, frozen) for _, rows in sorted(groups.items())]
    if len(records) != 179:
        raise ValueError(f"expected 179 valid candidate groups, found {len(records)}")
    complete_cases = sorted({row["case_id"] for row in records if sum(item["case_id"] == row["case_id"] for item in records) == 6})
    outcomes = ("o_minus_p_native_utility", "o_minus_p_binary_success", "p_minus_uniform_native_utility", "p_minus_uniform_binary_success")
    differences: list[dict[str, Any]] = []
    computed_effects: dict[str, Any] = {}
    analyzer_path = bundle / "frozen_wall_analysis.py"
    spec = importlib.util.spec_from_file_location("frozen_wall_analysis", analyzer_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load bundled frozen analyzer")
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    for outcome in outcomes:
        case_values = [float(np.mean([row[outcome] for row in records if row["case_id"] == case])) for case in complete_cases]
        expected = reference["primary"]["pooled"][outcome]
        computed_effects[outcome] = analyzer._stats(case_values, f"recompute:{outcome}")
        if len(case_values) != len(expected["case_values"]):
            differences.append({"field": f"primary.{outcome}.case_count", "actual": len(case_values), "expected": len(expected["case_values"])})
        for index, value in enumerate(case_values):
            compare_number(value, expected["case_values"][index], f"primary.{outcome}.case_values[{index}]", differences)
        for field in ("mean", "median", "sd", "two_sided_sign_flip_p"):
            compare_number(computed_effects[outcome][field], expected[field], f"primary.{outcome}.{field}", differences)
        if computed_effects[outcome]["case_count"] != expected["case_count"] or computed_effects[outcome]["sign_flip_method"] != expected["sign_flip_method"]:
            differences.append({"field": f"primary.{outcome}.metadata", "actual": computed_effects[outcome]["case_count"], "expected": expected["case_count"]})
        for index, value in enumerate(computed_effects[outcome]["percentile_bootstrap_95_ci_unadjusted"] or []):
            compare_number(value, expected["percentile_bootstrap_95_ci_unadjusted"][index], f"primary.{outcome}.bootstrap_ci[{index}]", differences)
    scores_near = [float(row["near_tie_score"]) for row in records]
    scores_margin = [-float(row["raw_margin"]) for row in records]
    switches = [bool(row["p_choice_o_choice_switch"]) for row in records]
    computed_auroc = {"near_tie_score": _auroc(scores_near, switches), "negative_raw_margin": _auroc(scores_margin, switches)}
    for key, value in computed_auroc.items():
        compare_number(value, reference["diagnostics"]["raw_auroc_for_switch"][key], f"diagnostics.raw_auroc_for_switch.{key}", differences)
    computed_holm = analyzer._holm({
        "native_utility": computed_effects["o_minus_p_native_utility"]["two_sided_sign_flip_p"],
        "binary_success": computed_effects["o_minus_p_binary_success"]["two_sided_sign_flip_p"],
    })
    expected_holm = reference["primary"]["pooled"]["holm_adjusted_p"]["O_minus_P_family"]
    for key, value in computed_holm.items():
        compare_number(value, expected_holm[key], f"primary.pooled.holm_adjusted_p.O_minus_P_family.{key}", differences)
    computed_holm_uniform = analyzer._holm({
        "native_utility": computed_effects["p_minus_uniform_native_utility"]["two_sided_sign_flip_p"],
        "binary_success": computed_effects["p_minus_uniform_binary_success"]["two_sided_sign_flip_p"],
    })
    expected_holm_uniform = reference["primary"]["pooled"]["holm_adjusted_p"]["P_minus_uniform_family"]
    for key, value in computed_holm_uniform.items():
        compare_number(value, expected_holm_uniform[key], f"primary.pooled.holm_adjusted_p.P_minus_uniform_family.{key}", differences)
    report = {
        "schema_version": 1, "valid_candidate_groups": len(records), "complete_case_count": len(complete_cases), "complete_case_ids": complete_cases,
        "primary_effects": computed_effects, "fixed_aurocs": computed_auroc, "holm_adjusted_p": {"O_minus_P_family": computed_holm, "P_minus_uniform_family": computed_holm_uniform}, "matches_frozen_analysis": not differences, "differences": differences,
    }
    if differences:
        raise ValueError(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("."))
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.bundle_dir.resolve())
    if args.write_report:
        args.write_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matches_frozen_analysis": report["matches_frozen_analysis"], "complete_case_count": report["complete_case_count"], "fixed_aurocs": report["fixed_aurocs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
