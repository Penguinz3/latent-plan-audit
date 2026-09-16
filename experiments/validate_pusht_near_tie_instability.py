"""Definitive post-hoc validation of a prediction-only near-tie indicator."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .analyze_pusht_margin_instability import (
        DEFAULT_INDEX, auc, cluster_bootstrap, interval, load_rows, mean, percentile, sha256,
    )
except ImportError:  # direct script execution
    from analyze_pusht_margin_instability import (
        DEFAULT_INDEX, auc, cluster_bootstrap, interval, load_rows, mean, percentile, sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = ROOT / "experiments" / "official_pusht_v14_margin_instability_posthoc_amendment_20260904.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "official_pusht_v14_near_tie_validation_20260904.json"


def model_matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["trajectory_id"], row["offset"], row["seed"])].append(row)
    return [row for group in grouped.values() if {item["model"] for item in group} == {"official_jepa_wm", "official_dino_wm"} for row in group]


def average_precision(scores: Iterable[float], labels: Iterable[bool]) -> float | None:
    pairs = sorted(zip(scores, labels), reverse=True)
    positives = sum(bool(label) for _, label in pairs)
    if not positives:
        return None
    found = 0
    total = 0.0
    for rank, (_, label) in enumerate(pairs, 1):
        if label:
            found += 1
            total += found / rank
    return total / positives


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for _ in range(100):
        probability = sigmoid(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        hessian = x.T @ (x * weight[:, None]) + penalty
        gradient = x.T @ (y - probability) - penalty @ beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return beta


def add_oof_predictions(rows: list[dict[str, Any]]) -> None:
    trajectories = sorted({row["trajectory_id"] for row in rows})
    for held_out in trajectories:
        train = [row for row in rows if row["trajectory_id"] != held_out]
        test = [row for row in rows if row["trajectory_id"] == held_out]
        prevalence = float(np.mean([row["switch"] for row in train]))
        model_prevalence = {
            model: float(np.mean([row["switch"] for row in train if row["model"] == model]))
            for model in {row["model"] for row in rows}
        }
        for feature, output in (("near_tie_score", "oof_near_tie_probability"), ("raw_ambiguity", "oof_raw_margin_probability")):
            if feature == "raw_ambiguity":
                for row in train + test:
                    row[feature] = -math.log10(max(row["raw_margin"], 1e-12))
            center = float(np.mean([row[feature] for row in train]))
            scale = float(np.std([row[feature] for row in train])) or 1.0
            x_train = np.asarray([
                [1.0, float(row["model"] == "official_dino_wm"), (row[feature] - center) / scale]
                for row in train
            ])
            beta = fit_logistic(x_train, np.asarray([row["switch"] for row in train], dtype=np.float64))
            for row in test:
                x_test = np.asarray([1.0, float(row["model"] == "official_dino_wm"), (row[feature] - center) / scale])
                row[output] = float(sigmoid(x_test @ beta))
        for row in test:
            row["oof_prevalence_probability"] = prevalence
            row["oof_model_probability"] = model_prevalence[row["model"]]
            threshold = percentile(
                (other["near_tie_score"] for other in train if other["model"] == row["model"]), 0.75
            )
            row["flagged_quarter"] = row["near_tie_score"] >= threshold


def brier(rows: list[dict[str, Any]], prediction: str) -> float:
    return float(np.mean([(row[prediction] - float(row["switch"])) ** 2 for row in rows]))


def log_loss(rows: list[dict[str, Any]], prediction: str) -> float:
    losses = []
    for row in rows:
        probability = min(max(row[prediction], 1e-12), 1.0 - 1e-12)
        target = float(row["switch"])
        losses.append(-(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability)))
    return float(np.mean(losses))


def trajectory_paired_interval(rows: list[dict[str, Any]], left: str, right: str, loss: str, rng: np.random.Generator, n: int) -> list[float]:
    differences = []
    for trajectory in sorted({row["trajectory_id"] for row in rows}):
        subset = [row for row in rows if row["trajectory_id"] == trajectory]
        function = brier if loss == "brier" else log_loss
        differences.append(function(subset, left) - function(subset, right))
    samples = [float(np.mean(rng.choice(differences, size=len(differences), replace=True))) for _ in range(n)]
    return interval(samples) or []


def risk(rows: list[dict[str, Any]], target: str, flagged: bool | None = None) -> float | None:
    selected = [row for row in rows if flagged is None or row["flagged_quarter"] is flagged]
    return mean(float(bool(row[target])) for row in selected)


def risk_ratio(rows: list[dict[str, Any]], target: str = "switch") -> float | None:
    flagged, unflagged = risk(rows, target, True), risk(rows, target, False)
    return None if flagged is None or unflagged in (None, 0.0) else flagged / unflagged


def difference(rows: list[dict[str, Any]], field: str) -> float | None:
    flagged = mean(row[field] for row in rows if row["flagged_quarter"])
    unflagged = mean(row[field] for row in rows if not row["flagged_quarter"])
    return None if flagged is None or unflagged is None else flagged - unflagged


def summarize(rows: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    add_oof_predictions(rows)
    rng = np.random.default_rng(seed)
    score_auc = auc((row["near_tie_score"] for row in rows), (row["switch"] for row in rows))
    auc_ci = interval(cluster_bootstrap(
        rows, lambda sample: auc((row["near_tie_score"] for row in sample), (row["switch"] for row in sample)), rng, resamples
    ))
    ratio = risk_ratio(rows)
    ratio_ci = interval(cluster_bootstrap(rows, risk_ratio, rng, resamples))
    models = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        models[model] = {
            "rows": len(subset),
            "switch_rate": risk(subset, "switch"),
            "near_tie_switch_auc": auc((row["near_tie_score"] for row in subset), (row["switch"] for row in subset)),
            "flagged_quarter_switch_risk_ratio": risk_ratio(subset),
        }
    probability_models = {}
    for name in ("oof_prevalence_probability", "oof_model_probability", "oof_raw_margin_probability", "oof_near_tie_probability"):
        probability_models[name] = {
            "brier": brier(rows, name),
            "log_loss": log_loss(rows, name),
            "auroc": auc((row[name] for row in rows), (row["switch"] for row in rows)),
            "average_precision": average_precision((row[name] for row in rows), (row["switch"] for row in rows)),
        }
    for metric in ("brier", "log_loss"):
        probability_models["oof_near_tie_probability"][f"minus_model_baseline_{metric}_trajectory_bootstrap_95_ci"] = trajectory_paired_interval(
            rows, "oof_near_tie_probability", "oof_model_probability", metric, rng, resamples
        )
        probability_models["oof_near_tie_probability"][f"minus_raw_margin_{metric}_trajectory_bootstrap_95_ci"] = trajectory_paired_interval(
            rows, "oof_near_tie_probability", "oof_raw_margin_probability", metric, rng, resamples
        )
    bounded_auc = auc((row["bounded_near_tie_score"] for row in rows), (row["bounded_switch"] for row in rows))
    physical = {}
    for target, magnitude in (("G_harm", "G_harm_magnitude"), ("J_harm", "J_harm_magnitude")):
        ratio_value = risk_ratio(rows, target)
        magnitude_difference = difference(rows, magnitude)
        physical[target] = {
            "event_count": sum(row[target] for row in rows),
            "flagged_risk": risk(rows, target, True),
            "unflagged_risk": risk(rows, target, False),
            "flagged_over_unflagged_risk_ratio": ratio_value,
            "risk_ratio_trajectory_bootstrap_95_ci": interval(cluster_bootstrap(rows, lambda sample, target=target: risk_ratio(sample, target), rng, resamples)),
            "flagged_minus_unflagged_mean_harm": magnitude_difference,
            "harm_difference_trajectory_bootstrap_95_ci": interval(cluster_bootstrap(rows, lambda sample, magnitude=magnitude: difference(sample, magnitude), rng, resamples)),
        }
    planner_regret = {}
    for target in ("P_G_regret", "P_J_regret"):
        flagged = mean(row[target] for row in rows if row["flagged_quarter"])
        unflagged = mean(row[target] for row in rows if not row["flagged_quarter"])
        planner_regret[target] = {
            "flagged_mean": flagged,
            "unflagged_mean": unflagged,
            "flagged_minus_unflagged": None if flagged is None or unflagged is None else flagged - unflagged,
            "trajectory_bootstrap_95_ci": interval(cluster_bootstrap(rows, lambda sample, target=target: difference(sample, target), rng, resamples)),
        }
    coverages = []
    for target_coverage in (0.25, 0.5, 0.75, 1.0):
        for row in rows:
            train = [other for other in rows if other["trajectory_id"] != row["trajectory_id"] and other["model"] == row["model"]]
            threshold = math.inf if target_coverage == 1.0 else percentile((other["near_tie_score"] for other in train), target_coverage)
            row["temporary_retain"] = row["near_tie_score"] <= threshold
        retained = [row for row in rows if row["temporary_retain"]]
        coverages.append({
            "target_coverage": target_coverage,
            "achieved_coverage": len(retained) / len(rows),
            "switch_risk": risk(retained, "switch"),
            "G_harm_risk": risk(retained, "G_harm"),
            "J_harm_risk": risk(retained, "J_harm"),
            "P_G_regret": mean(row["P_G_regret"] for row in retained),
            "P_J_regret": mean(row["P_J_regret"] for row in retained),
        })
    ordered = sorted(rows, key=lambda row: row["oof_near_tie_probability"])
    calibration = []
    for indices in np.array_split(np.arange(len(ordered)), 5):
        subset = [ordered[int(index)] for index in indices]
        calibration.append({
            "count": len(subset),
            "mean_predicted_switch_probability": mean(row["oof_near_tie_probability"] for row in subset),
            "observed_switch_fraction": risk(subset, "switch"),
        })
    row_predictions = [{
        key: row[key] for key in (
            "decision_id", "model", "trajectory_id", "offset", "seed", "near_tie_score",
            "raw_margin", "normalized_margin", "switch", "flagged_quarter",
            "oof_near_tie_probability", "oof_raw_margin_probability", "oof_model_probability",
            "P_G_regret", "P_J_regret",
        )
    } for row in rows]
    return {
        "rows": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "cells": len(rows) // 2,
        "switches": sum(row["switch"] for row in rows),
        "near_tie_switch_auc": score_auc,
        "near_tie_switch_auc_trajectory_bootstrap_95_ci": auc_ci,
        "flagged_quarter": {
            "achieved_fraction": mean(float(row["flagged_quarter"]) for row in rows),
            "switch_risk": risk(rows, "switch", True),
            "unflagged_switch_risk": risk(rows, "switch", False),
            "switch_risk_ratio": ratio,
            "switch_risk_ratio_trajectory_bootstrap_95_ci": ratio_ci,
        },
        "models": models,
        "probability_models": probability_models,
        "bounded_support": {
            "near_tie_switch_auc": bounded_auc,
            "direction_agrees": bounded_auc is not None and bounded_auc > 0.5,
        },
        "correction_switch_consequence": physical,
        "planner_regret": planner_regret,
        "risk_coverage": coverages,
        "calibration_bins": calibration,
        "row_predictions": row_predictions,
    }


def gate(primary: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    ci = primary["near_tie_switch_auc_trajectory_bootstrap_95_ci"] or [0.0, 0.0]
    ratio_ci = primary["flagged_quarter"]["switch_risk_ratio_trajectory_bootstrap_95_ci"] or [0.0, 0.0]
    near = primary["probability_models"]["oof_near_tie_probability"]
    model = primary["probability_models"]["oof_model_probability"]
    model_aucs = [entry["near_tie_switch_auc"] or 0.0 for entry in primary["models"].values()]
    selector = bool(
        (primary["near_tie_switch_auc"] or 0.0) >= 0.70 and ci[0] > 0.60
        and (primary["flagged_quarter"]["switch_risk_ratio"] or 0.0) >= 1.5 and ratio_ci[0] > 1.0
        and min(model_aucs) > 0.65 and primary["bounded_support"]["direction_agrees"]
        and (original["near_tie_switch_auc"] or 0.0) > 0.5
        and (near["brier"] < model["brier"] or near["log_loss"] < model["log_loss"])
    )
    regret_entries = primary["planner_regret"]
    physical = bool(
        all((entry["flagged_minus_unflagged"] or 0.0) > 0.0 for entry in regret_entries.values())
        and any((entry["trajectory_bootstrap_95_ci"] or [0.0])[0] > 0.0 for entry in regret_entries.values())
    )
    return {
        "selector_instability_passed": selector,
        "physical_risk_passed": physical,
        "headline": (
            "Prediction-only near ties identify selector instability on held-out Push-T trajectories" if selector
            else "Prediction-only near ties do not pass the locked selector-instability gate"
        ),
        "claim_boundary": "No planner-regret, safety, or general uncertainty claim." if not physical else "A scoped planner-regret association passed; no epistemic/aleatoric claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    rows, index = load_rows(args.index)
    primary_rows = model_matched(rows)
    original_rows = model_matched([row for row in rows if row["evidence_source"] == "original_v1.4"])
    if (len(primary_rows), len(original_rows)) != (562, 408):
        raise ValueError(f"unexpected matched populations: {len(primary_rows)}, {len(original_rows)}")
    primary = summarize(primary_rows, args.resamples, 260904)
    original = summarize(original_rows, args.resamples, 260905)
    all_valid = summarize(rows, args.resamples, 260906)
    result = {
        "schema_version": "PWA-PushT-v1.4-near-tie-held-out-validation-v1",
        "status": "complete_posthoc_exploratory",
        "source_bindings": {
            "merge_index": str(args.index.resolve()),
            "merge_index_file_sha256": sha256(args.index),
            "merge_index_declared_sha256": index["merge_sha256"],
            "amendment": str(args.amendment.resolve()),
            "amendment_sha256": sha256(args.amendment),
        },
        "bootstrap_resamples": args.resamples,
        "primary_model_matched": primary,
        "sensitivity_original_only_model_matched": original,
        "secondary_all_merge_valid": all_valid,
        "claim_gate": gate(primary, original),
        "scope": "Post-hoc exploratory validation; features are prediction-only and every fit/threshold excludes the evaluated trajectory.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "claim_gate": result["claim_gate"]}, indent=2))


if __name__ == "__main__":
    main()
