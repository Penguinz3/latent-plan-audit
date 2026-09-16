"""Post-hoc, trajectory-held-out decision-margin analysis for PWA-PushT-v1.4.

This reads frozen artifacts through the validated recovery merge index and writes
new analysis outputs.  It never edits the issued study, ledger, or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "experiments" / "official_pusht_v14_recovery_20260831" / "recovery_merge_index.json"
DEFAULT_SPEC = ROOT / "experiments" / "official_pusht_v14_margin_instability_posthoc_spec_20260904.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "official_pusht_v14_margin_instability_posthoc_20260904.json"
PREDICTORS = ("confidence", "raw_margin", "relative_margin", "best_score_confidence", "softmax_max", "negative_entropy", "score_spread")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=np.float64), q))


def auc(scores: Iterable[float], labels: Iterable[bool]) -> float | None:
    """Tie-aware Mann-Whitney AUROC; larger score predicts True."""
    x = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=bool)
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def distinct_action_features(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_action: dict[str, float] = {}
    errors: list[float] = []
    for index, candidate in enumerate(candidates):
        predicted = finite(candidate.get("predicted_score"))
        realized = finite(candidate.get("encoded_real_score"))
        if predicted is None or realized is None:
            continue
        key = str(candidate.get("physical_action_sha256") or candidate.get("normalized_action_sha256") or index)
        best_by_action[key] = min(predicted, best_by_action.get(key, math.inf))
        errors.append(abs(realized - predicted))
    scores = np.sort(np.asarray(list(best_by_action.values()), dtype=np.float64))
    if len(scores) < 2 or not errors:
        raise ValueError("row lacks two distinct finite actions or finite score errors")
    spread = float(np.quantile(scores, 0.9) - np.quantile(scores, 0.1))
    iqr = float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25))
    margin = float(scores[1] - scores[0])
    normalized_margin = margin / max(iqr, 1e-12)
    return {
        "distinct_actions": int(len(scores)),
        "raw_margin": margin,
        "relative_margin": margin / spread if spread > 0 else None,
        "normalized_margin": normalized_margin,
        "near_tie_score": -math.log10(max(normalized_margin, 1e-12)),
        "best_score": float(scores[0]),
        "score_spread": spread,
        "score_iqr": iqr,
        "scores": scores,
        "absolute_errors": np.asarray(errors, dtype=np.float64),
    }


def reversal(pair: Any) -> bool:
    if not isinstance(pair, dict):
        return False
    predicted = finite(pair.get("P_margin_selected_minus_counterfactual"))
    realized = finite(pair.get("O_margin_selected_minus_counterfactual"))
    return bool(predicted is not None and realized is not None and predicted < 0 < realized)


def load_rows(index_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "validated" or len(index.get("rows", [])) != 600:
        raise ValueError("expected the validated 600-row v1.4 recovery merge index")
    rows: list[dict[str, Any]] = []
    for entry in index["rows"]:
        if entry.get("valid") is not True:
            continue
        path = Path(entry["artifact_path"])
        artifact = json.loads(path.read_text(encoding="utf-8"))
        candidates = artifact.get("candidate_rows")
        if not isinstance(candidates, list) or len(candidates) != 300:
            raise ValueError(f"{path} does not contain exactly 300 candidate rows")
        features = distinct_action_features(candidates)
        decision = artifact.get("decision", {})
        pairs = artifact.get("consequential_pair_decomposition", {})
        metrics = artifact.get("standard_outer_metrics", {}).get("normalized_regret_at_1", {})
        predicted_choice = decision.get("predicted_choice")
        realized_choice = decision.get("encoded_real_choice")
        if predicted_choice is None or realized_choice is None:
            raise ValueError(f"{path} lacks finite P/O choices")
        candidate_by_id = {candidate.get("candidate_id"): candidate for candidate in candidates}
        predicted_candidate = candidate_by_id[predicted_choice]
        realized_candidate = candidate_by_id[realized_choice]
        p_g, o_g = finite(predicted_candidate.get("G")), finite(realized_candidate.get("G"))
        p_j, o_j = finite(predicted_candidate.get("J")), finite(realized_candidate.get("J"))
        if None in (p_g, o_g, p_j, o_j):
            raise ValueError(f"{path} lacks finite selected physical outcomes")
        bounded_candidates = []
        for candidate in candidates:
            action = np.asarray(candidate.get("action_model"), dtype=np.float64)
            if action.size and np.isfinite(action).all() and float(np.max(np.abs(action))) <= 3.0:
                bounded_candidates.append(candidate)
        bounded = distinct_action_features(bounded_candidates)
        bounded_recovery = artifact.get("standard_outer_metrics", {}).get("action_only_subset", {}).get("matched_recovery", {})
        bounded_p = bounded_recovery.get("predicted_choice_index")
        bounded_o = bounded_recovery.get("realized_latent_choice_index")
        rows.append({
            "decision_id": artifact["decision_id"],
            "model": artifact["model"],
            "trajectory_id": artifact["trajectory_id"],
            "offset": artifact["offset"],
            "seed": artifact["seed"],
            "evidence_source": entry["evidence_source"],
            "original_terminal_status": entry.get("original_terminal_status"),
            "distinct_actions": features["distinct_actions"],
            "raw_margin": features["raw_margin"],
            "relative_margin": features["relative_margin"],
            "normalized_margin": features["normalized_margin"],
            "near_tie_score": features["near_tie_score"],
            "best_score": features["best_score"],
            "score_spread": features["score_spread"],
            "score_iqr": features["score_iqr"],
            "scores": features["scores"],
            "absolute_errors": features["absolute_errors"],
            "stable": predicted_choice == realized_choice,
            "switch": predicted_choice != realized_choice,
            "G_reversal": reversal(pairs.get("P_vs_finite_menu_best_G")),
            "J_reversal": reversal(pairs.get("P_vs_finite_menu_best_J")),
            "P_G_regret": finite(metrics.get("P_G")),
            "P_J_regret": finite(metrics.get("P_J")),
            "G_harm": bool(predicted_choice != realized_choice and o_g < p_g - 1e-10),
            "J_harm": bool(predicted_choice != realized_choice and o_j < p_j - 1e-10),
            "G_harm_magnitude": max(0.0, p_g - o_g),
            "J_harm_magnitude": max(0.0, p_j - o_j),
            "bounded_near_tie_score": bounded["near_tie_score"],
            "bounded_switch": bool(bounded_p is not None and bounded_o is not None and bounded_p != bounded_o),
        })
    if len(rows) != int(index["counts"]["valid"]):
        raise ValueError("valid-row count disagrees with merge index")
    return rows, index


def error_scale(errors: np.ndarray) -> float:
    scale = float(np.median(errors)) if len(errors) else 0.0
    if scale <= 0:
        scale = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0
    if scale <= 0:
        raise ValueError("non-positive held-out historical error scale")
    return scale


def add_crossfit_predictors(rows: list[dict[str, Any]]) -> None:
    errors: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        errors[(row["model"], row["trajectory_id"])].append(row["absolute_errors"])
    scales: dict[tuple[str, str], float] = {}
    for model, trajectory in {(r["model"], r["trajectory_id"]) for r in rows}:
        training = [arrays for (m, t), parts in errors.items() if m == model and t != trajectory for arrays in parts]
        scales[(model, trajectory)] = error_scale(np.concatenate(training))
    for row in rows:
        scale = scales[(row["model"], row["trajectory_id"])]
        logits = -(row["scores"] - row["scores"].min()) / scale
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        row["historical_error_scale"] = scale
        row["confidence"] = row["raw_margin"] / scale
        row["best_score_confidence"] = -row["best_score"]
        row["softmax_max"] = float(probabilities.max())
        row["negative_entropy"] = float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))


def add_crossfit_retain(rows: list[dict[str, Any]], coverages: tuple[float, ...]) -> None:
    for row in rows:
        for predictor in PREDICTORS:
            training = [
                float(other[predictor]) for other in rows
                if other["model"] == row["model"] and other["trajectory_id"] != row["trajectory_id"]
                and finite(other.get(predictor)) is not None
            ]
            for coverage in coverages:
                threshold = percentile(training, 1.0 - coverage)
                row[f"retain_{predictor}_{coverage:.2f}"] = finite(row.get(predictor)) is not None and row[predictor] >= threshold


def mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def rate(rows: list[dict[str, Any]], field: str, retained: str | None = None) -> float | None:
    selected = [row for row in rows if retained is None or row[retained]]
    return mean(float(bool(row[field])) for row in selected)


def selective_mean(rows: list[dict[str, Any]], field: str, retained: str | None = None) -> float | None:
    selected = [row for row in rows if retained is None or row[retained]]
    return mean(row[field] for row in selected)


def cluster_bootstrap(rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None], rng: np.random.Generator, n: int) -> list[float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    clusters = sorted(grouped)
    estimates: list[float] = []
    for _ in range(n):
        sample = [row for cluster in rng.choice(clusters, size=len(clusters), replace=True) for row in grouped[str(cluster)]]
        estimate = statistic(sample)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    return estimates


def interval(samples: list[float]) -> list[float] | None:
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])] if samples else None


def auc_for(rows: list[dict[str, Any]], predictor: str, target: str, low_predicts_event: bool = False) -> float | None:
    sign = -1.0 if low_predicts_event else 1.0
    return auc((sign * float(row[predictor]) for row in rows), (bool(row[target]) for row in rows))


def summarize_population(rows: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    coverages = tuple(float(value) for value in np.linspace(0.1, 1.0, 10))
    add_crossfit_predictors(rows)
    add_crossfit_retain(rows, coverages)
    rng = np.random.default_rng(seed)
    predictor_summaries: dict[str, Any] = {}
    for predictor in PREDICTORS:
        retain = f"retain_{predictor}_0.50"
        stability_auc = auc_for(rows, predictor, "stable")
        risk_difference = (rate(rows, "switch", retain) or 0.0) - (rate(rows, "switch") or 0.0)
        predictor_summaries[predictor] = {
            "stability_auc": stability_auc,
            "crossfit_50_percent": {
                "achieved_coverage": mean(float(row[retain]) for row in rows),
                "switch_risk": rate(rows, "switch", retain),
                "all_rows_switch_risk": rate(rows, "switch"),
                "retained_minus_all_switch_risk": risk_difference,
            },
        }
    primary = predictor_summaries["confidence"]
    primary["stability_auc_trajectory_bootstrap_95_ci"] = interval(cluster_bootstrap(
        rows, lambda sample: auc_for(sample, "confidence", "stable"), rng, resamples
    ))
    primary["crossfit_50_percent"]["retained_minus_all_trajectory_bootstrap_95_ci"] = interval(cluster_bootstrap(
        rows,
        lambda sample: (rate(sample, "switch", "retain_confidence_0.50") or 0.0) - (rate(sample, "switch") or 0.0),
        rng,
        resamples,
    ))
    secondary: dict[str, Any] = {}
    for target in ("G_reversal", "J_reversal"):
        point = auc_for(rows, "confidence", target, low_predicts_event=True)
        secondary[target] = {
            "event_count": sum(bool(row[target]) for row in rows),
            "low_confidence_event_auc": point,
            "trajectory_bootstrap_95_ci": interval(cluster_bootstrap(
                rows, lambda sample, target=target: auc_for(sample, "confidence", target, low_predicts_event=True), rng, resamples
            )),
            "all_rows_risk": rate(rows, target),
            "crossfit_50_percent_risk": rate(rows, target, "retain_confidence_0.50"),
        }
    for target in ("P_G_regret", "P_J_regret"):
        all_mean = selective_mean(rows, target)
        kept_mean = selective_mean(rows, target, "retain_confidence_0.50")
        secondary[target] = {
            "all_rows_mean": all_mean,
            "crossfit_50_percent_mean": kept_mean,
            "retained_minus_all": None if all_mean is None or kept_mean is None else kept_mean - all_mean,
        }
    curve = []
    for coverage in coverages:
        retained = f"retain_confidence_{coverage:.2f}"
        curve.append({
            "target_coverage": coverage,
            "achieved_coverage": mean(float(row[retained]) for row in rows),
            "switch_risk": rate(rows, "switch", retained),
            "G_reversal_risk": rate(rows, "G_reversal", retained),
            "J_reversal_risk": rate(rows, "J_reversal", retained),
            "P_G_regret": selective_mean(rows, "P_G_regret", retained),
            "P_J_regret": selective_mean(rows, "P_J_regret", retained),
        })
    compact_rows = [{key: value for key, value in row.items() if key not in {"scores", "absolute_errors"} and not key.startswith("retain_")} for row in rows]
    return {
        "row_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "model_counts": {model: sum(row["model"] == model for row in rows) for model in sorted({row["model"] for row in rows})},
        "switch_count": sum(row["switch"] for row in rows),
        "predictors": predictor_summaries,
        "secondary": secondary,
        "primary_risk_coverage_curve": curve,
        "rows": compact_rows,
    }


def decide_claim(primary: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    full = primary["predictors"]["confidence"]
    sensitivity = original["predictors"]["confidence"]
    auc_ci = full["stability_auc_trajectory_bootstrap_95_ci"]
    risk_ci = full["crossfit_50_percent"]["retained_minus_all_trajectory_bootstrap_95_ci"]
    actionable = bool(
        auc_ci and auc_ci[0] > 0.5 and risk_ci and risk_ci[1] < 0
        and (sensitivity["stability_auc"] or 0.0) > 0.5
        and sensitivity["crossfit_50_percent"]["retained_minus_all_switch_risk"] < 0
    )
    reversals = primary["secondary"]
    physical = bool(
        (reversals["G_reversal"]["low_confidence_event_auc"] or 0.0) > 0.5
        and (reversals["J_reversal"]["low_confidence_event_auc"] or 0.0) > 0.5
        and any((reversals[name]["trajectory_bootstrap_95_ci"] or [0.0])[0] > 0.5 for name in ("G_reversal", "J_reversal"))
        and all(reversals[name]["crossfit_50_percent_risk"] <= reversals[name]["all_rows_risk"] for name in ("G_reversal", "J_reversal"))
    )
    return {
        "actionable_decision_confidence_passed": actionable,
        "physical_risk_passed": physical,
        "allowed_headline": (
            "Held-out margin-based confidence identifies ranking-stable decisions" if actionable
            else "Top-two latent-score margins did not support an actionable held-out confidence claim"
        ),
        "terminology": "margin-based decision confidence" if actionable else "descriptive decision-margin diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=None, help="Override only for tests; committed result uses the spec value.")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    resamples = int(args.resamples or spec["randomness"]["bootstrap_resamples"])
    rows, index = load_rows(args.index)
    primary = summarize_population(rows, resamples, int(spec["randomness"]["bootstrap_seed"]))
    original_rows = [row for row in rows if row["evidence_source"] == "original_v1.4"]
    original = summarize_population(original_rows, resamples, int(spec["randomness"]["bootstrap_seed"]) + 1)
    result = {
        "schema_version": "PWA-PushT-v1.4-margin-instability-posthoc-result-v1",
        "status": "complete",
        "scope": spec["scope"],
        "source_bindings": {
            "merge_index": str(args.index.resolve()),
            "merge_index_sha256": sha256(args.index),
            "merge_index_declared_sha256": index["merge_sha256"],
            "spec": str(args.spec.resolve()),
            "spec_sha256": sha256(args.spec),
        },
        "bootstrap_resamples": resamples,
        "primary_all_merge_valid": primary,
        "sensitivity_original_completions_only": original,
        "claim_gate": decide_claim(primary, original),
        "limitations": [
            "Post-hoc analysis, not part of the original preregistration.",
            "The confidence score is a decision-margin heuristic, not epistemic or aleatoric uncertainty.",
            "Only Push-T, two predictors with shared DINOv2 visual features, and fixed iteration-0 menus are evaluated.",
            "No ensemble-disagreement or out-of-distribution baseline is available in the frozen artifacts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rows": primary["row_count"],
        "original_rows": original["row_count"],
        "claim_gate": result["claim_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
