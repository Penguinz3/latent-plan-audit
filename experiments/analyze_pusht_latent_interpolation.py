"""Exploratory latent-correction sweep over the validated PWA-PushT-v1.4 rows.

This is a read-only post-hoc analysis.  It never changes the frozen study.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "experiments" / "official_pusht_v14_recovery_20260831" / "recovery_merge_index.json"
DEFAULT_CACHE = ROOT / "tmp" / "pusht_latent_interpolation_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "official_pusht_v14_latent_interpolation_20260908.json"
LAMBDAS = tuple(round(value / 10, 1) for value in range(11))
MODELS = ("official_jepa_wm", "official_dino_wm")


def strict_choice(values: np.ndarray) -> int | None:
    best = float(np.min(values))
    ties = np.flatnonzero(np.isclose(values, best, atol=1e-10, rtol=1e-8))
    return int(ties[0]) if len(ties) == 1 else None


def _component(predicted: np.ndarray, realized: np.ndarray, goal: np.ndarray, chunk: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = predicted.shape[0]
    observed = np.empty(count, dtype=np.float64)
    directional = np.empty(count, dtype=np.float64)
    magnitude = np.empty(count, dtype=np.float64)
    axes = tuple(range(1, predicted.ndim))
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        prediction = predicted[start:stop].astype(np.float64)
        observation = realized[start:stop].astype(np.float64)
        baseline = observation - goal
        residual = prediction - observation
        observed[start:stop] = np.mean(np.square(baseline), axis=axes)
        directional[start:stop] = 2.0 * np.mean(baseline * residual, axis=axes)
        magnitude[start:stop] = np.mean(np.square(residual), axis=axes)
    return observed, directional, magnitude


def decompose_sidecar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        predicted = archive["initial_predicted_visual"]
        realized = archive["initial_encoded_real_visual"]
        goal = archive["goal_visual"].astype(np.float64)
        observed, directional, magnitude = _component(predicted, realized, goal)
        del predicted, realized, goal

        predicted = archive["initial_predicted_proprio"]
        realized = archive["initial_encoded_real_proprio"]
        goal = archive["goal_proprio"].astype(np.float64)
        proprio = _component(predicted, realized, goal)
        del predicted, realized, goal

    return tuple(left + 0.1 * right for left, right in zip((observed, directional, magnitude), proprio))


def analyze_row(entry: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(Path(entry["artifact_path"]).read_text(encoding="utf-8"))
    candidates = artifact["candidate_rows"]
    observed, directional, magnitude = decompose_sidecar(Path(entry["arrays_path"]))
    reconstructed_p = observed + directional + magnitude
    recorded_p = np.asarray([row["predicted_score"] for row in candidates], dtype=np.float64)
    recorded_o = np.asarray([row["encoded_real_score"] for row in candidates], dtype=np.float64)
    if not np.allclose(reconstructed_p, recorded_p, atol=1e-7, rtol=1e-7) or not np.allclose(observed, recorded_o, atol=1e-7, rtol=1e-7):
        raise ValueError(f"score reconstruction failed for {entry['decision_id']}")
    # Anchor the interpolation exactly to the authoritative serialized P/O
    # endpoints; the sub-1e-7 float32 reconstruction residual is absorbed into
    # the signed cross term.
    directional_adjustment = recorded_p - recorded_o - magnitude - directional
    directional = recorded_p - recorded_o - magnitude

    g = np.asarray([row["G"] for row in candidates], dtype=np.float64)
    j = np.asarray([row["J"] for row in candidates], dtype=np.float64)
    p_choice = strict_choice(recorded_p)
    if p_choice is None or candidates[p_choice]["candidate_id"] != artifact["decision"]["predicted_choice"]:
        raise ValueError(f"P choice reconstruction failed for {entry['decision_id']}")

    selections: dict[str, Any] = {}
    for value in LAMBDAS:
        residual_fraction = 1.0 - value
        score = observed + residual_fraction * directional + residual_fraction**2 * magnitude
        choice = strict_choice(score)
        selections[f"{value:.1f}"] = None if choice is None or p_choice is None else {
            "choice": int(choice),
            "delta_G": float(g[choice] - g[p_choice]),
            "delta_J": float(j[choice] - j[p_choice]),
        }
    if selections["1.0"] is None or candidates[selections["1.0"]["choice"]]["candidate_id"] != artifact["decision"]["encoded_real_choice"]:
        raise ValueError(f"O choice reconstruction failed for {entry['decision_id']}")
    return {
        "decision_id": entry["decision_id"],
        "model": artifact["model"],
        "trajectory_id": artifact["trajectory_id"],
        "offset": int(artifact["offset"]),
        "seed": int(artifact["seed"]),
        "evidence_source": entry["evidence_source"],
        "max_directional_rounding_adjustment": float(np.max(np.abs(directional_adjustment))),
        "selections": selections,
    }


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["decision_id"]: row for row in records}


def _bootstrap(values: np.ndarray, rng: np.random.Generator, resamples: int) -> list[float]:
    means = np.mean(rng.choice(values, size=(resamples, len(values)), replace=True), axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _nice_correlation(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.corrcoef(np.arange(len(array), dtype=np.float64), array)[0, 1])


def summarize(records: list[dict[str, Any]], all_cells: dict[str, set[tuple[int, int]]], resamples: int, seed: int) -> dict[str, Any]:
    by_key = {(row["trajectory_id"], row["offset"], row["seed"], row["model"]): row for row in records}
    trajectories = sorted(all_cells)
    curves: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for value in LAMBDAS:
        label = f"{value:.1f}"
        cluster = {metric: [] for metric in ("delta_G", "delta_J")}
        model_cluster = {model: {metric: [] for metric in cluster} for model in MODELS}
        matched_cells = 0
        switches = 0
        selected_rows = 0
        for trajectory in trajectories:
            accum = {model: {metric: [] for metric in cluster} for model in MODELS}
            for offset, row_seed in sorted(all_cells[trajectory]):
                pair = [by_key.get((trajectory, offset, row_seed, model)) for model in MODELS]
                valid = all(row is not None and row["selections"][label] is not None for row in pair)
                matched_cells += int(valid)
                for model, row in zip(MODELS, pair):
                    selection = row["selections"][label] if valid and row is not None else None
                    if selection is not None:
                        selected_rows += 1
                        switches += int(selection["choice"] != row["selections"]["0.0"]["choice"])
                    for metric in cluster:
                        accum[model][metric].append(float(selection[metric]) if selection is not None else 0.0)
            for model in MODELS:
                for metric in cluster:
                    while len(accum[model][metric]) < 15:
                        accum[model][metric].append(0.0)
                    model_cluster[model][metric].append(float(np.mean(accum[model][metric])))
            for metric in cluster:
                cluster[metric].append(float(np.mean([model_cluster[model][metric][-1] for model in MODELS])))
        curves[label] = {
            "matched_cells": matched_cells,
            "switch_rate": switches / selected_rows if selected_rows else None,
            **{
                metric: {
                    "mean": float(np.mean(values)),
                    "trajectory_bootstrap_95_ci": _bootstrap(np.asarray(values), rng, resamples),
                    "model_specific": {model: float(np.mean(model_cluster[model][metric])) for model in MODELS},
                    "trajectory_values": values,
                }
                for metric, values in cluster.items()
            },
        }

    trend = {
        metric: _nice_correlation(curves[f"{value:.1f}"][metric]["mean"] for value in LAMBDAS)
        for metric in ("delta_G", "delta_J")
    }
    return {"rows": len(records), "trajectories": len(trajectories), "curve": curves, "lambda_effect_correlation": trend}


def gate(primary: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    for metric in ("delta_G", "delta_J"):
        means = [primary["curve"][f"{value:.1f}"][metric]["mean"] for value in LAMBDAS[1:]]
        endpoint = means[-1]
        direction = math.copysign(1.0, endpoint) if endpoint else 0.0
        consistent = sum(math.copysign(1.0, value) == direction for value in means if value) >= 8
        model_endpoints = primary["curve"]["1.0"][metric]["model_specific"].values()
        original_endpoint = original["curve"]["1.0"][metric]["mean"]
        stable_direction = direction != 0 and all(value * direction > 0 for value in model_endpoints) and original_endpoint * direction > 0
        monotone = abs(primary["lambda_effect_correlation"][metric]) >= 0.9
        if consistent and stable_direction and monotone:
            reasons.append(f"stable dose-response in {metric}")
    return {
        "main_paper_worthy": bool(reasons),
        "reasons": reasons,
        "rule": "Include only a same-direction effect at >=8/10 nonzero lambdas, |lambda-effect correlation| >= .9, and matching endpoint direction in both models and original-only sensitivity.",
        "scope": "Exploratory interpolation of stored latent endpoints; no new simulator outcomes and no causal objective-misalignment claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    if index.get("status") != "validated" or index.get("counts", {}).get("valid") != 577:
        raise ValueError("expected validated v1.4 recovery merge index with 577 valid rows")
    entries = [entry for entry in index["rows"] if entry.get("valid") is True]
    cache = _load_cache(args.cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("a", encoding="utf-8") as stream:
        for number, entry in enumerate(entries, 1):
            if entry["decision_id"] in cache:
                continue
            row = analyze_row(entry)
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            cache[row["decision_id"]] = row
            print(f"[{number}/{len(entries)}] {row['decision_id']}", flush=True)

    all_cells: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for entry in index["rows"]:
        match = re.search(r"_pusht_(val_episode_\d+)_offset_(\d+)_seed_(\d+)$", entry["decision_id"])
        if match is None:
            raise ValueError(f"cannot parse issued decision id: {entry['decision_id']}")
        trajectory, offset, row_seed = match.groups()
        all_cells[trajectory].add((int(offset), int(row_seed)))
    records = list(cache.values())
    matched_keys = {
        (row["trajectory_id"], row["offset"], row["seed"])
        for row in records
        if {other["model"] for other in records if (other["trajectory_id"], other["offset"], other["seed"]) == (row["trajectory_id"], row["offset"], row["seed"])} == set(MODELS)
    }
    primary_rows = [row for row in records if (row["trajectory_id"], row["offset"], row["seed"]) in matched_keys]
    original_candidates = [row for row in records if row["evidence_source"] == "original_v1.4"]
    original_keys = {
        (row["trajectory_id"], row["offset"], row["seed"])
        for row in original_candidates
        if {other["model"] for other in original_candidates if (other["trajectory_id"], other["offset"], other["seed"]) == (row["trajectory_id"], row["offset"], row["seed"])} == set(MODELS)
    }
    original_rows = [row for row in original_candidates if (row["trajectory_id"], row["offset"], row["seed"]) in original_keys]
    if (len(primary_rows), len(original_rows)) != (562, 408):
        raise ValueError(f"unexpected matched populations: {len(primary_rows)}, {len(original_rows)}")

    primary = summarize(primary_rows, all_cells, args.resamples, 260908)
    original = summarize(original_rows, all_cells, args.resamples, 260909)
    result = {
        "schema_version": "PWA-PushT-v1.4-latent-interpolation-posthoc-v1",
        "status": "complete_posthoc_exploratory",
        "lambda_definition": "z_lambda=(1-lambda)*predicted+lambda*realized; lambda=0 is P and lambda=1 is O",
        "lambdas": LAMBDAS,
        "source_merge_index": str(args.index.resolve().relative_to(ROOT.resolve())),
        "primary_model_matched": primary,
        "sensitivity_original_only_model_matched": original,
    }
    result["inclusion_gate"] = gate(primary, original)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": result["inclusion_gate"]}, indent=2))


if __name__ == "__main__":
    main()
