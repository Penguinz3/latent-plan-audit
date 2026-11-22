"""Recompute Push-T v1.4 P/O, component, and score-scale results from a portable bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "pusht-mechanism-candidates-v1"
FIELDS = ("predicted_visual", "observed_visual", "predicted_proprio", "observed_proprio")
COMPONENTS = ("agent_position", "block_position", "block_angle_circular", "agent_velocity", "terminal_coverage")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ties(scores: np.ndarray, atol: float, rtol: float) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (300,) or not np.isfinite(scores).all():
        raise ValueError("invalid candidate scores")
    return np.flatnonzero(np.isclose(scores, scores.min(), atol=atol, rtol=rtol)).tolist()


def metrics(states: np.ndarray, goal: np.ndarray, rewards: np.ndarray, native_returns: np.ndarray, index: int) -> dict[str, float]:
    state = states[index]
    angle = float(state[4] - goal[4])
    return {
        "native_G": -float(np.linalg.norm(state - goal)),
        "cumulative_J": float(native_returns[index]),
        "agent_position": -float(np.linalg.norm(state[:2] - goal[:2])),
        "block_position": -float(np.linalg.norm(state[2:4] - goal[2:4])),
        "block_angle_circular": -abs(math.atan2(math.sin(angle), math.cos(angle))),
        "agent_velocity": -float(np.linalg.norm(state[5:7] - goal[5:7])),
        "terminal_coverage": float(rewards[index, -1]),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    for start in range(len(values)):
        if start and values[order[start]] == values[order[start - 1]]:
            continue
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
    return result


def score_scale(p: np.ndarray, o: np.ndarray) -> dict[str, float]:
    a, b = _rank(p), _rank(o)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    error = float(np.median(np.abs(p - o)))
    ip = float(np.quantile(p, .75) - np.quantile(p, .25))
    io = float(np.quantile(o, .75) - np.quantile(o, .25))
    return {
        "spearman_P_O": float(np.dot(a, b) / denominator) if denominator else math.nan,
        "median_abs_P_minus_O": error,
        "IQR_P": ip, "IQR_O": io,
        "median_abs_P_minus_O_over_IQR_P": error / max(ip, 1e-12),
        "median_abs_P_minus_O_over_IQR_O": error / max(io, 1e-12),
    }


def exact_sign_flip(values: list[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    observed, total, exceed = abs(float(data.mean())), 1 << len(data), 0
    bits = np.arange(len(data), dtype=np.uint64)
    for start in range(0, total, 65_536):
        masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bits) & 1).astype(np.float64)
        exceed += int(np.count_nonzero(np.abs((signs @ data) / len(data)) >= observed - 1e-15))
    return float(exceed / total)


def stats(values: list[float], label: str) -> dict:
    data = np.asarray(values, dtype=np.float64)
    seed = int(hashlib.sha256(f"PWA-PushT-v1.4-component:{label}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(10_000, len(data)))].mean(axis=1)
    return {"mean": float(data.mean()), "median": float(np.median(data)), "trajectory_count": len(data),
            "trajectory_bootstrap_95_ci": [float(x) for x in np.quantile(draws, [.025, .975])],
            "two_sided_exact_sign_flip_p": exact_sign_flip(values), "trajectory_values": values}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted((value, key) for key, value in p_values.items())
    result, running = {}, 0.0
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        result[key] = running
    return result


def recompute(bundle: Path, *, verify_reference: bool = True) -> dict:
    rows_path, arrays_path = bundle / "rows.json", bundle / "candidates.npz"
    ledger = json.loads(rows_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != SCHEMA or ledger.get("cost_fields") != list(FIELDS):
        raise ValueError("unsupported bundle schema")
    rows = ledger["rows"]
    if len(rows) != 600 or [r["ordinal"] for r in rows] != list(range(600)):
        raise ValueError("issued-row ledger is incomplete")
    valid = [r for r in rows if r["valid"]]
    if len(valid) != 577 or sum(not r["valid"] for r in rows) != 23:
        raise ValueError("valid/invalid accounting mismatch")
    if [r["array_index"] for r in valid] != list(range(577)):
        raise ValueError("candidate index alignment mismatch")
    if verify_reference:
        binding = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        for name in ("rows.json", "candidates.npz"):
            if sha256(bundle / name) != binding["bundle_sha256"][name]:
                raise ValueError(f"bundle hash mismatch: {name}")
    with np.load(arrays_path, allow_pickle=False) as z:
        names = ("costs", "captured_p_scores", "final_states", "goal_states", "reward_traces", "actions_normalized", "actions_physical", "native_returns", "native_distances", "normalized_action_sha256", "normalized_plan_multiplicity")
        data = {name: z[name] for name in names}
    shapes = ((577, 300, 4), (577, 300), (577, 300, 7), (577, 7), (577, 300, 30), (577, 300, 6, 10), (577, 300, 30, 2), (577, 300), (577, 300), (577, 300), (577, 300))
    for name, shape in zip(names, shapes):
        if data[name].shape != shape:
            raise ValueError(f"invalid candidate array {name}")
        if name == "normalized_action_sha256":
            if data[name].dtype.kind != "S" or data[name].dtype.itemsize != 64:
                raise ValueError("invalid action hash array")
        elif not np.issubdtype(data[name].dtype, np.number) or not np.isfinite(data[name]).all():
            raise ValueError(f"invalid candidate array {name}")
    records = []
    for r in valid:
        i = r["array_index"]
        costs = data["costs"][i]
        p = data["captured_p_scores"][i]
        o = costs[:, 1] + ledger["weight"] * costs[:, 3]
        reconstructed_p = costs[:, 0] + ledger["weight"] * costs[:, 2]
        if np.max(np.abs(p - reconstructed_p)) > ledger["score_tolerance"]:
            raise ValueError(f"captured P score mismatch: {r['decision_id']}")
        pt, ot = ties(p, ledger["tie_atol"], ledger["tie_rtol"]), ties(o, ledger["tie_atol"], ledger["tie_rtol"])
        if pt != r["p_tie_indices"] or ot != r["o_tie_indices"] or pt != [r["p_index"]] or ot != [r["o_index"]]:
            raise ValueError(f"selector mismatch: {r['decision_id']}")
        states, goal, rewards = data["final_states"][i], data["goal_states"][i], data["reward_traces"][i]
        if not np.allclose(data["native_distances"][i], np.linalg.norm(states - goal, axis=1), atol=1e-12, rtol=0):
            raise ValueError(f"native G mismatch: {r['decision_id']}")
        if not np.array_equal(data["native_returns"][i], np.add.accumulate(rewards, axis=1, dtype=np.float32)[:, -1].astype(np.float64)):
            raise ValueError(f"native J mismatch: {r['decision_id']}")
        hashes = np.asarray([hashlib.sha256(action.tobytes()).hexdigest().encode("ascii") for action in data["actions_normalized"][i]], dtype="S64")
        if not np.array_equal(hashes, data["normalized_action_sha256"][i]):
            raise ValueError(f"action hash mismatch: {r['decision_id']}")
        _, inverse, multiplicities = np.unique(hashes, return_inverse=True, return_counts=True)
        if not np.array_equal(multiplicities[inverse], data["normalized_plan_multiplicity"][i]):
            raise ValueError(f"action multiplicity mismatch: {r['decision_id']}")
        native_returns = data["native_returns"][i]
        pm, om = metrics(states, goal, rewards, native_returns, r["p_index"]), metrics(states, goal, rewards, native_returns, r["o_index"])
        pd, od = states[r["p_index"]] - goal, states[r["o_index"]] - goal
        sq = {"agent_position_sq": float(np.dot(pd[:2], pd[:2]) - np.dot(od[:2], od[:2])),
              "block_position_sq": float(np.dot(pd[2:4], pd[2:4]) - np.dot(od[2:4], od[2:4])),
              "block_angle_raw_sq": float(pd[4] ** 2 - od[4] ** 2),
              "agent_velocity_sq": float(np.dot(pd[5:7], pd[5:7]) - np.dot(od[5:7], od[5:7]))}
        if abs(sum(sq.values()) - (data["native_distances"][i, r["p_index"]] ** 2 - data["native_distances"][i, r["o_index"]] ** 2)) > 1e-7:
            raise ValueError(f"squared-distance identity mismatch: {r['decision_id']}")
        records.append({**{k: r[k] for k in ("trajectory_id", "offset", "menu_seed", "model")},
                        "effect": {key: om[key] - pm[key] for key in pm}, "score_scale": score_scale(p, o), "squared": sq,
                        "switch": r["p_index"] != r["o_index"]})
    cells = {}
    for r in records:
        cells.setdefault((r["trajectory_id"], r["offset"], r["menu_seed"]), []).append(r)
    matched = [pair for pair in cells.values() if len(pair) == 2 and {r["model"] for r in pair} == {"official_jepa_wm", "official_dino_wm"}]
    if len(matched) != 281 or len({pair[0]["trajectory_id"] for pair in matched}) != 20:
        raise ValueError("matched-cell population mismatch")
    trajectories = sorted({pair[0]["trajectory_id"] for pair in matched})
    effect_keys = tuple(records[0]["effect"])
    scale_keys = tuple(records[0]["score_scale"])
    effects = {key: [] for key in effect_keys}
    scales = {key: [] for key in scale_keys}
    squared = {key: [] for key in records[0]["squared"]}
    for trajectory in trajectories:
        selected = [pair for pair in matched if pair[0]["trajectory_id"] == trajectory]
        for key in effect_keys:
            effects[key].append(float(sum(np.mean([r["effect"][key] for r in pair]) for pair in selected) / 15))
        for key in squared:
            squared[key].append(float(sum(np.mean([r["squared"][key] for r in pair]) for pair in selected) / 15))
        flat = [r for pair in selected for r in pair]
        for key in scale_keys:
            scales[key].append(float(np.mean([r["score_scale"][key] for r in flat])))
    result = {
        "counts": {"issued": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid),
                   "matched_rows": 2 * len(matched), "matched_cells": len(matched), "trajectory_clusters": len(trajectories),
                   "unmatched_valid_rows": len(valid) - 2 * len(matched), "switches_matched_rows": sum(r["switch"] for pair in matched for r in pair)},
        "effect_trajectory_values": effects, "score_scale_trajectory_values": scales,
        "squared_trajectory_values": squared,
        "effect_means": {key: float(np.mean(values)) for key, values in effects.items()},
        "score_scale_means": {key: float(np.mean(values)) for key, values in scales.items()},
        "component_effects": {key: stats(effects[key], key) for key in COMPONENTS},
        "primary_effects": {key: stats(effects[key], f"{key}_replication") for key in ("native_G", "cumulative_J")},
        "score_scale_diagnostics": {key: stats(values, f"score_{key}") for key, values in scales.items()},
        "native_squared_decomposition": {key: stats(values, f"sq_{key}") for key, values in squared.items()},
    }
    adjusted = holm({key: result["component_effects"][key]["two_sided_exact_sign_flip_p"] for key in COMPONENTS})
    for key in COMPONENTS:
        result["component_effects"][key]["holm_adjusted_p"] = adjusted[key]
    if verify_reference:
        reference = json.loads((bundle / "reference.json").read_text(encoding="utf-8"))
        sealed = reference["sealed_raw_inventory_means"]
        if abs(result["effect_means"]["native_G"] - sealed["Delta_G"]) > 1e-7 or abs(result["effect_means"]["cumulative_J"] - sealed["Delta_J"]) > 1e-5:
            raise ValueError("sealed primary G/J mismatch")
        if result["counts"] != reference["counts"]:
            raise ValueError("reference counts mismatch")
        for section in ("effect_trajectory_values", "score_scale_trajectory_values", "squared_trajectory_values"):
            for key, values in result[section].items():
                if not np.allclose(values, reference[section][key], atol=1e-10, rtol=0):
                    raise ValueError(f"reference mismatch: {section}.{key}")
        for section in ("component_effects", "primary_effects", "score_scale_diagnostics", "native_squared_decomposition"):
            for key, values in result[section].items():
                for field, value in values.items():
                    if not np.allclose(value, reference[section][key][field], atol=1e-10, rtol=0):
                        raise ValueError(f"reference mismatch: {section}.{key}.{field}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/pusht_mechanism_20260929/bundle")
    args = parser.parse_args()
    print(json.dumps(recompute(args.bundle_dir), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
