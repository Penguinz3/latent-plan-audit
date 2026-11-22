"""Exploratory last-step reward sensitivity for the saved Push-T v1.4 evidence.

This is deliberately a small consumer of the sealed merge index.  It reads
only ``initial_reward_traces`` from each NPZ and takes the last environment
step; it never loads the large latent or observation arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

MODELS = ("official_jepa_wm", "official_dino_wm")
FIXED_CELLS = 15
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def terminal_reward_from_trace(trace: np.ndarray) -> np.ndarray:
    """Return one last-step reward per candidate, preserving missingness."""
    values = np.asarray(trace)
    if values.shape != (300, 30):
        raise ValueError(f"expected initial_reward_traces shape (300, 30), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("initial_reward_traces contains non-finite values")
    return np.asarray(values[:, -1], dtype=np.float64)


def _choice_index(decision: Mapping[str, Any], candidates: list[Mapping[str, Any]], key: str, tie_key: str) -> int:
    selected = decision.get(key)
    ties = decision.get(tie_key)
    if not isinstance(selected, str) or not isinstance(ties, list) or len(ties) != 1 or ties[0] != selected:
        raise ValueError(f"{key} is absent or not a singleton JSON selection")
    ids = [row.get("candidate_id") for row in candidates]
    try:
        index = ids.index(selected)
    except ValueError as exc:
        raise ValueError(f"{key}={selected!r} is not in candidate_rows") from exc
    return index


def load_records(merge: Mapping[str, Any]) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], dict[str, Any]]:
    """Read selected IDs from JSON and last-step rewards from each valid NPZ."""
    records: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {"max_cumulative_J_reconciliation_error": 0.0, "rows_read": 0}
    for merged in merge.get("rows", []):
        if not merged.get("valid"):
            continue
        try:
            artifact_path = Path(str(merged["artifact_path"]))
            arrays_path = Path(str(merged["arrays_path"]))
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            decision = artifact["decision"]
            candidates = artifact["candidate_rows"]
            p_index = _choice_index(decision, candidates, "predicted_choice", "predicted_tie_candidate_ids")
            o_index = _choice_index(decision, candidates, "encoded_real_choice", "encoded_real_tie_candidate_ids")
            with np.load(arrays_path, allow_pickle=False) as archive:
                if artifact.get("physical_replay_steps") != 30:
                    raise ValueError("physical_replay_steps is not the official 30-step replay")
                trace = np.asarray(archive["initial_reward_traces"])
                terminal = terminal_reward_from_trace(trace)
                json_j = np.asarray([float(row["J"]) for row in candidates], dtype=np.float64)
                cumulative_j = np.asarray(np.add.accumulate(trace, axis=1, dtype=np.float32)[:, -1], dtype=np.float64)
                if json_j.shape != cumulative_j.shape:
                    raise ValueError("candidate_rows J count does not match reward trace candidates")
                j_error = float(np.max(np.abs(json_j - cumulative_j)))
                diagnostics["max_cumulative_J_reconciliation_error"] = max(diagnostics["max_cumulative_J_reconciliation_error"], j_error)
            key = (str(artifact["trajectory_id"]), int(artifact["offset"]), int(artifact["seed"]), str(artifact["model"]))
            if key in records:
                raise ValueError("duplicate valid row for cell/model")
            records[key] = {
                "decision_id": str(artifact["decision_id"]),
                "trajectory_id": key[0], "offset": key[1], "seed": key[2], "model": key[3],
                "p_index": p_index, "o_index": o_index,
                "p_choice_id": str(decision["predicted_choice"]),
                "o_choice_id": str(decision["encoded_real_choice"]),
                "p_json_score": float(candidates[p_index]["predicted_score"]),
                "o_json_score": float(candidates[o_index]["encoded_real_score"]),
                "p_terminal_reward": float(terminal[p_index]),
                "o_terminal_reward": float(terminal[o_index]),
                "uniform_terminal_reward": float(np.mean(terminal)),
                "o_minus_p_terminal_reward": float(terminal[o_index] - terminal[p_index]),
                "p_minus_uniform_terminal_reward": float(terminal[p_index] - np.mean(terminal)),
                "arrays_sha256": merged.get("arrays_sha256"),
                "artifact_sha256": merged.get("artifact_sha256"),
            }
            diagnostics["rows_read"] += 1
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"valid merged row could not be consumed: {merged.get('decision_id')}: {exc}") from exc
    return records, diagnostics


def _sign_flip(values: Iterable[float]) -> float | None:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        return None
    observed = abs(float(np.mean(data)))
    total = 1 << len(data)
    bits = np.arange(len(data), dtype=np.uint64)
    exceed = 0
    for start in range(0, total, 65_536):
        masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bits) & 1).astype(np.float64)
        exceed += int(np.count_nonzero(np.abs((signs @ data) / len(data)) >= observed - 1e-15))
    return float(exceed / total)


def _bootstrap(values: list[float], label: str) -> list[float] | None:
    if not values:
        return None
    seed = int(hashlib.sha256(f"PushT-terminal-reward:{label}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    draws = data[rng.integers(0, len(data), size=(BOOTSTRAP_REPLICATES, len(data)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def stats(values: list[float], label: str) -> dict[str, Any]:
    clean = [float(x) for x in values if _finite(x)]
    return {
        "mean": float(np.mean(clean)) if clean else None,
        "median": float(np.median(clean)) if clean else None,
        "sd": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0 if clean else None,
        "trajectory_count": len(clean),
        "trajectory_bootstrap_95_ci": _bootstrap(clean, label),
        "two_sided_exact_sign_flip_p": _sign_flip(clean),
        "trajectory_values": clean,
    }


def holm(p_values: Mapping[str, Any]) -> dict[str, float | None]:
    valid = sorted((float(value), key) for key, value in p_values.items() if _finite(value))
    result: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    for rank, (value, key) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        result[key] = running
    return result


def aggregate(
    records: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    cells_by_trajectory: Mapping[str, list[tuple[int, int]]],
    *, fixed_cells: int = FIXED_CELLS,
) -> dict[str, Any]:
    """Apply the v1.4 paired mask and fixed denominator to terminal contrasts."""
    trajectories = sorted(cells_by_trajectory)
    contrasts = ("o_minus_p_terminal_reward", "p_minus_uniform_terminal_reward")
    per_model: dict[str, dict[str, list[float]]] = {m: {c: [] for c in contrasts} for m in MODELS}
    per_trajectory: dict[str, Any] = {}
    matched_counts: dict[str, int] = {}
    matched_cell_keys: list[tuple[str, int, int]] = []
    for trajectory in trajectories:
        cells = cells_by_trajectory[trajectory]
        values: dict[str, dict[str, list[float]]] = {m: {c: [] for c in contrasts} for m in MODELS}
        matched = 0
        for offset, seed in cells:
            pair = [records.get((trajectory, offset, seed, model)) for model in MODELS]
            is_matched = all(row is not None for row in pair)
            if is_matched:
                matched += 1
                matched_cell_keys.append((trajectory, offset, seed))
            for model, row in zip(MODELS, pair):
                for contrast in contrasts:
                    values[model][contrast].append(float(row[contrast]) if is_matched and row is not None else 0.0)
        # Canonical v1.4 has 15 cells. Keep this guard for synthetic tests and
        # preserve a fixed denominator when a damaged input omits whole cells.
        for model in MODELS:
            for contrast in contrasts:
                values[model][contrast].extend([0.0] * max(0, fixed_cells - len(cells)))
        model_means = {m: {c: float(np.mean(values[m][c])) for c in contrasts} for m in MODELS}
        for model in MODELS:
            for contrast in contrasts:
                per_model[model][contrast].append(model_means[model][contrast])
        per_trajectory[trajectory] = {
            "matched_cells": matched,
            "cells": len(cells),
            "model_means": model_means,
            "pooled_mean": {c: float(np.mean([model_means[m][c] for m in MODELS])) for c in contrasts},
        }
        matched_counts[trajectory] = matched

    pooled = {
        c: stats([per_trajectory[t]["pooled_mean"][c] for t in trajectories], f"pooled:{c}")
        for c in contrasts
    }
    model_summary = {
        model: {c: stats(per_model[model][c], f"{model}:{c}") for c in contrasts}
        for model in MODELS
    }
    exploratory_holm = {
        "o_minus_p_terminal_reward": holm({model: model_summary[model]["o_minus_p_terminal_reward"]["two_sided_exact_sign_flip_p"] for model in MODELS}),
        "p_minus_uniform_terminal_reward": holm({model: model_summary[model]["p_minus_uniform_terminal_reward"]["two_sided_exact_sign_flip_p"] for model in MODELS}),
    }
    complete = [t for t in trajectories if matched_counts[t] == fixed_cells]
    complete_summary = {
        "trajectory_ids": complete,
        "trajectory_count": len(complete),
        "model_specific": {
            model: {
                c: stats([per_trajectory[t]["model_means"][model][c] for t in complete], f"complete:{model}:{c}")
                for c in contrasts
            }
            for model in MODELS
        },
        "pooled": {
            c: stats([per_trajectory[t]["pooled_mean"][c] for t in complete], f"complete:pooled:{c}")
            for c in contrasts
        },
    }
    complete_summary["exploratory_holm_adjusted_p_by_contrast"] = {
        "o_minus_p_terminal_reward": holm({model: complete_summary["model_specific"][model]["o_minus_p_terminal_reward"]["two_sided_exact_sign_flip_p"] for model in MODELS}),
        "p_minus_uniform_terminal_reward": holm({model: complete_summary["model_specific"][model]["p_minus_uniform_terminal_reward"]["two_sided_exact_sign_flip_p"] for model in MODELS}),
    }
    return {
        "pooled": pooled,
        "model_specific": model_summary,
        "trajectory_level": per_trajectory,
        "matched_cell_counts": matched_counts,
        "matched_cell_keys": [list(k) for k in matched_cell_keys],
        "complete_valid_sensitivity": complete_summary,
        "exploratory_holm_adjusted_p_by_contrast": exploratory_holm,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    miss = result["missingness"]
    pooled = result["primary_fixed_15_cell_sensitivity"]["pooled"]
    models = result["primary_fixed_15_cell_sensitivity"]["model_specific"]
    complete = result["primary_fixed_15_cell_sensitivity"]["complete_valid_sensitivity"]
    holm = result["primary_fixed_15_cell_sensitivity"]["exploratory_holm_adjusted_p_by_contrast"]
    def fmt(name: str, item: Mapping[str, Any]) -> str:
        ci = item.get("trajectory_bootstrap_95_ci")
        return f"{name}: mean {item.get('mean'):.6g}, 95% CI [{ci[0]:.6g}, {ci[1]:.6g}], exact sign-flip p={item.get('two_sided_exact_sign_flip_p'):.6g}" if ci else f"{name}: unavailable"
    def fmt_desc(name: str, item: Mapping[str, Any]) -> str:
        ci = item.get("trajectory_bootstrap_95_ci")
        return f"{name}: mean {item.get('mean'):.6g}, 95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci else f"{name}: unavailable"
    lines = [
        "# Push-T v1.4 terminal-reward sensitivity",
        "",
        "This posthoc exploratory analysis reads only each retained NPZ's `initial_reward_traces[:, -1]`, the last official environment reward in the 30-step replay. It uses the JSON P/O selected IDs and does not recompute or replace the preregistered P/O results.",
        "",
        f"The saved merge contains {miss['valid_merge_rows']} valid model rows; the common paired mask contains {miss['matched_rows']} rows = {miss['matched_cells']} cells across {miss['trajectory_count']} trajectories. {miss['unpaired_valid_rows']} valid rows are unpaired and excluded from cell effects. No missing physical outcome is imputed; fixed-denominator zeros are aggregation placeholders only.",
        "",
        "Primary fixed 15-cell denominator (equal trajectory weighting):",
        "",
        f"- {fmt('pooled O−P terminal reward', pooled['o_minus_p_terminal_reward'])}",
        f"- {fmt('pooled P−uniform terminal reward', pooled['p_minus_uniform_terminal_reward'])}",
        "",
        "Model-specific trajectory summaries:",
        "",
    ]
    for model in MODELS:
        lines.append(f"- `{model}`: {fmt_desc('O−P', models[model]['o_minus_p_terminal_reward'])}; {fmt_desc('P−uniform', models[model]['p_minus_uniform_terminal_reward'])}")
    lines += [
        "",
        "Planned exploratory Holm adjustment across the two model-specific contrasts in each family (reported for transparency, not as confirmatory claims):",
        f"- O−P family: official_dino_wm={holm['o_minus_p_terminal_reward']['official_dino_wm']:.6g}; official_jepa_wm={holm['o_minus_p_terminal_reward']['official_jepa_wm']:.6g}.",
        f"- P−uniform family: official_dino_wm={holm['p_minus_uniform_terminal_reward']['official_dino_wm']:.6g}; official_jepa_wm={holm['p_minus_uniform_terminal_reward']['official_jepa_wm']:.6g}.",
        "",
        f"Complete-valid sensitivity retains {complete['trajectory_count']} trajectories with all 15 paired cells. Its pooled O−P estimate is {complete['pooled']['o_minus_p_terminal_reward'].get('mean'):.6g} (95% CI [{complete['pooled']['o_minus_p_terminal_reward']['trajectory_bootstrap_95_ci'][0]:.6g}, {complete['pooled']['o_minus_p_terminal_reward']['trajectory_bootstrap_95_ci'][1]:.6g}]) and pooled P−uniform is {complete['pooled']['p_minus_uniform_terminal_reward'].get('mean'):.6g} (95% CI [{complete['pooled']['p_minus_uniform_terminal_reward']['trajectory_bootstrap_95_ci'][0]:.6g}, {complete['pooled']['p_minus_uniform_terminal_reward']['trajectory_bootstrap_95_ci'][1]:.6g}]).",
        "",
        "Interpretation: positive O−P means the realized-latent JSON selection receives a higher last-step reward than the predicted-latent selection on the identical candidate menu. Positive P−uniform means the JSON P selection exceeds the menu's mean last-step reward. The pooled O−P estimate is about +0.009 but its exact sign-flip p is about 0.055, so it is not robust evidence of improvement; the complete-valid estimate is near zero with a CI crossing zero. Changing from cumulative J to terminal reward therefore does not turn the prior negative or uncertain result into an established improvement. These are exploratory terminal-reward contrasts; terminal reward is distinct from cumulative J, the realized latent is not a physical oracle, and this does not establish physical success or generalization. The exact sign-flip and Holm values are exploratory and do not reclassify any existing confirmed result.",
        "",
        "Missingness by trajectory (paired cells): " + ", ".join(f"{t}={n}" for t, n in sorted(result['primary_fixed_15_cell_sensitivity']['matched_cell_counts'].items())) + ".",
        "",
    ]
    return "\n".join(lines)


def run_analysis(spec_path: Path, manifest_path: Path, merge_path: Path, script_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("frozen_before_outcomes") is not True:
        raise ValueError("analysis spec must be frozen before outcomes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 600:
        raise ValueError(f"expected canonical 600-row manifest, got {len(entries) if isinstance(entries, list) else None}")
    cells_by_trajectory: dict[str, list[tuple[int, int]]] = {}
    for entry in entries:
        cells_by_trajectory.setdefault(str(entry["trajectory_id"]), []).append((int(entry["offset"]), int(entry["seed"])))
    cells_by_trajectory = {t: sorted(set(cells)) for t, cells in cells_by_trajectory.items()}
    if len(cells_by_trajectory) != 20 or any(len(cells) != 15 for cells in cells_by_trajectory.values()):
        raise ValueError("canonical manifest does not contain 20 trajectories with 15 cells each")
    records, diagnostics = load_records(merge)
    paired = {(t, o, s) for t, o, s, m in records if all((t, o, s, other) in records for other in MODELS)}
    if len(records) != 577 or len(paired) != 281:
        raise ValueError(f"unexpected retained evidence mask: {len(records)} rows, {len(paired)} paired cells")
    aggregate_result = aggregate(records, cells_by_trajectory)
    matched_rows = 2 * sum(aggregate_result["matched_cell_counts"].values())
    if matched_rows != 562:
        raise ValueError(f"expected 562 matched rows, got {matched_rows}")
    source_hashes = {
        "analysis_spec_sha256": sha256_file(spec_path),
        "analysis_script_sha256": sha256_file(script_path),
        "canonical_manifest_file_sha256": sha256_file(manifest_path),
        "recovery_merge_index_file_sha256": sha256_file(merge_path),
        "original_manifest_content_sha256": merge.get("original_manifest_sha256"),
        "recovery_merge_content_sha256": merge.get("merge_sha256"),
        "original_ledger_sha256": merge.get("original_ledger_sha256"),
        "recovery_ledger_sha256": merge.get("recovery_ledger_sha256"),
        "matched_recovery_analyzer_sha256": sha256_file(Path("experiments/analyze_official_pusht_v14_recovery.py")),
        "matched_design_analyzer_sha256": sha256_file(Path("experiments/analyze_official_pusht_v14.py")),
    }
    elapsed = time.perf_counter() - started
    return {
        "analysis_name": spec["analysis_name"], "status": "posthoc_exploratory",
        "analysis_spec_sha256": source_hashes["analysis_spec_sha256"],
        "source_hashes": source_hashes,
        "runtime": {"python": "bundled codex-primary-runtime", "device": "CPU", "elapsed_seconds": elapsed},
        "official_semantics": {"physical_replay_steps": 30, "trace_shape": [300, 30], "terminal_reward": "initial_reward_traces[:, -1]", "cumulative_J_not_used": True, "cumulative_J_json_reconciliation_max_abs_error": diagnostics["max_cumulative_J_reconciliation_error"]},
        "selection_source": {"json_p_choice_field": "decision.predicted_choice", "json_o_choice_field": "decision.encoded_real_choice", "json_p_score_field": "candidate_rows[].predicted_score", "json_o_score_field": "candidate_rows[].encoded_real_score", "singleton_selection_rows": diagnostics["rows_read"]},
        "missingness": {
            "manifest_rows": 600, "canonical_cells": 300, "trajectory_count": 20,
            "valid_merge_rows": len(records), "paired_cells": len(paired), "matched_cells": len(paired), "matched_rows": matched_rows,
            "unpaired_valid_rows": len(records) - matched_rows, "invalid_merge_rows": 600 - len(records),
            "valid_rows_by_model": {model: sum(1 for key in records if key[3] == model) for model in MODELS},
            "complete_valid_trajectory_count": aggregate_result["complete_valid_sensitivity"]["trajectory_count"],
            "no_physical_outcome_imputation": True,
        },
        "contrast_definitions": spec["contrasts"],
        "primary_fixed_15_cell_sensitivity": aggregate_result,
        "interpretation_guards": spec["interpretation_guards"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("output/terminal_reward_analysis_spec_20260910.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/official_pusht_v14_manifest.json"))
    parser.add_argument("--merge", type=Path, default=Path("experiments/official_pusht_v14_recovery_20260831/recovery_merge_index.json"))
    parser.add_argument("--out-json", type=Path, default=Path("output/terminal_reward_sensitivity_20260910.json"))
    parser.add_argument("--out-md", type=Path, default=Path("output/terminal_reward_sensitivity_20260910.md"))
    args = parser.parse_args()
    result = run_analysis(args.spec, args.manifest, args.merge, Path(__file__).resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "elapsed_seconds": result["runtime"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
