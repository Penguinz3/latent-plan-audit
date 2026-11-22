"""Export the sealed Push-T v1.4-R candidate evidence to a compact CPU bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from recompute_pusht_bundle import FIELDS, SCHEMA, recompute, sha256, ties


ROOT = Path(__file__).resolve().parents[1]
MERGE = ROOT / "experiments/official_pusht_v14_recovery_20260831/recovery_merge_index.json"
DEST = ROOT / "output/pusht_mechanism_20260929/bundle"
STATUS = ROOT / "output/pusht_mechanism_20260929/bundle_status.json"
SOURCE_KEYS = {
    "costs": ("initial_predicted_visual_mse", "initial_observed_visual_mse", "initial_predicted_proprio_mse", "initial_observed_proprio_mse"),
    "captured_p_scores": "initial_objective_scores",
    "final_states": "initial_final_states",
    "goal_states": "goal_state",
    "reward_traces": "initial_reward_traces",
    "actions_normalized": "initial_actions_normalized",
    "actions_physical": "initial_physical_actions",
    "native_returns": "initial_native_returns",
    "native_distances": "initial_native_state_distances",
}
SHAPES = {
    "costs": (300, 4), "captured_p_scores": (300,), "final_states": (300, 7),
    "goal_states": (7,), "reward_traces": (300, 30), "actions_normalized": (300, 6, 10),
    "actions_physical": (300, 30, 2), "native_returns": (300,), "native_distances": (300,),
}
DTYPES = {key: np.float32 if key in ("reward_traces", "actions_normalized", "actions_physical") else np.float64 for key in SHAPES}
SHAPES.update({"normalized_action_sha256": (300,), "normalized_plan_multiplicity": (300,)})
DTYPES.update({"normalized_action_sha256": "S64", "normalized_plan_multiplicity": np.uint16})


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def status(stage: str, done: int, total: int, started: float, **extra: object) -> None:
    write_json(STATUS, {"stage": stage, "rows_done": done, "rows_total": total,
                        "elapsed_seconds": round(time.monotonic() - started, 3), **extra})


def metadata(row: dict, artifact: dict | None) -> dict:
    match = re.fullmatch(r"v14_(official_(?:jepa|dino)_wm)_pusht_(val_episode_\d+)_offset_(\d+)_seed_(\d+)", row["decision_id"])
    if match is None:
        raise ValueError(f"unrecognized decision ID: {row['decision_id']}")
    result = {key: row.get(key) for key in ("ordinal", "decision_id", "effective_status", "evidence_source", "original_terminal_status", "recovery_selected", "original_terminal_event_sha256", "recovery_terminal_event_sha256", "manifest_entry_sha256", "failure_first_line")}
    result.update({"model": match[1], "trajectory_id": match[2], "offset": int(match[3]), "menu_seed": int(match[4]),
                   "valid": bool(row["valid"]), "array_index": None, "artifact_sha256_recorded": row.get("artifact_sha256"),
                   "arrays_sha256_recorded_unverified": row.get("arrays_sha256")})
    if artifact is not None:
        for key in ("decision_id", "model", "trajectory_id", "offset", "menu_seed"):
            if artifact[key] != result[key]:
                raise ValueError(f"artifact metadata mismatch: {row['decision_id']} {key}")
        recovery = artifact["standard_outer_metrics"]["matched_recovery"]
        if recovery["abstained"] or float(artifact["protocol"]["alpha"]) != .1:
            raise ValueError(f"unexpected P/O abstention or weight: {row['decision_id']}")
        result.update({"p_index": int(recovery["predicted_choice_index"]),
                       "o_index": int(recovery["realized_latent_choice_index"]),
                       "protocol_sha256": artifact["protocol_sha256"],
                       "source_schema_version": artifact["schema_version"]})
    return result


def export(out: Path = DEST) -> dict:
    started = time.monotonic()
    source = json.loads(MERGE.read_text(encoding="utf-8"))
    if source["schema_version"] != "PWA-PushT-v1.4-R-merge-index-v1" or len(source["rows"]) != 600 or source["counts"] != {"invalid": 23, "nonselected_original_failures": 11, "rows": 600, "selected_for_recovery": 169, "valid": 577}:
        raise ValueError("unexpected sealed merge population")
    out.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.empty((577, *shape), dtype=DTYPES[key]) for key, shape in SHAPES.items()}
    ledger_rows = []
    status("exporting", 0, 600, started)
    valid_count = 0
    for ordinal, row in enumerate(source["rows"]):
        if row["ordinal"] != ordinal:
            raise ValueError("merge rows out of issued order")
        if not row["valid"]:
            ledger_rows.append(metadata(row, None))
            continue
        artifact_path, archive_path = Path(row["artifact_path"]), Path(row["arrays_path"])
        if sha256(artifact_path) != row["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch: {row['decision_id']}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        record = metadata(row, artifact)
        record["array_index"] = valid_count
        with np.load(archive_path, allow_pickle=False) as archive:
            for key, source_key in SOURCE_KEYS.items():
                value = np.stack([archive[x] for x in source_key], axis=-1) if isinstance(source_key, tuple) else archive[source_key]
                if value.shape != SHAPES[key] or not np.isfinite(value).all():
                    raise ValueError(f"missing, misaligned or nonfinite {key}: {row['decision_id']}")
                arrays[key][valid_count] = value
        candidate_rows = artifact["candidate_rows"]
        if len(candidate_rows) != 300:
            raise ValueError(f"candidate row count mismatch: {row['decision_id']}")
        for j, candidate in enumerate(candidate_rows):
            if candidate["candidate_id"] != f"proposal_{j:04d}":
                raise ValueError(f"candidate identity mismatch: {row['decision_id']}")
            action_hash = hashlib.sha256(arrays["actions_normalized"][valid_count, j].tobytes()).hexdigest()
            if action_hash != candidate["normalized_action_sha256"]:
                raise ValueError(f"candidate action hash mismatch: {row['decision_id']}")
            arrays["normalized_action_sha256"][valid_count, j] = action_hash.encode("ascii")
            arrays["normalized_plan_multiplicity"][valid_count, j] = candidate["normalized_plan_multiplicity"]
        _, inverse, counts = np.unique(arrays["normalized_action_sha256"][valid_count], return_inverse=True, return_counts=True)
        if not np.array_equal(counts[inverse], arrays["normalized_plan_multiplicity"][valid_count]):
            raise ValueError(f"candidate deduplication mismatch: {row['decision_id']}")
        costs, p = arrays["costs"][valid_count], arrays["captured_p_scores"][valid_count]
        o = costs[:, 1] + .1 * costs[:, 3]
        if np.max(np.abs(p - (costs[:, 0] + .1 * costs[:, 2]))) > 2e-5:
            raise ValueError(f"source score mismatch: {row['decision_id']}")
        pt, ot = ties(p, 1e-10, 1e-8), ties(o, 1e-10, 1e-8)
        if pt != [record["p_index"]] or ot != [record["o_index"]]:
            raise ValueError(f"source selector mismatch: {row['decision_id']}")
        record["p_tie_indices"], record["o_tie_indices"] = pt, ot
        states, goal = arrays["final_states"][valid_count], arrays["goal_states"][valid_count]
        if not np.allclose(arrays["native_distances"][valid_count], np.linalg.norm(states - goal, axis=1), atol=1e-12, rtol=0):
            raise ValueError(f"source physical G mismatch: {row['decision_id']}")
        rewards = arrays["reward_traces"][valid_count]
        if not np.array_equal(arrays["native_returns"][valid_count], np.add.accumulate(rewards, axis=1, dtype=np.float32)[:, -1].astype(np.float64)):
            raise ValueError(f"source physical J mismatch: {row['decision_id']}")
        ledger_rows.append(record)
        valid_count += 1
        if ordinal % 10 == 9:
            status("exporting", ordinal + 1, 600, started, valid_rows=valid_count)
    if valid_count != 577:
        raise ValueError("valid-row count changed")
    write_json(out / "rows.json", {"schema_version": SCHEMA, "weight": .1, "tie_atol": 1e-10,
                                  "tie_rtol": 1e-8, "score_tolerance": 2e-5, "cost_fields": list(FIELDS),
                                  "candidate_id_rule": "proposal_%04d for column 0..299; no deduplication",
                                  "rows": ledger_rows})
    status("compressing", 600, 600, started, valid_rows=valid_count)
    np.savez_compressed(out / "candidates.npz", **arrays)
    shutil.copy2(Path(__file__).with_name("recompute_pusht_bundle.py"), out / "recompute_pusht_bundle.py")
    write_json(out / "provenance.json", {
        "schema_version": SCHEMA,
        "scope": "analysis reproduction from exported candidate evidence; no simulator rollout or model inference verification",
        "source_merge_index_sha256_verified": sha256(MERGE),
        "source_merge_commitment": source["merge_sha256"],
        "source_artifact_hashes_verified": True,
        "source_npz_hashes": "recorded from sealed merge index; not recomputed during selective extraction",
        "bundle_sha256": {name: sha256(out / name) for name in ("rows.json", "candidates.npz", "recompute_pusht_bundle.py")},
        "python": sys.version.split()[0], "numpy": np.__version__,
        "source_analysis_sha256_recorded": sha256(ROOT / "experiments/official_pusht_v14_recovery_analysis_20260901.json"),
        "source_component_sha256_recorded": sha256(ROOT / "output/pusht_component_posthoc_20260914.json"),
    })
    status("recomputing_reference", 600, 600, started, valid_rows=valid_count)
    result = recompute(out, verify_reference=False)
    sealed = json.loads((ROOT / "experiments/official_pusht_v14_recovery_analysis_20260901.json").read_text(encoding="utf-8"))
    result["sealed_raw_inventory_means"] = {label: float(sealed["raw_inventory"][label]["mean"]) for label in ("Delta_G", "Delta_J")}
    if abs(result["effect_means"]["native_G"] - result["sealed_raw_inventory_means"]["Delta_G"]) > 1e-7 or abs(result["effect_means"]["cumulative_J"] - result["sealed_raw_inventory_means"]["Delta_J"]) > 1e-5:
        raise ValueError("exported primary G/J does not reproduce sealed means")
    write_json(out / "reference.json", result)
    (out / "README.md").write_text(
        "# Push-T compact candidate analysis bundle\n\n"
        "Use Python 3.12 and NumPy (produced with NumPy " + np.__version__ + "). Run `python recompute_pusht_bundle.py --bundle-dir .` inside this directory. "
        "The command checks all 600 issued rows, 577 valid menus, candidate score formulas, original strict P/O selectors, physical outcomes, "
        "the 562 matched rows/281 cells/20 trajectories, and numerical reference values. It uses only files in this directory.\n\n"
        "`rows.json` is in issued order. Valid rows map to `candidates.npz` through `array_index`; invalid rows retain status and failure provenance without invented outcomes. "
        "Cost channel order is listed in `rows.json`. Each of the 300 candidate columns is `proposal_%04d`; repeated action plans remain separate candidates. "
        "The complete normalized and physical action arrays, original per-candidate normalized action SHA-256 hashes, and normalized-plan multiplicities allow exact duplicate checks. Strict ties use NumPy `isclose` with the released tolerances; tied menus abstain. "
        "Cumulative J uses the retained native return, which is sequential float32 accumulation of the reward trace. The reference compares native G/J means with the sealed analysis. "
        "P uses original captured scores. O uses observed visual + 0.1 observed proprio. The four channel costs also permit exploratory hybrid scoring.\n\n"
        "This bundle reproduces analyses from saved candidate evidence. It does not independently rerun model inference or simulator trajectories. "
        "Those require the original v1.4/v1.4-R code, checkpoint, Push-T dataset, manifest, issuances, and source artifacts. "
        "Source JSON hashes were verified at export. Source NPZ hashes in `rows.json` are recorded commitments from the sealed merge index; selective NPZ extraction did not rehash the full raw archives.\n",
        encoding="utf-8")
    status("complete", 600, 600, started, valid_rows=valid_count, bundle_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    result = export(args.out)
    print(json.dumps({"counts": result["counts"], "bundle_dir": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
