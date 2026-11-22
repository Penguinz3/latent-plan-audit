"""Export a compact, anonymized Wall candidate table and CPU verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from recompute_wall import derive_record


EXPECTED_STUDY = "wall-initial-audit-20260910"
EXPECTED_ROWS = 180
EXPECTED_CANDIDATES = 300
TERMINAL_FAILURES = 1
COMPLETE_CASES = 29
FORBIDDEN_TEXT = ("F:\\stockast", "C:\\Users", "RanaGaming", "kulbi")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "item"


def result_path(run_dir: Path, item: dict[str, Any]) -> Path:
    return run_dir / "results" / f"{safe_name(str(item['decision_id']))}.json"


def attempt_dir(run_dir: Path, index: int, item: dict[str, Any]) -> Path:
    key = f"{item['case_id']}::{item['case_seed']}::{item['menu_seed']}::{item['model']}"
    return run_dir / "attempts" / f"{index:05d}_{safe_name(key)}"


def latest_receipt(path: Path) -> dict[str, Any] | None:
    values = []
    for receipt_path in path.glob("*/attempt.json"):
        try:
            value = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            values.append(value)
    return max(values, key=lambda value: int(value.get("attempt", 0)), default=None)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.replace("official_pusht_v14_near_tie_validation_20260904.json", "<redacted-training-reference>")
    text = re.sub(r"(?i)[A-Z]:[\\/][^\"']*", "<redacted-local-path>", text)
    text = text.replace("RanaGaming", "<redacted-host>").replace("kulbi", "<redacted-user>")
    return text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    analysis = json.loads(Path("output/wall_analysis_20260910.json").read_text(encoding="utf-8"))
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bindings = json.loads((run_dir / "execution_bindings.json").read_text(encoding="utf-8"))
    if protocol.get("study_id") != EXPECTED_STUDY or bindings.get("study_id") != EXPECTED_STUDY:
        raise ValueError("source study_id is not the frozen Wall audit")
    items = manifest.get("rows")
    if not isinstance(items, list) or len(items) != EXPECTED_ROWS:
        raise ValueError("manifest does not contain the frozen 180 rows")
    if analysis.get("coverage", {}).get("valid_rows") != EXPECTED_ROWS - TERMINAL_FAILURES:
        raise ValueError("analysis valid-row count is not 179")
    if analysis.get("coverage", {}).get("complete_case_count") != COMPLETE_CASES:
        raise ValueError("analysis complete-case count is not 29")
    return analysis, protocol, manifest, [dict(item) for item in items], bindings


def read_verified_results(run_dir: Path, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        path = result_path(run_dir, item)
        if not path.exists():
            missing.append(item)
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if any(result.get(key) != item.get(key) for key in ("case_id", "case_seed", "menu_seed", "model", "decision_id")):
            raise ValueError(f"identity mismatch in {path.name}")
        details = result.get("candidates_detail")
        if result.get("status") != "PASS" or result.get("scientific_result") is not True or result.get("candidates") != EXPECTED_CANDIDATES or not isinstance(details, list) or len(details) != EXPECTED_CANDIDATES:
            raise ValueError(f"incomplete scientific result {path.name}")
        receipt = latest_receipt(attempt_dir(run_dir, index, item))
        result_hash = sha256_bytes(canonical(result))
        if not receipt or receipt.get("status") != "complete" or str(receipt.get("result_sha256", "")).lower() != result_hash:
            raise ValueError(f"missing matching complete receipt for {path.name}")
        valid.append({"item": item, "result": result, "result_path": path, "result_sha256": result_hash})
    if len(valid) != EXPECTED_ROWS - TERMINAL_FAILURES or len(missing) != TERMINAL_FAILURES:
        raise ValueError(f"expected 179 valid results and 1 missing failure, got {len(valid)} and {len(missing)}")
    return valid, missing


def make_reference(analysis: dict[str, Any]) -> dict[str, Any]:
    keys = ("analysis_name", "status", "scientific_result", "coverage", "primary", "diagnostics", "flagged_vs_unflagged_regret", "protocol", "uncertainty", "interpretation_guards")
    return {key: sanitize(analysis[key]) for key in keys}


def build_bundle(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing export directory: {output_dir}")
    output_dir.mkdir(parents=True)
    analysis, protocol, manifest, items, bindings = load_inputs(run_dir)
    frozen = json.loads((run_dir / "frozen_near_tie.json").read_text(encoding="utf-8"))
    valid, missing = read_verified_results(run_dir, items)

    metadata = output_dir / "metadata"
    metadata.mkdir()
    write_json(metadata / "protocol.json", sanitize(protocol))
    write_json(metadata / "analysis_plan.json", sanitize(json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))))
    write_json(metadata / "frozen_near_tie.json", sanitize(json.loads((run_dir / "frozen_near_tie.json").read_text(encoding="utf-8"))))
    write_json(metadata / "manifest.json", sanitize(manifest))
    write_json(metadata / "analysis_reference.json", make_reference(analysis))
    failures = []
    for failure in analysis.get("source", {}).get("terminal_failures", []):
        failures.append({
            "item": sanitize(failure.get("item")),
            "status": failure.get("status"),
            "failure_class": failure.get("failure_class"),
            "error": sanitize(failure.get("error")),
            "imputed": False,
            "primary_case_effect": "excluded because wall_028 has only five valid rows",
        })
    if len(failures) != TERMINAL_FAILURES:
        raise ValueError("source failure ledger does not contain exactly one terminal failure")
    failure_keys = {(row["item"]["case_id"], int(row["item"]["case_seed"]), int(row["item"]["menu_seed"]), row["item"]["model"]) for row in failures}
    missing_keys = {(row["case_id"], int(row["case_seed"]), int(row["menu_seed"]), row["model"]) for row in missing}
    if failure_keys != missing_keys:
        raise ValueError(f"missing result identities do not match the frozen failure ledger: {missing_keys} vs {failure_keys}")
    write_json(metadata / "failure_ledger.json", {"terminal_failure_rows": len(failures), "rows": failures})

    row_summaries: list[dict[str, Any]] = []
    candidate_path = output_dir / "candidate_rows.jsonl"
    arrays_physical: list[Any] = []
    arrays_terminal: list[Any] = []
    source_hashes: dict[str, Any] = {}
    with candidate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for source in valid:
            item, result = source["item"], source["result"]
            details = result["candidates_detail"]
            flat_rows: list[dict[str, Any]] = []
            for candidate in details:
                row = {
                    "case_id": item["case_id"], "case_seed": item["case_seed"], "menu_seed": item["menu_seed"], "model": item["model"], "decision_id": item["decision_id"],
                    "candidate": int(candidate["candidate"]), "predicted_choice_P": int(result["predicted_choice_P"]), "encoded_real_choice_O": int(result["encoded_real_choice_O"]),
                    "predicted_cost_P": float(candidate["predicted_cost_P"]), "encoded_real_cost_O": float(candidate["encoded_real_cost_O"]),
                    "native_terminal_distance": float(candidate["native_terminal_distance"]), "native_terminal_success": bool(candidate["native_terminal_success"]),
                    "actions_physical_sha256": str(candidate["actions_physical_sha256"]),
                }
                flat_rows.append(row)
                handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                arrays_physical.append(candidate["actions_physical"])
                arrays_terminal.append(candidate["terminal_state"])
            source_hashes[item["decision_id"]] = {
                "result_sha256": source["result_sha256"], "sidecar_sha256": result.get("arrays_sidecar_sha256"),
                "adapter_sha256": result.get("adapter_sha256"), "checkpoint_sha256": result.get("checkpoint_sha256"),
                "config_sha256": result.get("config_sha256"), "protocol_sha256": result.get("protocol_sha256"),
            }
            summary = derive_record(flat_rows, frozen)
            summary.update({
                "initial_state": result["initial_state"], "goal_state": result["goal_state"],
                "P_margin_second_minus_best": result.get("P_margin_second_minus_best"), "O_margin_second_minus_best": result.get("O_margin_second_minus_best"),
                "source_result_sha256": source["result_sha256"], "source_sidecar_sha256": result.get("arrays_sidecar_sha256"),
            })
            row_summaries.append(summary)
    write_jsonl(output_dir / "row_summaries.jsonl", row_summaries)

    np.savez_compressed(output_dir / "candidate_arrays.npz", actions_physical=np.asarray(arrays_physical, dtype=np.float32), terminal_state=np.asarray(arrays_terminal, dtype=np.float32))
    provenance = {
        "schema_version": 2, "artifact_version": 2, "artifact_type": "compact_anonymized_wall_recompute", "study_id": EXPECTED_STUDY,
        "valid_candidate_rows": len(valid) * EXPECTED_CANDIDATES, "valid_result_rows": len(valid), "terminal_failure_rows": len(missing), "complete_case_count": COMPLETE_CASES,
        "original_source_hashes": {
            "protocol_sha256": sha256_file(run_dir / "protocol.json"), "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "analysis_plan_sha256": sha256_file(run_dir / "analysis_plan.json"), "frozen_near_tie_sha256": sha256_file(run_dir / "frozen_near_tie.json"),
            "execution_bindings_sha256": sha256_file(run_dir / "execution_bindings.json"), "analysis_output_sha256": sha256_file(Path("output/wall_analysis_20260910.json")),
            "frozen_wall_analysis_sha256": sha256_file(Path("experiments/analyze_wall_audit.py")),
            "official_commit": bindings.get("official_commit"), "checkpoint_sha256": bindings.get("checkpoints"),
        },
        "result_hashes_by_decision_id": source_hashes,
        "missing_failure_policy": "The one terminal scientific failure is retained in metadata/failure_ledger.json and never imputed.",
        "raw_data_policy": "The original result JSON, NPZ sidecars, attempts, and receipts remain local and are not included in this compact artifact.",
    }
    write_json(metadata / "provenance.json", provenance)
    shutil.copyfile(Path("experiments/analyze_wall_audit.py"), output_dir / "frozen_wall_analysis.py")
    shutil.copyfile(Path(__file__).resolve().parent / "recompute_wall.py", output_dir / "recompute_wall.py")
    write_readme(output_dir)
    verification = output_dir / "metadata" / "recompute_verification.json"
    subprocess.run([sys.executable, str(output_dir / "recompute_wall.py"), "--bundle-dir", str(output_dir), "--write-report", str(verification)], check=True)
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "export_manifest.json" and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc":
            files.append({"path": path.relative_to(output_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output_dir / "export_manifest.json", {"schema_version": 1, "study_id": EXPECTED_STUDY, "files": files})
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        raise SystemExit(f"Refusing to overwrite existing export ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc":
                archive.write(path, path.relative_to(output_dir).as_posix())
    return zip_path


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Compact anonymized Wall recomputation artifact v2

This supplement contains 179 verified Wall result rows and 53,700 candidate rows from the finalized Wall decision audit. The one terminal scientific failure is retained in `metadata/failure_ledger.json`; its outcomes are not imputed, and `wall_028` is excluded from the 29 complete case clusters used for primary case-level effects.

The bundle is an analysis artifact, not the original sealed evidence package. Original result JSON, compressed sidecars, attempt receipts, private papers, and the original Push-T NPZ dataset remain local to the study workspace. This bundle does not claim to reproduce the full simulator or model inference. It reproduces the exported candidate-level summaries, primary case effects, and fixed frozen-feature AUROCs from those summaries.

## CPU recomputation

Use Python 3.11 with NumPy (the producing environment used NumPy 2.4.6):

```text
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe recompute_wall.py --bundle-dir .
```

The command validates manifest coverage, the one-row failure ledger, finite P/O vectors, and first argmins. It recomputes P/O effects over complete six-row cases, fixed near-tie and negative-margin AUROCs, bootstrap CIs, sign-flip p-values, and Holm adjustments. It imports only the pure `_stats` and `_holm` helpers from the unchanged bundled `frozen_wall_analysis.py` and compares all of those values with `metadata/analysis_reference.json` at 1e-12 tolerance. It does not run the analyzer CLI, model inference, or a full simulator. Frozen thresholds and logistic coefficients are copied under `metadata/frozen_near_tie.json`; they were not retuned on Wall outcomes. Physical actions and terminal states are in `candidate_arrays.npz` in the same JSONL order.

The provenance file records immutable source hashes, official commit, checkpoint hashes, and per-result receipt/sidecar hashes without local paths, hostnames, or usernames. Public repository references remain metadata only. This artifact is for local analysis and review; it is not an external publication package.
""",
        encoding="utf-8",
    )
    (output_dir / "requirements.txt").write_text("numpy==2.4.6\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/wall_audit_20260910"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/wall_recompute_20260910"))
    args = parser.parse_args(argv)
    zip_path = build_bundle(args)
    text_files = [path for path in args.output_dir.resolve().rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".md", ".py", ".txt"}]
    forbidden = [str(path) for path in text_files if any(token.lower() in path.read_text(encoding="utf-8").lower() for token in FORBIDDEN_TEXT)]
    if forbidden:
        raise SystemExit(f"forbidden local identity text found: {forbidden}")
    print(f"created {zip_path} ({zip_path.stat().st_size} bytes, {sha256_file(zip_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
