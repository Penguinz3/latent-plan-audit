"""Outcome-blind v1.4-R recovery for the isolated CUDA failure cascade.

This module never edits or replaces v1.4 evidence.  It freezes the exact
eligible rows in a separate issuance, runs each authorized row in one fresh
OS process, and emits a deterministic 600-row merge index that retains every
unrecovered v1.4 failure as invalid evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CUDA_ILLEGAL_ADDRESS = "CUDA error: an illegal memory access was encountered"
EXPECTED_SELECTED_COUNT = 169
EXPECTED_SELECTED_ORDINALS = tuple(range(431, 600))
EXPECTED_NONSELECTED_FAILURES = 11
ROW_TIMEOUT_SECONDS = 30 * 60
ISSUANCE_SCHEMA = "PWA-PushT-v1.4-R-recovery-issuance-v1"
CAPABILITY_SCHEMA = "PWA-PushT-v1.4-R-row-capability-v1"
LEDGER_SCHEMA = "PWA-PushT-v1.4-R-recovery-ledger-v1"
MERGE_SCHEMA = "PWA-PushT-v1.4-R-merge-index-v1"

DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "official_pusht_v14_manifest.json"
DEFAULT_LEDGER = PROJECT_ROOT / "experiments" / "official_pusht_v14_study_ledger.jsonl"
DEFAULT_STUDY_ISSUANCE = PROJECT_ROOT / "experiments" / "official_pusht_v14_study_issuance.json"
DEFAULT_CERTIFICATE_ISSUANCE = PROJECT_ROOT / "experiments" / "official_pusht_v14_certificate_issuance.json"
DEFAULT_CERTIFICATE_BUNDLE = PROJECT_ROOT / "experiments" / "official_pusht_v14_certificate_20260830" / "certificate_bundle.json"
DEFAULT_RECOVERY_ADDENDUM = PROJECT_ROOT / "PWA_PUSHT_V14_RECOVERY_ADDENDUM.md"
DEFAULT_RECOVERY_ROOT = PROJECT_ROOT / "experiments" / "official_pusht_v14_recovery_20260831"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {Path(path)}")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _event_hash(event: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in event.items() if key != "event_sha256"})


def _read_ledger(path: Path | str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str]:
    try:
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid append-only ledger: {exc}") from exc
    if not records or not isinstance(records[0], dict) or records[0].get("status") != "run_header":
        raise ValueError("append-only ledger lacks its run header")
    header, previous = records[0], None
    histories: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"ledger event {index} is not an object")
        if record.get("previous_event_sha256") != previous:
            raise ValueError(f"ledger event {index} breaks its predecessor chain")
        if record.get("event_sha256") != _event_hash(record):
            raise ValueError(f"ledger event {index} hash mismatch")
        if index and record.get("run_id") != header.get("run_id"):
            raise ValueError(f"ledger event {index} run_id mismatch")
        if index and record.get("run_commitment_sha256") != header.get("event_sha256"):
            raise ValueError(f"ledger event {index} run commitment mismatch")
        previous = str(record["event_sha256"])
        if index:
            decision_id = record.get("decision_id")
            if not isinstance(decision_id, str):
                raise ValueError(f"ledger event {index} lacks decision_id")
            histories.setdefault(decision_id, []).append(record)
    return header, histories, previous


def _new_ledger(path: Path, commitments: Mapping[str, Any], run_id: str) -> tuple[dict[str, Any], str]:
    header = {
        "status": "run_header", "schema_version": LEDGER_SCHEMA,
        "run_id": run_id, "previous_event_sha256": None, **dict(commitments),
    }
    header["event_sha256"] = _event_hash(header)
    _exclusive_json(path, header)
    return header, str(header["event_sha256"])


def _append_event(path: Path, payload: Mapping[str, Any], header: Mapping[str, Any], previous: str) -> str:
    event = {
        **dict(payload), "run_id": header["run_id"],
        "run_commitment_sha256": header["event_sha256"],
        "previous_event_sha256": previous,
    }
    event["event_sha256"] = _event_hash(event)
    with path.open("ab") as handle:
        handle.write((json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return str(event["event_sha256"])


def _first_error_line(record: Mapping[str, Any]) -> str | None:
    error = record.get("error")
    if not isinstance(error, str) or not error.splitlines():
        return None
    return error.splitlines()[0]


def _original_histories(manifest: Mapping[str, Any], ledger_path: Path | str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("original manifest lacks entries")
    ids = [entry.get("decision_id") for entry in entries if isinstance(entry, Mapping)]
    if len(ids) != len(entries) or any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("original manifest decision IDs are missing or duplicated")
    header, histories, _tail = _read_ledger(ledger_path)
    if set(histories) != set(ids):
        raise ValueError("original ledger is missing or contains non-manifest rows")
    for ordinal, entry in enumerate(entries):
        decision_id = str(entry["decision_id"])
        history = histories[decision_id]
        statuses = [record.get("status") for record in history]
        if statuses not in (["started", "completed"], ["started", "failed"]):
            raise ValueError(f"original row {decision_id} is not exactly terminal")
        expected = {
            "ordinal": ordinal, "model": entry.get("model"), "episode": entry.get("episode"),
            "offset": entry.get("offset"), "menu_seed": entry.get("seed"),
        }
        for record in history:
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"original row {decision_id} differs from its manifest binding")
        if history[0].get("row_bindings") != history[-1].get("row_bindings"):
            raise ValueError(f"original row {decision_id} changed row_bindings between receipts")
    return header, histories


def select_cuda_failures(manifest: Mapping[str, Any], ledger_path: Path | str) -> list[dict[str, Any]]:
    """Select terminal failures solely by the exact first error line."""
    _header, histories = _original_histories(manifest, ledger_path)
    selected: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(manifest["entries"]):
        history = histories[str(entry["decision_id"])]
        terminal = history[-1]
        if terminal.get("status") != "failed" or _first_error_line(terminal) != CUDA_ILLEGAL_ADDRESS:
            continue
        selected.append({
            "recovery_ordinal": len(selected), "original_ordinal": ordinal,
            "decision_id": entry["decision_id"], "trajectory_id": entry.get("trajectory_id"),
            "model": entry.get("model"), "episode": entry.get("episode"),
            "offset": entry.get("offset"), "seed": entry.get("seed"),
            "manifest_entry": dict(entry), "manifest_entry_sha256": canonical_sha256(entry),
            "original_started_event_sha256": history[0]["event_sha256"],
            "original_terminal_event_sha256": terminal["event_sha256"],
            "original_first_error_line": CUDA_ILLEGAL_ADDRESS,
            "row_bindings": terminal.get("row_bindings"),
            "row_bindings_sha256": canonical_sha256(terminal.get("row_bindings")),
            "certified_menu_sha256": terminal.get("certified_menu_sha256"),
            "certificate_sha256": terminal.get("certificate_sha256"),
        })
    return selected


def _source_record(kind: str, path: Path | str, inner: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = file_sha256(path)
    if digest is None:
        raise FileNotFoundError(f"missing {kind}: {path}")
    return {"kind": kind, "path": str(path), "sha256": digest, **dict(inner or {})}


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


@contextlib.contextmanager
def _supervisor_lock(path: Path):
    """Hold one nonblocking OS lock; the OS releases it if the process dies."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another v1.4-R recovery supervisor is already running") from exc
            unlock = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another v1.4-R recovery supervisor is already running") from exc
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()
    finally:
        handle.close()


def prepare_recovery(
    *, manifest_path: Path | str = DEFAULT_MANIFEST, original_ledger_path: Path | str = DEFAULT_LEDGER,
    study_issuance_path: Path | str = DEFAULT_STUDY_ISSUANCE,
    certificate_issuance_path: Path | str = DEFAULT_CERTIFICATE_ISSUANCE,
    certificate_bundle_path: Path | str = DEFAULT_CERTIFICATE_BUNDLE,
    recovery_addendum_path: Path | str = DEFAULT_RECOVERY_ADDENDUM,
    recovery_root: Path | str = DEFAULT_RECOVERY_ROOT,
    expected_count: int | None = EXPECTED_SELECTED_COUNT,
    expected_ordinals: Sequence[int] | None = EXPECTED_SELECTED_ORDINALS,
    row_timeout_seconds: int = ROW_TIMEOUT_SECONDS,
) -> tuple[Path, dict[str, Any]]:
    """Freeze the outcome-blind recovery addendum without running a row."""
    paths = [Path(value).resolve() for value in (
        manifest_path, original_ledger_path, study_issuance_path,
        certificate_issuance_path, certificate_bundle_path, recovery_addendum_path,
    )]
    recovery_root = Path(recovery_root).resolve()
    if isinstance(row_timeout_seconds, bool) or not isinstance(row_timeout_seconds, int) or row_timeout_seconds <= 0:
        raise ValueError("recovery row timeout must be a positive integer")
    if any(_paths_overlap(recovery_root, path) for path in paths):
        raise ValueError("recovery root overlaps frozen v1.4 source evidence")
    manifest, study, certificate, bundle = map(_json, (paths[0], paths[2], paths[3], paths[4]))
    if study.get("issuance_sha256") != canonical_sha256({key: value for key, value in study.items() if key != "issuance_sha256"}):
        raise ValueError("original study issuance hash mismatch")
    if certificate.get("issuance_sha256") != canonical_sha256({key: value for key, value in certificate.items() if key != "issuance_sha256"}):
        raise ValueError("original certificate issuance hash mismatch")
    if bundle.get("bundle_sha256") != canonical_sha256({key: value for key, value in bundle.items() if key != "bundle_sha256"}):
        raise ValueError("original certificate bundle hash mismatch")
    original_header, _histories = _original_histories(manifest, paths[1])
    manifest_hash = canonical_sha256(manifest)
    if study.get("manifest_sha256") != manifest_hash or original_header.get("manifest_sha256") != manifest_hash:
        raise ValueError("original manifest commitment mismatch")
    if study.get("study_ledger_path") != str(paths[1]):
        raise ValueError("original study issuance points to a different ledger")
    if study.get("certificate_bundle_sha256") != original_header.get("certificate_bundle_sha256"):
        raise ValueError("original certificate-bundle commitment mismatch")
    if bundle.get("bundle_sha256") != study.get("certificate_bundle_sha256"):
        raise ValueError("certificate bundle does not match the original study issuance")
    if file_sha256(paths[3]) != study.get("certificate_issuance_sha256"):
        raise ValueError("certificate issuance does not match the original study issuance")
    certificate_reference = bundle.get("certificate_issuance")
    if not isinstance(certificate_reference, Mapping) or certificate_reference.get("sha256") != study.get("certificate_issuance_sha256") or certificate_reference.get("issuance_sha256") != certificate.get("issuance_sha256"):
        raise ValueError("certificate bundle issuance reference mismatch")
    if study.get("source_bundle_sha256") != original_header.get("source_bundle_sha256"):
        raise ValueError("original source-bundle commitment mismatch")
    selected = select_cuda_failures(manifest, paths[1])
    ordinals = [row["original_ordinal"] for row in selected]
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(f"recovery selection has {len(selected)} rows, expected {expected_count}")
    if expected_ordinals is not None and ordinals != list(expected_ordinals):
        raise ValueError("recovery selection ordinals differ from the frozen addendum scope")
    sources = {
        "manifest": _source_record("manifest", paths[0], {"canonical_sha256": manifest_hash}),
        "original_ledger": _source_record("original_ledger", paths[1], {
            "run_id": original_header.get("run_id"), "header_event_sha256": original_header.get("event_sha256"),
        }),
        "study_issuance": _source_record("study_issuance", paths[2], {"issuance_sha256": study.get("issuance_sha256")}),
        "certificate_issuance": _source_record("certificate_issuance", paths[3], {"issuance_sha256": certificate.get("issuance_sha256")}),
        "certificate_bundle": _source_record("certificate_bundle", paths[4], {"bundle_sha256": bundle.get("bundle_sha256")}),
        "recovery_addendum": _source_record("recovery_addendum", paths[5]),
    }
    issuance_path = recovery_root / "recovery_issuance.json"
    issuance = {
        "schema_version": ISSUANCE_SCHEMA, "status": "consumed", "one_shot": True,
        "issuance_kind": "outcome_blind_cuda_recovery_addendum", "run_id": uuid.uuid4().hex,
        "selection_rule": {"terminal_status": "failed", "first_error_line_equals": CUDA_ILLEGAL_ADDRESS},
        "sources": sources, "source_bundle_sha256": study.get("source_bundle_sha256"),
        "protocol_sha256": study.get("protocol_sha256"),
        "recovery_source_sha256": file_sha256(Path(__file__).resolve()),
        "recovery_root": str(recovery_root), "output_dir": str((recovery_root / "rows").resolve()),
        "log_dir": str((recovery_root / "logs").resolve()),
        "ledger_path": str((recovery_root / "recovery_ledger.jsonl").resolve()),
        "capability_dir": str((recovery_root / ".row_capabilities").resolve()),
        "row_timeout_seconds": row_timeout_seconds,
        "models": study.get("models"), "official_root": study.get("official_root"),
        "dataset_root": study.get("dataset_root"), "dataset_split": study.get("dataset_split"),
        "runtime": study.get("runtime"), "hardware": study.get("hardware"), "dinov2": study.get("dinov2"),
        "parent_launcher_attestation": study.get("launcher_attestation"),
        "certificate_issuance_reference": bundle.get("certificate_issuance"),
        "selected_rows": selected, "selected_rows_sha256": canonical_sha256(selected),
    }
    issuance["issuance_sha256"] = canonical_sha256(issuance)
    _exclusive_json(issuance_path, issuance)
    return issuance_path, issuance


def _issuance_reference(path: Path, issuance: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "issuance_sha256": issuance["issuance_sha256"]}


def validate_recovery_issuance(path: Path | str) -> dict[str, Any]:
    path, issuance = Path(path).resolve(), _json(path)
    if issuance.get("schema_version") != ISSUANCE_SCHEMA or issuance.get("status") != "consumed" or issuance.get("one_shot") is not True:
        raise ValueError("recovery issuance is not a consumed one-shot addendum")
    advertised = issuance.get("issuance_sha256")
    if advertised != canonical_sha256({key: value for key, value in issuance.items() if key != "issuance_sha256"}):
        raise ValueError("recovery issuance hash mismatch")
    if issuance.get("recovery_source_sha256") != file_sha256(Path(__file__).resolve()):
        raise ValueError("recovery implementation changed after issuance")
    sources = issuance.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "manifest", "original_ledger", "study_issuance", "certificate_issuance", "certificate_bundle",
        "recovery_addendum",
    }:
        raise ValueError("recovery issuance source schema mismatch")
    for name, source in sources.items():
        if not isinstance(source, Mapping) or file_sha256(Path(str(source.get("path")))) != source.get("sha256"):
            raise ValueError(f"frozen {name} evidence changed after recovery issuance")
    manifest = _json(Path(str(sources["manifest"]["path"])))
    if canonical_sha256(manifest) != sources["manifest"].get("canonical_sha256"):
        raise ValueError("frozen manifest canonical hash mismatch")
    selected = select_cuda_failures(manifest, Path(str(sources["original_ledger"]["path"])))
    if selected != issuance.get("selected_rows") or canonical_sha256(selected) != issuance.get("selected_rows_sha256"):
        raise ValueError("recovery selection no longer reproduces from frozen evidence")
    ids = [row.get("decision_id") for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("recovery issuance contains duplicate rows")
    if isinstance(issuance.get("row_timeout_seconds"), bool) or not isinstance(issuance.get("row_timeout_seconds"), int) or issuance["row_timeout_seconds"] <= 0:
        raise ValueError("recovery issuance row timeout is invalid")
    original_study = _json(Path(str(sources["study_issuance"]["path"])))
    if not isinstance(issuance.get("parent_launcher_attestation"), Mapping) or issuance.get("parent_launcher_attestation") != original_study.get("launcher_attestation"):
        raise ValueError("recovery issuance lacks the frozen parent launcher attestation")
    return issuance


def _capability_path(issuance: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    return Path(str(issuance["capability_dir"])) / f"{int(row['recovery_ordinal']):03d}_{row['decision_id']}.json"


def _consumed_capability_path(path: Path) -> Path:
    return path.with_name(path.stem + ".consumed.json")


def _row_paths(issuance: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(str(issuance["output_dir"]))
    return root / f"{row['decision_id']}.json", root / f"{row['decision_id']}.npz"


def _row_log_paths(issuance: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(str(issuance["log_dir"]))
    return root / f"{row['decision_id']}.stdout.txt", root / f"{row['decision_id']}.stderr.txt"


def _create_capability(
    issuance_path: Path, issuance: Mapping[str, Any], row: Mapping[str, Any],
    header: Mapping[str, Any], started_event_sha256: str,
) -> Path:
    path = _capability_path(issuance, row)
    artifact, arrays = _row_paths(issuance, row)
    record = {
        "schema_version": CAPABILITY_SCHEMA, "status": "authorized", "one_shot": True,
        "recovery_issuance": _issuance_reference(issuance_path, issuance),
        "run_id": issuance["run_id"], "run_commitment_sha256": header["event_sha256"],
        "started_event_sha256": started_event_sha256, "ledger_path": issuance["ledger_path"],
        "selected_row": dict(row), "selected_row_sha256": canonical_sha256(row),
        "artifact_path": str(artifact.resolve()), "arrays_path": str(arrays.resolve()),
    }
    record["capability_sha256"] = canonical_sha256(record)
    _exclusive_json(path, record)
    return path


def _consume_capability(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    record = _json(path)
    required = {
        "schema_version", "status", "one_shot", "recovery_issuance", "run_id",
        "run_commitment_sha256", "started_event_sha256", "ledger_path", "selected_row",
        "selected_row_sha256", "artifact_path", "arrays_path", "capability_sha256",
    }
    if set(record) != required or record.get("schema_version") != CAPABILITY_SCHEMA or record.get("status") != "authorized" or record.get("one_shot") is not True:
        raise ValueError("recovery row capability has an unexpected schema or state")
    if record.get("capability_sha256") != canonical_sha256({key: value for key, value in record.items() if key != "capability_sha256"}):
        raise ValueError("recovery row capability hash mismatch")
    reference = record["recovery_issuance"]
    issuance_path = Path(str(reference["path"])).resolve()
    if file_sha256(issuance_path) != reference.get("sha256"):
        raise ValueError("recovery row capability issuance file changed")
    issuance = validate_recovery_issuance(issuance_path)
    if issuance.get("issuance_sha256") != reference.get("issuance_sha256") or issuance.get("run_id") != record.get("run_id"):
        raise ValueError("recovery row capability issuance binding mismatch")
    row = record.get("selected_row")
    if not isinstance(row, Mapping) or canonical_sha256(row) != record.get("selected_row_sha256") or row not in issuance["selected_rows"]:
        raise ValueError("recovery row capability is not bound to an issued row")
    if path != _capability_path(issuance, row):
        raise ValueError("recovery row capability path mismatch")
    artifact, arrays = _row_paths(issuance, row)
    if record.get("artifact_path") != str(artifact.resolve()) or record.get("arrays_path") != str(arrays.resolve()):
        raise ValueError("recovery row capability output binding mismatch")
    header, histories, _tail = _read_ledger(Path(str(record["ledger_path"])))
    history = histories.get(str(row["decision_id"]), [])
    if header.get("event_sha256") != record.get("run_commitment_sha256") or [item.get("status") for item in history] != ["started"] or history[0].get("event_sha256") != record.get("started_event_sha256"):
        raise ValueError("recovery row capability lacks exactly one live started receipt")
    consumed = _consumed_capability_path(path)
    payload = path.read_bytes()
    consumed.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.unlink()
    consumed_reference = {
        "status": "consumed", "path": str(consumed.resolve()), "sha256": file_sha256(consumed),
        "capability_sha256": record["capability_sha256"],
    }
    return issuance, dict(row), consumed_reference


def _terminal_payload(issuance: Mapping[str, Any], row: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    artifact, arrays = _row_paths(issuance, row)
    stdout, stderr = _row_log_paths(issuance, row)
    capability = _capability_path(issuance, row)
    consumed = _consumed_capability_path(capability)
    cap_path = consumed if consumed.is_file() else capability
    return {
        "decision_id": row["decision_id"], "status": status,
        "recovery_ordinal": row["recovery_ordinal"], "original_ordinal": row["original_ordinal"],
        "original_terminal_event_sha256": row["original_terminal_event_sha256"],
        "selected_row_sha256": canonical_sha256(row),
        "artifact_path": str(artifact.resolve()), "arrays_path": str(arrays.resolve()),
        "artifact_exists": artifact.is_file(), "arrays_exists": arrays.is_file(),
        "artifact_sha256": file_sha256(artifact), "arrays_sha256": file_sha256(arrays),
        "stdout_path": str(stdout.resolve()), "stderr_path": str(stderr.resolve()),
        "stdout_exists": stdout.is_file(), "stderr_exists": stderr.is_file(),
        "stdout_sha256": file_sha256(stdout), "stderr_sha256": file_sha256(stderr),
        "capability_path": str(cap_path.resolve()), "capability_sha256": file_sha256(cap_path),
        **extra,
    }


def _recovery_commitments(issuance_path: Path, issuance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recovery_issuance": _issuance_reference(issuance_path, issuance),
        "selected_rows_sha256": issuance["selected_rows_sha256"],
        "source_bundle_sha256": issuance["source_bundle_sha256"],
        "protocol_sha256": issuance["protocol_sha256"],
    }


def _validate_recovery_ledger(issuance_path: Path, issuance: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str]:
    header, histories, tail = _read_ledger(Path(str(issuance["ledger_path"])))
    commitments = _recovery_commitments(issuance_path, issuance)
    if header.get("schema_version") != LEDGER_SCHEMA or header.get("run_id") != issuance.get("run_id"):
        raise ValueError("recovery ledger header identity mismatch")
    if any(header.get(key) != value for key, value in commitments.items()):
        raise ValueError("recovery ledger header commitment mismatch")
    issued = {row["decision_id"]: row for row in issuance["selected_rows"]}
    if not set(histories).issubset(issued):
        raise ValueError("recovery ledger contains an unissued row")
    for decision_id, history in histories.items():
        statuses = [item.get("status") for item in history]
        if statuses not in (["started"], ["started", "completed"], ["started", "failed"]):
            raise ValueError(f"recovery row {decision_id} is duplicated or has an illegal history")
        row = issued[decision_id]
        for event in history:
            if event.get("selected_row_sha256") != canonical_sha256(row):
                raise ValueError(f"recovery row {decision_id} binding mismatch")
        if len(history) == 2:
            _verify_recovery_logs(history[-1], f"recovery row {decision_id}")
            if history[-1].get("status") == "completed":
                _verify_receipt_files(history[-1], f"recovery row {decision_id}")
    return header, histories, tail


def _worker_command(issuance: Mapping[str, Any], capability: Path) -> list[str]:
    runtime = issuance.get("runtime")
    executable = runtime.get("python_executable") if isinstance(runtime, Mapping) else None
    if not isinstance(executable, str) or not Path(executable).is_file():
        raise FileNotFoundError("issued v1.4 Python runtime is unavailable")
    return [executable, "-I", "-B", str(Path(__file__).resolve()), "_worker", "--capability", str(capability.resolve())]


def _run_recovery_locked(
    issuance_path: Path | str,
    *, executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run each unstarted issued row once, in its own fresh OS process."""
    issuance_path = Path(issuance_path).resolve()
    issuance = validate_recovery_issuance(issuance_path)
    ledger = Path(str(issuance["ledger_path"])).resolve()
    output = Path(str(issuance["output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    Path(str(issuance["log_dir"])).mkdir(parents=True, exist_ok=True)
    Path(str(issuance["capability_dir"])).mkdir(parents=True, exist_ok=True)
    if ledger.is_file():
        header, histories, previous = _validate_recovery_ledger(issuance_path, issuance)
    else:
        header, previous = _new_ledger(ledger, _recovery_commitments(issuance_path, issuance), issuance["run_id"])
        histories = {}
    # A start is an attempt.  If its process outcome was not receipted, resume
    # finalizes it as failed and never launches it again.
    for row in issuance["selected_rows"]:
        decision_id = row["decision_id"]
        history = histories.get(decision_id, [])
        if [item.get("status") for item in history] == ["started"]:
            stdout_path, stderr_path = _row_log_paths(issuance, row)
            if not stdout_path.exists():
                _exclusive_text(stdout_path, "")
            if not stderr_path.exists():
                _exclusive_text(stderr_path, "supervisor interrupted before a terminal receipt\n")
            terminal = _terminal_payload(
                issuance, row, "failed", failure_kind="supervisor_interrupted",
                error="automatic outcome-blind finalization of orphaned recovery start",
            )
            previous = _append_event(ledger, terminal, header, previous)
            histories[decision_id] = [*history, terminal]
    launched = completed = failed = skipped = 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    for row in issuance["selected_rows"]:
        decision_id = row["decision_id"]
        history = histories.get(decision_id, [])
        if history:
            skipped += 1
            continue
        artifact, arrays = _row_paths(issuance, row)
        stdout_path, stderr_path = _row_log_paths(issuance, row)
        capability = _capability_path(issuance, row)
        consumed = _consumed_capability_path(capability)
        if any(path.exists() for path in (artifact, arrays, stdout_path, stderr_path, capability, consumed)):
            raise FileExistsError(f"recovery evidence exists without a ledger start: {decision_id}")
        started = {
            "decision_id": decision_id, "status": "started",
            "recovery_ordinal": row["recovery_ordinal"], "original_ordinal": row["original_ordinal"],
            "original_terminal_event_sha256": row["original_terminal_event_sha256"],
            "selected_row_sha256": canonical_sha256(row),
            "artifact_path": str(artifact.resolve()), "arrays_path": str(arrays.resolve()),
            "stdout_path": str(stdout_path.resolve()), "stderr_path": str(stderr_path.resolve()),
            "row_timeout_seconds": issuance["row_timeout_seconds"],
        }
        previous = _append_event(ledger, started, header, previous)
        capability = _create_capability(issuance_path, issuance, row, header, previous)
        launched += 1
        stdout_text = stderr_text = ""
        worker_returncode: int | None = None
        timed_out = False
        try:
            child = executor(
                _worker_command(issuance, capability), capture_output=True, text=True,
                env=env, cwd=str(PROJECT_ROOT), check=False,
                timeout=issuance["row_timeout_seconds"],
            )
            stdout_text, stderr_text = child.stdout or "", child.stderr or ""
            worker_returncode = child.returncode
            if child.returncode == 0 and artifact.is_file() and arrays.is_file() and consumed.is_file() and not capability.exists():
                terminal_status = "completed"
                completed += 1
            else:
                detail = (child.stderr or child.stdout or f"worker exited {child.returncode}").splitlines()
                terminal_status = "failed"
                failure_kind = "isolated_worker_failure"
                failure_error = detail[-1] if detail else f"worker exited {child.returncode}"
                failed += 1
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout_text = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr_text = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
            terminal_status = "failed"
            failure_kind = "isolated_worker_timeout"
            failure_error = f"worker exceeded {issuance['row_timeout_seconds']} seconds"
            failed += 1
        except Exception as exc:
            terminal_status = "failed"
            failure_kind = "isolated_worker_launch_failure"
            failure_error = str(exc)
            stderr_text = str(exc)
            failed += 1
        _exclusive_text(stdout_path, stdout_text)
        _exclusive_text(stderr_path, stderr_text)
        terminal_extra: dict[str, Any] = {
            "worker_returncode": worker_returncode, "timed_out": timed_out,
            "row_timeout_seconds": issuance["row_timeout_seconds"],
        }
        if terminal_status == "failed":
            terminal_extra.update({"failure_kind": failure_kind, "error": failure_error})
        terminal = _terminal_payload(issuance, row, terminal_status, **terminal_extra)
        previous = _append_event(ledger, terminal, header, previous)
        histories[decision_id] = [started, terminal]
    _header, final, _tail = _validate_recovery_ledger(issuance_path, issuance)
    expected = {row["decision_id"] for row in issuance["selected_rows"]}
    if set(final) != expected or any([item.get("status") for item in history] not in (["started", "completed"], ["started", "failed"]) for history in final.values()):
        raise ValueError("recovery ledger is not sealed with exactly one terminal attempt per issued row")
    total_failed = sum(history[-1].get("status") == "failed" for history in final.values())
    return {
        "status": "sealed_with_terminal_failures" if total_failed else "sealed",
        "selected": len(expected), "launched": launched, "completed_this_run": completed,
        "failed_this_run": failed, "skipped": skipped, "terminal_failed": total_failed,
        "ledger": str(ledger), "sealed": True,
    }


def run_recovery(
    issuance_path: Path | str,
    *, executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Acquire the process-held supervisor lock before touching the ledger."""
    issuance_path = Path(issuance_path).resolve()
    with _supervisor_lock(issuance_path.parent / ".supervisor.lock"):
        return _run_recovery_locked(issuance_path, executor=executor)


def _verify_receipt_files(receipt: Mapping[str, Any], label: str) -> None:
    for kind in ("artifact", "arrays"):
        exists = bool(receipt.get(f"{kind}_exists"))
        path = Path(str(receipt.get(f"{kind}_path", "")))
        if exists != path.is_file() or receipt.get(f"{kind}_sha256") != file_sha256(path):
            raise ValueError(f"{label} {kind} evidence changed or is missing")


def _verify_recovery_logs(receipt: Mapping[str, Any], label: str) -> None:
    for kind in ("stdout", "stderr"):
        path = Path(str(receipt.get(f"{kind}_path", "")))
        if receipt.get(f"{kind}_exists") is not True or not path.is_file() or receipt.get(f"{kind}_sha256") != file_sha256(path):
            raise ValueError(f"{label} {kind} log changed or is missing")


def build_merge_index(
    issuance_path: Path | str, *, output_path: Path | str | None = None,
    expected_nonselected_failures: int | None = EXPECTED_NONSELECTED_FAILURES,
) -> dict[str, Any]:
    """Build a 600-row provenance index; never discard an original failure."""
    issuance_path = Path(issuance_path).resolve()
    issuance = validate_recovery_issuance(issuance_path)
    manifest = _json(Path(issuance["sources"]["manifest"]["path"]))
    _original_header, original = _original_histories(manifest, Path(issuance["sources"]["original_ledger"]["path"]))
    _recovery_header, recovery, _tail = _validate_recovery_ledger(issuance_path, issuance)
    selected = {row["decision_id"]: row for row in issuance["selected_rows"]}
    if set(recovery) != set(selected):
        raise ValueError("recovery merge rejects missing or extra recovery rows")
    rows: list[dict[str, Any]] = []
    nonselected_failures = 0
    for ordinal, entry in enumerate(manifest["entries"]):
        decision_id = entry["decision_id"]
        original_terminal = original[decision_id][-1]
        record = {
            "ordinal": ordinal, "decision_id": decision_id,
            "manifest_entry_sha256": canonical_sha256(entry),
            "original_terminal_status": original_terminal["status"],
            "original_terminal_event_sha256": original_terminal["event_sha256"],
            "recovery_selected": decision_id in selected,
        }
        if original_terminal["status"] == "completed":
            _verify_receipt_files(original_terminal, f"original row {decision_id}")
            record.update({
                "effective_status": "completed", "valid": True, "evidence_source": "original_v1.4",
                "artifact_path": original_terminal["artifact_path"], "arrays_path": original_terminal["arrays_path"],
                "artifact_sha256": original_terminal["artifact_sha256"], "arrays_sha256": original_terminal["arrays_sha256"],
            })
        elif decision_id not in selected:
            nonselected_failures += 1
            record.update({
                "effective_status": "failed", "valid": False,
                "evidence_source": "original_v1.4_nonselected_failure",
                "failure_first_line": _first_error_line(original_terminal),
            })
        else:
            recovered = recovery[decision_id][-1]
            _verify_recovery_logs(recovered, f"recovery row {decision_id}")
            if recovered["status"] == "completed":
                _verify_receipt_files(recovered, f"recovery row {decision_id}")
                record.update({
                    "effective_status": "completed", "valid": True, "evidence_source": "v1.4-R",
                    "recovery_terminal_event_sha256": recovered["event_sha256"],
                    "artifact_path": recovered["artifact_path"], "arrays_path": recovered["arrays_path"],
                    "artifact_sha256": recovered["artifact_sha256"], "arrays_sha256": recovered["arrays_sha256"],
                })
            else:
                record.update({
                    "effective_status": "failed", "valid": False, "evidence_source": "v1.4-R_failed",
                    "recovery_terminal_event_sha256": recovered["event_sha256"],
                })
        rows.append(record)
    if expected_nonselected_failures is not None and nonselected_failures != expected_nonselected_failures:
        raise ValueError(f"merge preserved {nonselected_failures} nonselected failures, expected {expected_nonselected_failures}")
    if len(rows) != len(manifest["entries"]) or len({row["decision_id"] for row in rows}) != len(rows):
        raise ValueError("merge index contains duplicate or missing manifest rows")
    valid = sum(row["valid"] for row in rows)
    merge = {
        "schema_version": MERGE_SCHEMA, "status": "validated",
        "recovery_issuance": _issuance_reference(issuance_path, issuance),
        "original_manifest_sha256": issuance["sources"]["manifest"]["sha256"],
        "original_ledger_sha256": issuance["sources"]["original_ledger"]["sha256"],
        "recovery_ledger_sha256": file_sha256(Path(issuance["ledger_path"])),
        "counts": {
            "rows": len(rows), "valid": valid, "invalid": len(rows) - valid,
            "selected_for_recovery": len(selected), "nonselected_original_failures": nonselected_failures,
        },
        "rows": rows,
    }
    merge["merge_sha256"] = canonical_sha256(merge)
    if output_path is not None:
        _exclusive_json(Path(output_path).resolve(), merge)
    return merge


def _recovery_parent_launcher_attestation(issuance: Mapping[str, Any]) -> dict[str, Any]:
    """Bridge recovery capability use through the original frozen parent proof."""
    record = issuance.get("parent_launcher_attestation")
    if not isinstance(record, Mapping) or record.get("target_module") != "experiments.official_pusht_v14_batch_study" or record.get("status") != "consumed":
        raise ValueError("frozen recovery parent launcher attestation is invalid")
    return dict(record)


def _worker(capability_path: Path) -> int:
    issuance, row, consumed_capability = _consume_capability(capability_path)
    artifact_path, arrays_path = _row_paths(issuance, row)
    if artifact_path.exists() or arrays_path.exists():
        raise FileExistsError("recovery worker output already exists")
    from experiments import official_pusht_v14_decision_audit as worker
    from experiments.official_pusht_v14_common_initial import validate_v14_artifact

    recovery_reference = _issuance_reference(Path(str(issuance["recovery_root"])) / "recovery_issuance.json", issuance)
    parent_launcher_attestation = _recovery_parent_launcher_attestation(issuance)

    def consume_recovery(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return recovery_reference, consumed_capability

    worker._consume_study_row_capability = consume_recovery
    model_spec = issuance["models"][row["model"]]
    dino = issuance["dinov2"]
    artifact = worker.run_one_decision(
        official_root=Path(issuance["official_root"]), checkpoint=Path(model_spec["checkpoint"]),
        config=Path(model_spec["config"]), dataset_root=Path(issuance["dataset_root"]),
        dataset_split=issuance["dataset_split"], episode=int(row["episode"]), offset=int(row["offset"]),
        seed=int(row["seed"]), arrays_out=arrays_path, repeat_count=1, planning_repeats=0,
        inventory_mode="common_initial_only", model_name=row["model"], decision_id=row["decision_id"],
        trajectory_id=row["trajectory_id"], manifest_sha256=issuance["sources"]["manifest"]["canonical_sha256"],
        dinov2_source_root=Path(dino["source_root"]), dinov2_source_archive=Path(dino["source_archive"]),
        dinov2_weights=Path(dino["weights"]), torch_home=Path(dino["torch_home"]),
        row_capability_path=Path(capability_path), parent_launcher_attestation=parent_launcher_attestation,
    )
    artifact["reproducibility_certificate"] = {
        "status": "passed", "model": row["model"], "sha256": row["certificate_sha256"],
    }
    artifact["certificate_issuance"] = issuance["certificate_issuance_reference"]
    errors = validate_v14_artifact(
        artifact, row["manifest_entry"], arrays_path=arrays_path,
        expected_bindings=row["row_bindings"],
        expected_certificate_issuance=issuance["certificate_issuance_reference"],
        expected_certificate_sha256=row["certificate_sha256"],
        expected_study_issuance=recovery_reference,
        expected_row_capability=consumed_capability,
        expected_menu_sha256=row["certified_menu_sha256"],
    )
    if errors:
        raise ValueError("recovery artifact validation failed: " + "; ".join(errors))
    _exclusive_json(artifact_path, artifact)
    print(json.dumps({"status": "completed", "decision_id": row["decision_id"]}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="freeze the 169-row recovery issuance only")
    prepare.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--original-ledger", type=Path, default=DEFAULT_LEDGER)
    prepare.add_argument("--study-issuance", type=Path, default=DEFAULT_STUDY_ISSUANCE)
    prepare.add_argument("--certificate-issuance", type=Path, default=DEFAULT_CERTIFICATE_ISSUANCE)
    prepare.add_argument("--certificate-bundle", type=Path, default=DEFAULT_CERTIFICATE_BUNDLE)
    prepare.add_argument("--recovery-addendum", type=Path, default=DEFAULT_RECOVERY_ADDENDUM)
    run = sub.add_parser("run", help="run/resume the issued rows once")
    run.add_argument("--issuance", type=Path, required=True)
    merge = sub.add_parser("merge", help="write the deterministic 600-row provenance index")
    merge.add_argument("--issuance", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    worker_parser = sub.add_parser("_worker")
    worker_parser.add_argument("--capability", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path, issuance = prepare_recovery(
            manifest_path=args.manifest, original_ledger_path=args.original_ledger,
            study_issuance_path=args.study_issuance, certificate_issuance_path=args.certificate_issuance,
            certificate_bundle_path=args.certificate_bundle, recovery_addendum_path=args.recovery_addendum,
            recovery_root=args.recovery_root,
        )
        print(json.dumps({"status": "prepared", "issuance": str(path), "selected": len(issuance["selected_rows"])}, sort_keys=True))
    elif args.command == "run":
        print(json.dumps(run_recovery(args.issuance), sort_keys=True))
    elif args.command == "merge":
        result = build_merge_index(args.issuance, output_path=args.output)
        print(json.dumps({"status": "validated", "output": str(args.output.resolve()), "counts": result["counts"]}, sort_keys=True))
    else:
        return _worker(args.capability)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CUDA_ILLEGAL_ADDRESS", "select_cuda_failures", "prepare_recovery",
    "validate_recovery_issuance", "run_recovery", "build_merge_index", "main",
]
