"""Validated v1.4-R evidence adapter for the frozen v1.4 analyzer.

The adapter changes only evidence resolution.  Every statistical function,
validity threshold, multiplicity correction, and claim gate is executed by
``analyze_official_pusht_v14.py`` unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import analyze_official_pusht_v14 as frozen
from experiments import official_pusht_v14_batch_study as batch
from experiments import official_pusht_v14_certificate_supervisor as certificate
from experiments import official_pusht_v14_common_initial as common
from experiments import official_pusht_v14_recovery as recovery
from experiments.official_pusht_v14_common_initial import V14_CROSS_MODEL_ARRAY_KEYS


BRIDGE_SCHEMA = "PWA-PushT-v1.4-R-analysis-bridge-v1"
CHECKPOINT_SCHEMA = "PWA-PushT-v1.4-R-analysis-checkpoint-v1"
CHECKPOINT_HEADER_SCHEMA = "PWA-PushT-v1.4-R-analysis-checkpoint-header-v1"


class _DigestCache:
    """Per-analysis file digest cache; only row payload bytes are retained."""

    def __init__(self) -> None:
        self._hashes: dict[Path, str | None] = {}
        self._bytes: dict[Path, bytes] = {}

    def read(self, path: Path | str, *, retain: bool = False) -> bytes:
        path = Path(path).resolve()
        if path in self._bytes:
            return self._bytes[path]
        payload = path.read_bytes()
        self._hashes[path] = hashlib.sha256(payload).hexdigest()
        if retain:
            self._bytes[path] = payload
        return payload

    def digest(self, path: Path | str) -> str | None:
        path = Path(path).resolve()
        if path in self._hashes:
            return self._hashes[path]
        if not path.is_file():
            self._hashes[path] = None
            return None
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self._hashes[path] = digest
        return digest


_ACTIVE_CACHE: _DigestCache | None = None
_ACTIVE_ARRAY_SNAPSHOTS: dict[Path, dict[str, Any]] | None = None
_ACTIVE_ARRAY_HASHES: dict[str, dict[str, str]] | None = None
_ACTIVE_ARRAY_SHAPES: dict[str, dict[str, dict[str, Any]]] | None = None
_ACTIVE_CHECKPOINTS: "_CheckpointStore | None" = None


class _MemoryArchive:
    def __init__(self, arrays: Mapping[str, Any]) -> None:
        self._arrays = arrays
        self.files = tuple(arrays)

    def __getitem__(self, key: str) -> Any:
        return self._arrays[key]

    def __enter__(self) -> "_MemoryArchive":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def close(self) -> None:
        return None


def _snapshot_np_load(file: Any, *args: Any, **kwargs: Any) -> Any:
    snapshots = _ACTIVE_ARRAY_SNAPSHOTS
    if snapshots is not None and isinstance(file, (str, bytes, os.PathLike)):
        path = Path(file).resolve()
        if path in snapshots:
            return _MemoryArchive(snapshots[path])
    return _ORIGINAL_NP_LOAD(file, *args, **kwargs)


_ORIGINAL_NP_LOAD = frozen.np.load


@contextlib.contextmanager
def _source_context() -> Any:
    """Share digest validation and in-memory sidecars across imported modules."""
    global _ACTIVE_CACHE, _ACTIVE_ARRAY_SNAPSHOTS, _ACTIVE_ARRAY_HASHES, _ACTIVE_ARRAY_SHAPES, _ACTIVE_CHECKPOINTS
    cache = _DigestCache()
    snapshots: dict[Path, dict[str, Any]] = {}
    hashes: dict[str, dict[str, str]] = {}
    shapes: dict[str, dict[str, dict[str, Any]]] = {}
    previous = (_ACTIVE_CACHE, _ACTIVE_ARRAY_SNAPSHOTS, _ACTIVE_ARRAY_HASHES, _ACTIVE_ARRAY_SHAPES, _ACTIVE_CHECKPOINTS)
    _ACTIVE_CACHE, _ACTIVE_ARRAY_SNAPSHOTS, _ACTIVE_ARRAY_HASHES, _ACTIVE_ARRAY_SHAPES, _ACTIVE_CHECKPOINTS = cache, snapshots, hashes, shapes, None
    modules = (common, batch, certificate, recovery, frozen)
    originals = [module.file_sha256 for module in modules]
    original_np_load = frozen.np.load
    for module in modules:
        module.file_sha256 = cache.digest
    frozen.np.load = _snapshot_np_load
    try:
        yield cache
    finally:
        frozen.np.load = original_np_load
        for module, original in zip(modules, originals):
            module.file_sha256 = original
        _ACTIVE_CACHE, _ACTIVE_ARRAY_SNAPSHOTS, _ACTIVE_ARRAY_HASHES, _ACTIVE_ARRAY_SHAPES, _ACTIVE_CHECKPOINTS = previous


def _file_sha256(path: Path | str) -> str | None:
    if _ACTIVE_CACHE is not None:
        return _ACTIVE_CACHE.digest(path)
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if _ACTIVE_CACHE is not None:
        value = json.loads(_ACTIVE_CACHE.read(path).decode("utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_json_snapshot(path: Path, cache: _DigestCache) -> tuple[dict[str, Any], str]:
    payload = cache.read(path, retain=True)
    digest = cache.digest(path)
    if digest is None:
        raise ValueError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, digest


def _read_npz_snapshot(
    path: Path, cache: _DigestCache,
) -> tuple[dict[str, Any], str, dict[str, str], dict[str, dict[str, Any]]]:
    payload = cache.read(path, retain=True)
    digest = cache.digest(path)
    if digest is None:
        raise ValueError(f"missing NPZ sidecar: {path}")
    arrays: dict[str, Any] = {}
    try:
        with _ORIGINAL_NP_LOAD(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: frozen.np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"invalid NPZ sidecar: {path}") from exc
    return arrays, digest, {
        key: frozen.array_sha256(value) for key, value in sorted(arrays.items())
    }, {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }


def _atomic_create_payload(path: Path, payload: bytes) -> None:
    """Publish a file atomically without ever replacing a committed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _CheckpointStore:
    def __init__(self, root: Path, header: Mapping[str, Any]) -> None:
        self.root = Path(root).resolve()
        self.header_path = self.root / "header.json"
        self.row_dir = self.root / "rows"
        self.header = dict(header)
        self.header_sha256 = str(self.header["header_sha256"])
        self.records: dict[int, dict[str, Any]] = {}
        self._open()

    def _open(self) -> None:
        if self.header_sha256 != _canonical_without(self.header, "header_sha256"):
            raise ValueError("checkpoint header commitment mismatch")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.header_path.exists():
            actual = _checkpoint_json(self.header_path)
            advertised = actual.get("header_sha256")
            if advertised != _canonical_without(actual, "header_sha256"):
                raise ValueError("checkpoint header commitment mismatch")
            if actual != self.header:
                raise ValueError("checkpoint header does not match this sealed analysis")
        else:
            _exclusive_json(self.header_path, self.header)
        self.row_dir.mkdir(parents=True, exist_ok=True)
        self._load_records()

    def _load_records(self) -> None:
        files = {path.name for path in self.row_dir.iterdir() if path.is_file()}
        expected_names = {f"row-{ordinal:06d}.json" for ordinal in range(int(self.header["row_count"]))}
        unexpected = sorted(files - expected_names)
        if unexpected:
            raise ValueError("checkpoint directory contains unexpected row files")
        previous_sha = None
        for ordinal in range(int(self.header["row_count"])):
            path = self.row_dir / f"row-{ordinal:06d}.json"
            if not path.exists():
                if any(name > path.name for name in files):
                    raise ValueError("checkpoint hash chain has a gap")
                break
            record = _checkpoint_json(path)
            self._validate_record(record, ordinal, previous_sha)
            self.records[ordinal] = record
            previous_sha = record["checkpoint_sha256"]

    def _validate_record(self, record: Mapping[str, Any], ordinal: int, previous_sha: str | None) -> None:
        if record.get("schema_version") != CHECKPOINT_SCHEMA or record.get("header_sha256") != self.header_sha256:
            raise ValueError(f"checkpoint row {ordinal} header binding mismatch")
        if record.get("ordinal") != ordinal or record.get("decision_id") != self.header["decision_ids"][ordinal]:
            raise ValueError(f"checkpoint row {ordinal} identity mismatch")
        expected_entry_hashes = self.header.get("manifest_entry_sha256s", [])
        if not isinstance(expected_entry_hashes, list) or len(expected_entry_hashes) != int(self.header["row_count"]) or record.get("manifest_entry_sha256") != expected_entry_hashes[ordinal]:
            raise ValueError(f"checkpoint row {ordinal} manifest binding mismatch")
        if record.get("previous_checkpoint_sha256") != previous_sha:
            raise ValueError(f"checkpoint row {ordinal} hash-chain mismatch")
        advertised = record.get("checkpoint_sha256")
        if advertised != _canonical_without(record, "checkpoint_sha256"):
            raise ValueError(f"checkpoint row {ordinal} commitment mismatch")
        if record.get("status") not in {"valid", "invalid"} or not isinstance(record.get("row"), Mapping):
            raise ValueError(f"checkpoint row {ordinal} has an invalid state")
        if record["row"].get("decision_id") != record.get("decision_id"):
            raise ValueError(f"checkpoint row {ordinal} row identity mismatch")
        invalid = record.get("invalid_records")
        if not isinstance(invalid, list) or any(not isinstance(item, Mapping) for item in invalid):
            raise ValueError(f"checkpoint row {ordinal} invalid-record payload mismatch")
        if record.get("status") == "valid":
            if not _is_digest(record.get("artifact_sha256")) or not _is_digest(record.get("arrays_sha256")):
                raise ValueError(f"checkpoint row {ordinal} source commitment missing")
            array_hashes = record.get("array_hashes")
            array_shapes = record.get("array_shapes")
            if (
                not isinstance(array_hashes, Mapping)
                or any(not _is_digest(value) for value in array_hashes.values())
                or any(key not in array_hashes for key in V14_CROSS_MODEL_ARRAY_KEYS)
                or not isinstance(array_shapes, Mapping)
                or any(key not in array_shapes for key in V14_CROSS_MODEL_ARRAY_KEYS)
                or any(
                    not isinstance(value, Mapping)
                    or not isinstance(value.get("shape"), list)
                    or not all(isinstance(dim, int) and dim >= 0 for dim in value["shape"])
                    or not isinstance(value.get("dtype"), str)
                    for value in array_shapes.values()
                )
            ):
                raise ValueError(f"checkpoint row {ordinal} array commitments missing")
        elif (
            record.get("array_hashes") != {}
            or record.get("array_shapes") != {}
            or record.get("artifact_sha256") is not None
            or record.get("arrays_sha256") is not None
        ):
            raise ValueError(f"checkpoint row {ordinal} invalid array commitments")

    def write(self, *, ordinal: int, decision_id: str, entry: Mapping[str, Any], status: str,
              row: Mapping[str, Any], invalid_records: Sequence[Mapping[str, Any]],
              artifact_sha256: str | None = None, arrays_sha256: str | None = None,
              array_hashes: Mapping[str, str] | None = None,
              array_shapes: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        if ordinal != len(self.records):
            raise ValueError("checkpoint rows must be committed contiguously")
        previous = self.records.get(ordinal - 1)
        record: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA, "header_sha256": self.header_sha256,
            "ordinal": ordinal, "decision_id": decision_id,
            "manifest_entry_sha256": frozen.canonical_sha256(entry),
            "previous_checkpoint_sha256": previous.get("checkpoint_sha256") if previous else None,
            "status": status, "row": dict(row), "invalid_records": [dict(item) for item in invalid_records],
            "artifact_sha256": artifact_sha256, "arrays_sha256": arrays_sha256,
            "array_hashes": dict(array_hashes or {}),
            "array_shapes": {key: dict(value) for key, value in (array_shapes or {}).items()},
        }
        record["checkpoint_sha256"] = _canonical_without(record, "checkpoint_sha256")
        path = self.row_dir / f"row-{ordinal:06d}.json"
        _exclusive_json(path, record)
        self.records[ordinal] = record
        return record


def _canonical_without(value: Mapping[str, Any], key: str) -> str:
    return frozen.canonical_sha256({name: item for name, item in value.items() if name != key})


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _seed_committed_digest(cache: _DigestCache, path: Path | str, digest: str) -> None:
    """Bind a completed checkpoint's source digest without reopening its bytes."""
    resolved = Path(path).resolve()
    if resolved in cache._hashes and cache._hashes[resolved] != digest:
        raise ValueError(f"checkpoint source commitment differs from cached source: {resolved}")
    cache._hashes[resolved] = digest


def _checkpoint_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"corrupt checkpoint record: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint record is not a JSON object: {path}")
    return value


@contextlib.contextmanager
def _deferred_merge_content_hashes(merge: Mapping[str, Any]) -> Any:
    """Use committed row hashes during structural merge validation.

    ``build_merge_index`` proves the deterministic row selection and all
    control-file receipts.  The large row payloads are bound by the supplied
    merge index here and are read exactly once, into memory, by the row
    resolver below.  A missing payload still returns ``None`` after an
    existence check, preserving the validator's missing-file behavior.
    """
    deferred: dict[Path, str] = {}
    rows = merge.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping) or row.get("valid") is not True:
                continue
            for path_key, digest_key in (("artifact_path", "artifact_sha256"), ("arrays_path", "arrays_sha256")):
                path_value, digest = row.get(path_key), row.get(digest_key)
                if not isinstance(path_value, str) or not _is_digest(digest):
                    continue
                path = Path(path_value).resolve()
                previous = deferred.get(path)
                if previous is not None and previous != digest:
                    raise ValueError(f"merged row content binding is inconsistent for {path}")
                deferred[path] = digest
    original = recovery.file_sha256

    def deferred_file_sha256(path: Path | str) -> str | None:
        resolved = Path(path).resolve()
        advertised = deferred.get(resolved)
        if advertised is not None:
            return advertised if resolved.is_file() else None
        return original(path)

    recovery.file_sha256 = deferred_file_sha256
    try:
        yield
    finally:
        recovery.file_sha256 = original


def validate_merge_index(merge_path: Path | str, issuance_path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require byte-current source evidence and exact deterministic merge output."""
    merge_path, issuance_path = Path(merge_path).resolve(), Path(issuance_path).resolve()
    supplied = _json(merge_path)
    with _deferred_merge_content_hashes(supplied):
        issuance = recovery.validate_recovery_issuance(issuance_path)
        expected = recovery.build_merge_index(issuance_path)
    if supplied != expected:
        raise ValueError("merged evidence differs from the deterministic v1.4-R merge validator")
    if supplied.get("merge_sha256") != recovery.canonical_sha256({key: value for key, value in supplied.items() if key != "merge_sha256"}):
        raise ValueError("merged evidence commitment mismatch")
    reference = supplied.get("recovery_issuance")
    if not isinstance(reference, Mapping) or reference.get("path") != str(issuance_path) or reference.get("sha256") != _file_sha256(issuance_path) or reference.get("issuance_sha256") != issuance.get("issuance_sha256"):
        raise ValueError("merged evidence recovery-issuance binding mismatch")
    return supplied, issuance


def _recovery_terminal_histories(
    issuance_path: Path, issuance: Mapping[str, Any], validate: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    if validate:
        _header, histories, _tail = recovery._validate_recovery_ledger(issuance_path, issuance)
    else:
        _header, histories, _tail = recovery._read_ledger(Path(str(issuance["ledger_path"])))
    return histories


def _consumed_recovery_capability(terminal: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(terminal.get("capability_path", ""))).resolve()
    digest = _file_sha256(path)
    if digest is None or digest != terminal.get("capability_sha256"):
        raise ValueError("recovery terminal capability changed or is missing")
    record = _json(path)
    advertised = record.get("capability_sha256")
    if advertised != recovery.canonical_sha256({key: value for key, value in record.items() if key != "capability_sha256"}):
        raise ValueError("recovery terminal capability commitment mismatch")
    selected = record.get("selected_row")
    if (
        not isinstance(selected, Mapping)
        or recovery.canonical_sha256(selected) != record.get("selected_row_sha256")
        or selected.get("decision_id") != terminal.get("decision_id")
        or record.get("artifact_path") != terminal.get("artifact_path")
        or record.get("arrays_path") != terminal.get("arrays_path")
    ):
        raise ValueError("recovery terminal capability row binding mismatch")
    return {"status": "consumed", "path": str(path), "sha256": digest, "capability_sha256": advertised}


def _invalid_record(merge_row: Mapping[str, Any], terminal: Mapping[str, Any] | None) -> dict[str, Any]:
    record = {
        "decision_id": merge_row["decision_id"],
        "reason": "terminal_failed_attempt_preregistered_zero",
        "evidence_source": merge_row.get("evidence_source"),
        "original_terminal_status": merge_row.get("original_terminal_status"),
        "original_terminal_event_sha256": merge_row.get("original_terminal_event_sha256"),
        "recovery_selected": merge_row.get("recovery_selected"),
        "recovery_terminal_event_sha256": merge_row.get("recovery_terminal_event_sha256"),
    }
    if isinstance(terminal, Mapping):
        record["recovery_failure_kind"] = terminal.get("failure_kind")
        error = terminal.get("error")
        record["recovery_failure_first_line"] = error.splitlines()[0] if isinstance(error, str) and error.splitlines() else None
    else:
        record["original_failure_first_line"] = merge_row.get("failure_first_line")
    return record


def _resolve_completed_row(
    merge_row: Mapping[str, Any], entry: Mapping[str, Any], issuance: Mapping[str, Any],
    issuance_path: Path, bundle: Mapping[str, Any], certificate_hashes: Mapping[str, Any],
    certificates: Mapping[str, Any], certificate_issuance: Mapping[str, Any] | None,
    issuance_record: Mapping[str, Any] | None, certified_menus: Mapping[Any, Any],
    original_study_reference: Mapping[str, Any], recovery_reference: Mapping[str, Any],
    histories: Mapping[str, list[dict[str, Any]]], manifest_sha256: str,
    cache: _DigestCache | None,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], str | None, str | None,
    dict[str, str] | None, dict[str, dict[str, Any]] | None,
]:
    decision_id = str(entry["decision_id"])
    artifact_path = Path(str(merge_row.get("artifact_path", ""))).resolve()
    arrays_path = Path(str(merge_row.get("arrays_path", ""))).resolve()
    try:
        if cache is None:
            artifact_sha256 = _file_sha256(artifact_path)
            arrays_sha256 = _file_sha256(arrays_path)
            if artifact_sha256 != merge_row.get("artifact_sha256") or arrays_sha256 != merge_row.get("arrays_sha256"):
                raise ValueError(f"completed merged row {decision_id} changed after merge validation")
            artifact = _json(artifact_path)
            array_snapshot, accepted_array_hashes = None, None
        else:
            artifact, artifact_sha256 = _read_json_snapshot(artifact_path, cache)
            array_snapshot, arrays_sha256, accepted_array_hashes, accepted_array_shapes = _read_npz_snapshot(arrays_path, cache)
            if artifact_sha256 != merge_row.get("artifact_sha256") or arrays_sha256 != merge_row.get("arrays_sha256"):
                raise ValueError(f"completed merged row {decision_id} changed after merge validation")
            missing_array_bindings = [
                key for key in V14_CROSS_MODEL_ARRAY_KEYS
                if key not in accepted_array_hashes or key not in accepted_array_shapes
            ]
            if missing_array_bindings:
                raise ValueError(
                    f"completed merged row {decision_id} lacks accepted array commitments: "
                    + ", ".join(missing_array_bindings)
                )
            if _ACTIVE_ARRAY_SNAPSHOTS is not None:
                _ACTIVE_ARRAY_SNAPSHOTS[arrays_path] = array_snapshot
        if cache is None:
            accepted_array_shapes = None
        advertised = artifact.get("arrays_sidecar")
        if not isinstance(advertised, str) or Path(advertised).resolve() != arrays_path:
            raise ValueError(f"completed merged row {decision_id} points to a different sidecar")
        model = str(entry.get("model"))
        bindings = frozen._expected_bindings_for_entry(
            bundle, model, entry, certificates.get(model), issuance_record, manifest_sha256,
        )
        bindings.update(frozen._retained_npz_bindings(arrays_path))
        missing = frozen._missing_full_bindings(bindings)
        if missing:
            raise ValueError(f"completed merged row {decision_id} lacks bindings: {', '.join(missing)}")
        is_recovery = merge_row.get("evidence_source") == "v1.4-R"
        expected_capability = None
        if is_recovery:
            recovery_terminal = histories.get(decision_id, [None])[-1] if decision_id in histories else None
            if not isinstance(recovery_terminal, Mapping) or recovery_terminal.get("status") != "completed":
                raise ValueError(f"completed merged row {decision_id} lacks a recovery terminal receipt")
            expected_capability = _consumed_recovery_capability(recovery_terminal)
        errors = frozen.validate_v14_artifact(
            artifact, entry, arrays_path=arrays_path, expected_bindings=bindings,
            expected_certificate_issuance=certificate_issuance,
            expected_certificate_sha256=certificate_hashes.get(model),
            expected_study_issuance=recovery_reference if is_recovery else original_study_reference,
            expected_row_capability=expected_capability,
            expected_menu_sha256=certified_menus.get(int(entry["seed"])),
        )
        if artifact.get("v1_2_evidence_reused") is not False:
            errors.append("v1.4 artifact must explicitly disclose no v1.2 evidence reuse")
        if errors:
            raise ValueError(f"completed merged row {decision_id} is invalid: {'; '.join(errors)}")
        row = frozen._recomputed_row(artifact, entry, arrays_path)
        row_invalid: list[dict[str, Any]] = []
        if not row["raw_valid"] or not row["support_valid"]:
            row_invalid.append({
                "decision_id": decision_id, "evidence_source": merge_row.get("evidence_source"),
                "raw_valid": row["raw_valid"], "support_valid": row["support_valid"],
            })
        return row, row_invalid, artifact_sha256, arrays_sha256, accepted_array_hashes, accepted_array_shapes
    finally:
        if cache is not None:
            cache._bytes.pop(artifact_path, None)
            cache._bytes.pop(arrays_path, None)
        if _ACTIVE_ARRAY_SNAPSHOTS is not None:
            _ACTIVE_ARRAY_SNAPSHOTS.pop(arrays_path, None)


def _read_merged_rows(
    merge: Mapping[str, Any], issuance_path: Path, issuance: Mapping[str, Any],
    manifest: Mapping[str, Any], bundle: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merge_rows = merge.get("rows")
    entries = manifest.get("entries")
    if not isinstance(merge_rows, list) or not isinstance(entries, list) or len(merge_rows) != len(entries):
        raise ValueError("merged evidence does not contain exactly one row per manifest entry")
    certificate_hashes = bundle.get("certificate_sha256")
    certificates = bundle.get("certificates")
    if not isinstance(certificate_hashes, Mapping) or not isinstance(certificates, Mapping):
        raise ValueError("certificate bundle lacks analyzer model certificates")
    certificate_issuance = frozen._issuance_reference(bundle, "certificate_issuance")
    issuance_record = frozen._load_issuance_record(certificate_issuance)
    certified_menus = frozen._certified_menu_hashes(bundle)
    original_study = issuance["sources"]["study_issuance"]
    original_study_reference = {
        "path": original_study["path"], "sha256": original_study["sha256"],
        "issuance_sha256": original_study["issuance_sha256"],
    }
    recovery_reference = dict(merge["recovery_issuance"])
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    manifest_sha256 = frozen.canonical_sha256(manifest)
    cache, checkpoints = _ACTIVE_CACHE, _ACTIVE_CHECKPOINTS
    accepted_hashes, accepted_shapes = _ACTIVE_ARRAY_HASHES, _ACTIVE_ARRAY_SHAPES
    # The structural merge validator has already checked the complete
    # recovery ledger.  Read its parsed histories here without rehashing the
    # ledger's row payloads (or revalidating every recovery receipt).
    histories = _recovery_terminal_histories(issuance_path, issuance, False)
    for ordinal, (entry, merge_row) in enumerate(zip(entries, merge_rows)):
        decision_id = str(entry["decision_id"])
        if merge_row.get("ordinal") != ordinal or merge_row.get("decision_id") != decision_id or merge_row.get("manifest_entry_sha256") != frozen.canonical_sha256(entry):
            raise ValueError(f"merged row {ordinal} differs from the original manifest")
        if checkpoints is not None and ordinal in checkpoints.records:
            committed = checkpoints.records[ordinal]
            if committed.get("manifest_entry_sha256") != frozen.canonical_sha256(entry):
                raise ValueError(f"checkpoint row {ordinal} differs from the original manifest")
            if (committed.get("status") == "valid") != (merge_row.get("valid") is True):
                raise ValueError(f"checkpoint row {ordinal} status differs from the merge index")
            if committed.get("status") == "valid" and (
                committed.get("artifact_sha256") != merge_row.get("artifact_sha256")
                or committed.get("arrays_sha256") != merge_row.get("arrays_sha256")
            ):
                raise ValueError(f"checkpoint row {ordinal} source commitment differs from the merge index")
            row = dict(committed["row"])
            if row.get("decision_id") != decision_id:
                raise ValueError(f"checkpoint row {ordinal} decision binding mismatch")
            rows.append(row)
            invalid.extend(dict(item) for item in committed["invalid_records"])
            if committed.get("status") == "valid" and accepted_hashes is not None:
                accepted_hashes[decision_id] = dict(committed["array_hashes"])
            if committed.get("status") == "valid" and accepted_shapes is not None:
                accepted_shapes[decision_id] = {
                    key: dict(value) for key, value in committed["array_shapes"].items()
                }
            if committed.get("status") == "valid" and cache is not None:
                _seed_committed_digest(cache, merge_row["artifact_path"], committed["artifact_sha256"])
                _seed_committed_digest(cache, merge_row["arrays_path"], committed["arrays_sha256"])
            continue
        recovery_terminal = histories.get(decision_id, [None])[-1] if decision_id in histories else None
        if merge_row.get("valid") is not True:
            row = frozen._zero_row(entry, "terminal_failed_attempt_preregistered_zero")
            invalid_record = _invalid_record(merge_row, recovery_terminal if merge_row.get("recovery_selected") else None)
            rows.append(row)
            invalid.append(invalid_record)
            if checkpoints is not None:
                checkpoints.write(
                    ordinal=ordinal, decision_id=decision_id, entry=entry, status="invalid", row=row,
                    invalid_records=[invalid_record],
                )
            continue
        row, row_invalid, artifact_sha256, arrays_sha256, accepted_array_hashes, accepted_array_shapes = _resolve_completed_row(
            merge_row, entry, issuance, issuance_path, bundle, certificate_hashes, certificates,
            certificate_issuance, issuance_record, certified_menus, original_study_reference,
            recovery_reference, histories, manifest_sha256, cache,
        )
        rows.append(row)
        if accepted_hashes is not None and accepted_array_hashes is not None:
            accepted_hashes[decision_id] = accepted_array_hashes
        if accepted_shapes is not None and accepted_array_shapes is not None:
            accepted_shapes[decision_id] = accepted_array_shapes
        invalid.extend(row_invalid)
        if checkpoints is not None:
            checkpoints.write(
                ordinal=ordinal, decision_id=decision_id, entry=entry, status="valid", row=row,
                invalid_records=row_invalid, artifact_sha256=artifact_sha256,
                arrays_sha256=arrays_sha256, array_hashes=accepted_array_hashes,
                array_shapes=accepted_array_shapes,
            )
    return rows, invalid


def _paired_merge_errors(
    merge: Mapping[str, Any], manifest: Mapping[str, Any],
    array_hashes: Mapping[str, Mapping[str, str]] | None = None,
    array_shapes: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> list[str]:
    by_id = {row["decision_id"]: row for row in merge["rows"]}
    groups: dict[tuple[Any, Any, Any], dict[str, tuple[str, Path]]] = {}
    for entry in manifest["entries"]:
        row = by_id[str(entry["decision_id"])]
        if row.get("valid") is not True:
            continue
        groups.setdefault((entry.get("trajectory_id"), entry.get("offset"), entry.get("seed")), {})[str(entry.get("model"))] = (
            str(entry["decision_id"]), Path(row["arrays_path"]),
        )
    errors: list[str] = []
    if array_hashes is None:
        array_hashes = _ACTIVE_ARRAY_HASHES
    if array_shapes is None:
        array_shapes = _ACTIVE_ARRAY_SHAPES
    for cell, models in groups.items():
        if set(models) != set(frozen.V14_LOCKED_MODELS):
            continue
        if array_hashes is not None:
            left_id, _left_path = models[frozen.V14_LOCKED_MODELS[0]]
            right_id, _right_path = models[frozen.V14_LOCKED_MODELS[1]]
            left, right = array_hashes.get(left_id), array_hashes.get(right_id)
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                errors.append(f"{cell}: paired merged sidecar lacks accepted array hashes")
                continue
            left_shapes = array_shapes.get(left_id) if array_shapes is not None else None
            right_shapes = array_shapes.get(right_id) if array_shapes is not None else None
            if array_shapes is None:
                errors.append(f"{cell}: paired merged sidecar lacks accepted array shapes")
                continue
            for key in V14_CROSS_MODEL_ARRAY_KEYS:
                if key not in left or key not in right:
                    errors.append(f"{cell}: paired merged {key} lacks an accepted array hash")
                elif left.get(key) != right.get(key):
                    errors.append(f"{cell}: paired merged v1.4 {key} differs between JEPA and DINO")
                if not isinstance(left_shapes, Mapping) or not isinstance(right_shapes, Mapping):
                    errors.append(f"{cell}: paired merged sidecar lacks accepted array shapes")
                    break
                if key not in left_shapes or key not in right_shapes:
                    errors.append(f"{cell}: paired merged {key} lacks an accepted array shape")
                elif left_shapes[key] != right_shapes[key]:
                    errors.append(f"{cell}: paired merged v1.4 {key} shape differs between JEPA and DINO")
            continue
        loaded: dict[str, dict[str, Any]] = {}
        for model, (_decision_id, path) in models.items():
            try:
                with frozen.np.load(path, allow_pickle=False) as archive:
                    loaded[model] = {key: frozen.np.asarray(archive[key]) for key in V14_CROSS_MODEL_ARRAY_KEYS}
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{cell}: unreadable merged paired sidecar ({exc})")
        if len(loaded) == 2:
            left, right = (loaded[model] for model in frozen.V14_LOCKED_MODELS)
            for key in V14_CROSS_MODEL_ARRAY_KEYS:
                if not frozen.np.array_equal(left[key], right[key]):
                    errors.append(f"{cell}: paired merged v1.4 {key} differs between JEPA and DINO")
    return sorted(set(errors))


def _validate_checkpoint_dir(
    checkpoint_dir: Path, issuance: Mapping[str, Any], *protected_paths: Path,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    sealed_roots = [Path(str(issuance.get(key, ""))).resolve() for key in ("recovery_root", "output_dir") if issuance.get(key)]
    sources = issuance.get("sources", {})
    if isinstance(sources, Mapping):
        sealed_roots.extend(
            Path(str(source["path"])).resolve()
            for source in sources.values()
            if isinstance(source, Mapping) and isinstance(source.get("path"), str)
        )
    sealed_roots.extend(Path(path).resolve() for path in protected_paths)
    if any(root == checkpoint_dir or root in checkpoint_dir.parents or checkpoint_dir in root.parents for root in sealed_roots):
        raise ValueError("checkpoint directory must be outside immutable sealed evidence")
    return checkpoint_dir


def _checkpoint_header(
    merge_path: Path, issuance_path: Path, merge: Mapping[str, Any],
    issuance: Mapping[str, Any], manifest: Mapping[str, Any], bundle: Mapping[str, Any],
) -> dict[str, Any]:
    sources = issuance.get("sources", {})
    bundle_source = sources.get("certificate_bundle", {}) if isinstance(sources, Mapping) else {}
    study_source = sources.get("study_issuance", {}) if isinstance(sources, Mapping) else {}
    source_files = {
        name: {
            "path": source.get("path"),
            "sha256": _file_sha256(source.get("path")),
        }
        for name, source in sorted(sources.items())
        if isinstance(source, Mapping) and isinstance(source.get("path"), str)
    } if isinstance(sources, Mapping) else {}
    core: dict[str, Any] = {
        "schema_version": CHECKPOINT_HEADER_SCHEMA, "bridge_schema_version": BRIDGE_SCHEMA,
        "cache_trust": "unauthenticated_resume_cache",
        "merge_index_path": str(merge_path), "merge_index_file_sha256": _file_sha256(merge_path),
        "merge_sha256": merge.get("merge_sha256"), "recovery_issuance_path": str(issuance_path),
        "recovery_issuance_file_sha256": _file_sha256(issuance_path),
        "recovery_issuance": dict(merge.get("recovery_issuance", {})),
        "source_files": source_files,
        "recovery_ledger_path": issuance.get("ledger_path"),
        "recovery_ledger_sha256": _file_sha256(issuance.get("ledger_path")),
        "manifest_sha256": frozen.canonical_sha256(manifest),
        "certificate_bundle_path": bundle_source.get("path"),
        "certificate_bundle_file_sha256": _file_sha256(bundle_source.get("path")) if bundle_source.get("path") else None,
        "certificate_bundle_sha256": bundle.get("bundle_sha256"),
        "study_issuance_path": study_source.get("path"),
        "study_issuance_file_sha256": _file_sha256(study_source.get("path")) if study_source.get("path") else None,
        "bridge_source_sha256": _file_sha256(Path(__file__).resolve()),
        "frozen_analyzer_source_sha256": _file_sha256(Path(frozen.__file__).resolve()),
        "row_count": len(manifest.get("entries", [])),
        "decision_ids": [str(entry["decision_id"]) for entry in manifest.get("entries", [])],
        "manifest_entry_sha256s": [frozen.canonical_sha256(entry) for entry in manifest.get("entries", [])],
    }
    core["header_sha256"] = _canonical_without(core, "header_sha256")
    return core


def _resume_inputs(
    merge_path: Path, issuance_path: Path, checkpoint_header: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only lightweight committed JSON; do not rebuild the 600-row merge."""
    merge, issuance = _checkpoint_json(merge_path), _checkpoint_json(issuance_path)
    if checkpoint_header.get("merge_index_path") != str(merge_path) or checkpoint_header.get("recovery_issuance_path") != str(issuance_path):
        raise ValueError("checkpoint header input path binding mismatch")
    if checkpoint_header.get("merge_index_file_sha256") != _file_sha256(merge_path) or checkpoint_header.get("recovery_issuance_file_sha256") != _file_sha256(issuance_path):
        raise ValueError("checkpoint header input file changed")
    if checkpoint_header.get("recovery_ledger_path") != issuance.get("ledger_path") or checkpoint_header.get("recovery_ledger_sha256") != _file_sha256(issuance.get("ledger_path")):
        raise ValueError("checkpoint header recovery ledger changed")
    advertised_merge = merge.get("merge_sha256")
    if _is_digest(advertised_merge) and advertised_merge != recovery.canonical_sha256({key: value for key, value in merge.items() if key != "merge_sha256"}):
        raise ValueError("merged evidence commitment mismatch")
    advertised_issuance = issuance.get("issuance_sha256")
    if _is_digest(advertised_issuance) and advertised_issuance != recovery.canonical_sha256({key: value for key, value in issuance.items() if key != "issuance_sha256"}):
        raise ValueError("recovery issuance commitment mismatch")
    source_files = checkpoint_header.get("source_files", {})
    if not isinstance(source_files, Mapping):
        raise ValueError("checkpoint header source binding schema mismatch")
    for source in source_files.values():
        if not isinstance(source, Mapping) or _file_sha256(source.get("path")) != source.get("sha256"):
            raise ValueError("checkpoint header source file changed")
    return merge, issuance


def analyze_recovery_merge(
    merge_path: Path | str, issuance_path: Path | str, *, checkpoint_dir: Path | str | None = None,
) -> dict[str, Any]:
    merge_path, issuance_path = Path(merge_path).resolve(), Path(issuance_path).resolve()
    with _source_context():
        checkpoint_root = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        if checkpoint_root is not None:
            # Reject a checkpoint location before any expensive sealed-evidence
            # validation.  The full issuance validation below remains the
            # authority for all scientific and provenance checks.
            _validate_checkpoint_dir(checkpoint_root, _checkpoint_json(issuance_path), merge_path, issuance_path)
        existing_header = _checkpoint_json(checkpoint_root / "header.json") if checkpoint_root is not None and (checkpoint_root / "header.json").is_file() else None
        if existing_header is None:
            merge, issuance = validate_merge_index(merge_path, issuance_path)
        else:
            merge, issuance = _resume_inputs(merge_path, issuance_path, existing_header)
            structurally_validated_merge, structurally_validated_issuance = validate_merge_index(merge_path, issuance_path)
            if structurally_validated_merge != merge or structurally_validated_issuance != issuance:
                raise ValueError("checkpoint inputs differ from the deterministic v1.4-R merge validator")
        manifest = _json(issuance["sources"]["manifest"]["path"])
        bundle = _json(issuance["sources"]["certificate_bundle"]["path"])
        original_study_record = _json(issuance["sources"]["study_issuance"]["path"])
        study_dir = Path(str(original_study_record["output_dir"])).resolve()
        checkpoints = None
        if checkpoint_dir is not None:
            root = _validate_checkpoint_dir(Path(checkpoint_dir), issuance, merge_path, issuance_path)
            checkpoints = _CheckpointStore(root, _checkpoint_header(merge_path, issuance_path, merge, issuance, manifest, bundle))
        global _ACTIVE_CHECKPOINTS
        previous_checkpoints = _ACTIVE_CHECKPOINTS
        _ACTIVE_CHECKPOINTS = checkpoints
        try:
            rows, invalid = _read_merged_rows(merge, issuance_path, issuance, manifest, bundle)
            pair_errors = _paired_merge_errors(merge, manifest)
            if pair_errors:
                raise ValueError("unsealed merged v1.4 evidence: " + "; ".join(pair_errors))

            original_reader = frozen._read_rows
            original_pair_validator = frozen.paired_v14_inventory_errors
            try:
                frozen._read_rows = lambda *_args, **_kwargs: (rows, invalid)
                frozen.paired_v14_inventory_errors = lambda *_args, **_kwargs: []
                result = frozen.analyze_v14(study_dir, manifest, bundle, sealed=True)
            finally:
                frozen._read_rows = original_reader
                frozen.paired_v14_inventory_errors = original_pair_validator
            result["recovery_analysis_bridge"] = {
                "schema_version": BRIDGE_SCHEMA,
                "merge_index_path": str(merge_path), "merge_index_file_sha256": _file_sha256(merge_path),
                "merge_sha256": merge["merge_sha256"], "recovery_issuance": dict(merge["recovery_issuance"]),
                "effective_counts": dict(merge["counts"]),
                "bridge_source_sha256": _file_sha256(Path(__file__).resolve()),
                "frozen_analyzer_source_sha256": _file_sha256(Path(frozen.__file__).resolve()),
                "evidence_resolution_only": True, "preregistered_gates_changed": False,
            }
            result["analysis_commitment_sha256"] = frozen.canonical_sha256({key: value for key, value in result.items() if key != "analysis_commitment_sha256"})
            return result
        finally:
            _ACTIVE_CHECKPOINTS = previous_checkpoints


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_create_payload(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-index", type=Path, required=True)
    parser.add_argument("--recovery-issuance", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output.suffix.lower() != ".json" or output.exists():
        raise FileExistsError("analysis output must be a new .json file")
    issuance = recovery.validate_recovery_issuance(args.recovery_issuance)
    protected = {
        Path(source["path"]).resolve() for source in issuance["sources"].values()
        if isinstance(source, Mapping) and isinstance(source.get("path"), str)
    } | {args.merge_index.resolve(), args.recovery_issuance.resolve(), Path(issuance["ledger_path"]).resolve()}
    protected_roots = [Path(issuance["recovery_root"]).resolve(), Path(issuance["output_dir"]).resolve()]
    if output in protected or any(output == root or root in output.parents for root in protected_roots):
        raise ValueError("analysis output must be outside immutable original/recovery evidence")
    if args.checkpoint_dir is not None:
        _validate_checkpoint_dir(args.checkpoint_dir, issuance, args.merge_index, args.recovery_issuance)
        if args.checkpoint_dir.resolve() == output:
            raise ValueError("checkpoint directory and analysis output must be different paths")
    result = analyze_recovery_merge(args.merge_index, args.recovery_issuance, checkpoint_dir=args.checkpoint_dir)
    _exclusive_json(output, result)
    print(json.dumps({
        "status": "analysis_written", "output": str(output),
        "analysis_commitment_sha256": result["analysis_commitment_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate_merge_index", "analyze_recovery_merge", "main"]
