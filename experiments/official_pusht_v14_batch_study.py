"""Versioned v1.4 batch/sidecar guards and terminal ledger.

This file deliberately has no fallback to v1.1 or v1.2 validators.  A full
run must receive a validated v1.4 certificate bundle before its first row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from experiments.official_pusht_v14_common_initial import (
    V14_ARRAY_KEYS, V14_CROSS_MODEL_ARRAY_KEYS, V14_LOCKED_MODELS, V14_PROTOCOL, V14_SCHEMA_VERSION, array_sha256,
    canonical_sha256, file_sha256, validate_v14_args, validate_v14_artifact, _dataset_segment_hashes,
    validate_v14_manifest, v14_protocol_sha256, v14_source_bundle_sha256,
    validate_v14_sidecar,
)
from experiments.official_pusht_v14_bootstrap import (
    V14_CANONICAL_LAUNCHER_PATH, V14_CANONICAL_PROJECT_ROOT, V14_SOURCE_MODULES,
    _consume_v14_launcher_attestation, v14_execution_preflight,
)


LEDGER_NAME = "attempts.jsonl"
ROW_CAPABILITY_DIR = ".row_capabilities"
V14_ROW_CAPABILITY_SCHEMA_VERSION = "PWA-PushT-row-capability-v1.4-common-initial"
V14_STUDY_ISSUANCE_RELATIVE_PATH = "experiments/official_pusht_v14_study_issuance.json"
V14_STUDY_LEDGER_RELATIVE_PATH = "experiments/official_pusht_v14_study_ledger.jsonl"
V14_STUDY_ISSUANCE_SCHEMA_VERSION = "PWA-PushT-study-issuance-v1.4-common-initial"
V14_STUDY_ISSUANCE_PATH = (V14_CANONICAL_PROJECT_ROOT / V14_STUDY_ISSUANCE_RELATIVE_PATH).resolve()
V14_STUDY_LEDGER_PATH = (V14_CANONICAL_PROJECT_ROOT / V14_STUDY_LEDGER_RELATIVE_PATH).resolve()


def _module_origin_error(name: str, module: Any) -> str | None:
    value = getattr(module, "__file__", None)
    roots = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    if not isinstance(value, str):
        return f"v1.4 {name} module has no filesystem origin"
    origin = Path(value).resolve()
    if not any(origin == root or root in origin.parents for root in roots):
        return f"v1.4 {name} module origin is outside the isolated runtime"
    return None


def _full_run_environment_errors(*, torch_module: Any | None = None, official_root: Path | str | None = None) -> list[str]:
    """Shared non-importing boundary check, with an optional post-import check."""
    errors = v14_execution_preflight(
        official_root or V14_CANONICAL_PROJECT_ROOT,
        project_root=V14_CANONICAL_PROJECT_ROOT,
        source_modules=V14_SOURCE_MODULES,
    )
    if torch_module is not None:
        torch_error = _module_origin_error("torch", torch_module)
        if torch_error:
            errors.append(torch_error)
    return sorted(set(errors))


def _study_issuance_record(bindings: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(bindings)
    normalized.setdefault("canonical_project_root", str(V14_CANONICAL_PROJECT_ROOT))
    normalized.setdefault("launcher_path", str(V14_CANONICAL_LAUNCHER_PATH))
    normalized.setdefault("launcher_sha256", file_sha256(V14_CANONICAL_LAUNCHER_PATH))
    normalized.setdefault("launcher_attestation", None)
    required = {
        "manifest_sha256", "certificate_bundle_sha256", "certificate_issuance_sha256",
        "protocol_sha256", "source_bundle_sha256", "output_dir", "models",
        "official_root", "dataset_root", "dataset_split", "study_ledger_path", "dinov2", "runtime", "hardware",
        "canonical_project_root", "launcher_path", "launcher_sha256", "launcher_attestation",
    }
    if set(normalized) != required:
        raise ValueError("v1.4 study issuance bindings are incomplete")
    base = {
        "schema_version": V14_STUDY_ISSUANCE_SCHEMA_VERSION, "status": "consumed",
        "one_shot": True, "issuance_kind": "study", "run_id": uuid.uuid4().hex,
        **normalized,
    }
    base["runtime_sha256"] = canonical_sha256(base["runtime"])
    base["hardware_sha256"] = canonical_sha256(base["hardware"])
    base["issuance_sha256"] = canonical_sha256(base)
    return base


def validate_v14_study_issuance(issuance: Mapping[str, Any], *, expected_bindings: Mapping[str, Any] | None = None) -> list[str]:
    required = {
        "schema_version", "status", "one_shot", "issuance_kind", "manifest_sha256",
        "certificate_bundle_sha256", "certificate_issuance_sha256", "protocol_sha256",
        "source_bundle_sha256", "output_dir", "models", "official_root", "dataset_root",
        "study_ledger_path",
        "dataset_split", "dinov2", "runtime", "hardware", "runtime_sha256",
        "hardware_sha256", "canonical_project_root", "launcher_path", "launcher_sha256",
        "launcher_attestation", "run_id", "issuance_sha256",
    }
    if not isinstance(issuance, Mapping) or set(issuance) != required:
        return ["v1.4 study issuance has an unexpected schema"]
    errors: list[str] = []
    if issuance.get("schema_version") != V14_STUDY_ISSUANCE_SCHEMA_VERSION or issuance.get("status") != "consumed" or issuance.get("one_shot") is not True or issuance.get("issuance_kind") != "study":
        errors.append("v1.4 study issuance is not a consumed one-shot record")
    if not isinstance(issuance.get("run_id"), str) or len(str(issuance.get("run_id"))) != 32:
        errors.append("v1.4 study issuance lacks its immutable run_id")
    if issuance.get("protocol_sha256") != v14_protocol_sha256() or issuance.get("source_bundle_sha256") != v14_source_bundle_sha256():
        errors.append("v1.4 study issuance protocol/source binding mismatch")
    if issuance.get("canonical_project_root") != str(V14_CANONICAL_PROJECT_ROOT):
        errors.append("v1.4 study issuance canonical project-root binding mismatch")
    if issuance.get("launcher_path") != str(V14_CANONICAL_LAUNCHER_PATH) or issuance.get("launcher_sha256") != file_sha256(V14_CANONICAL_LAUNCHER_PATH):
        errors.append("v1.4 study issuance launcher binding mismatch")
    if not isinstance(issuance.get("runtime"), Mapping) or issuance.get("runtime_sha256") != canonical_sha256(issuance.get("runtime")):
        errors.append("v1.4 study issuance runtime binding mismatch")
    if not isinstance(issuance.get("hardware"), Mapping) or issuance.get("hardware_sha256") != canonical_sha256(issuance.get("hardware")):
        errors.append("v1.4 study issuance hardware binding mismatch")
    if issuance.get("issuance_sha256") != canonical_sha256({key: value for key, value in issuance.items() if key != "issuance_sha256"}):
        errors.append("v1.4 study issuance hash mismatch")
    for key, value in (expected_bindings or {}).items():
        if issuance.get(key) != value:
            errors.append(f"v1.4 study issuance {key} binding mismatch")
    return sorted(set(errors))


def _consume_v14_study_issuance(bindings: Mapping[str, Any], *, issuance_path: Path | str | None = None) -> tuple[Path, dict[str, Any]]:
    path = V14_STUDY_ISSUANCE_PATH if issuance_path is None else Path(issuance_path).resolve()
    if issuance_path is None and path != V14_STUDY_ISSUANCE_PATH:
        raise ValueError("v1.4 study issuance path is not canonical")
    if issuance_path is None and not bindings.get("launcher_attestation"):
        raise RuntimeError("v1.4 study issuance requires a consumed canonical launcher attestation")
    record = _study_issuance_record(bindings)
    errors = validate_v14_study_issuance(record, expected_bindings=bindings)
    if errors:
        raise ValueError("invalid v1.4 study issuance: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"v1.4 study issuance already consumed: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path, record


def _study_issuance_reference(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path)), "issuance_sha256": record.get("issuance_sha256")}


def _row_capability_path(output_dir: Path, ordinal: int, decision_id: str) -> Path:
    return (Path(output_dir).resolve() / ROW_CAPABILITY_DIR / f"{int(ordinal):03d}_{decision_id}.json").resolve()


def _consumed_row_capability_path(path: Path) -> Path:
    path = Path(path).resolve()
    return path.with_name(path.stem + ".consumed.json")


def _create_row_capability(
    *, out_dir: Path, ledger: Path, header: Mapping[str, Any], started_event_sha256: str,
    study_reference: Mapping[str, Any], manifest_sha256: str, entry: Mapping[str, Any],
    ordinal: int, artifact_path: Path, arrays_path: Path, official_root: Path,
    dataset_root: Path, model_spec: Mapping[str, str],
) -> Path:
    path = _row_capability_path(out_dir, ordinal, str(entry["decision_id"]))
    record = {
        "schema_version": V14_ROW_CAPABILITY_SCHEMA_VERSION, "status": "authorized", "one_shot": True,
        "study_issuance": dict(study_reference), "run_id": header["run_id"],
        "run_commitment_sha256": header["event_sha256"], "started_event_sha256": started_event_sha256,
        "ledger_path": str(Path(ledger).resolve()), "decision_id": str(entry["decision_id"]),
        "trajectory_id": str(entry["trajectory_id"]), "ordinal": int(ordinal),
        "model": str(entry["model"]), "episode": int(entry["episode"]),
        "offset": int(entry["offset"]), "seed": int(entry["seed"]),
        "manifest_sha256": manifest_sha256, "official_root": str(Path(official_root).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()), "dataset_split": "val",
        "checkpoint": str(Path(model_spec["checkpoint"]).resolve()),
        "config": str(Path(model_spec["config"]).resolve()),
        "artifact_path": str(Path(artifact_path).resolve()), "arrays_path": str(Path(arrays_path).resolve()),
        "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
    }
    record["capability_sha256"] = canonical_sha256(record)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _row_capability_state(output_dir: Path, ordinal: int, decision_id: str) -> dict[str, Any]:
    authorized = _row_capability_path(output_dir, ordinal, decision_id)
    consumed = _consumed_row_capability_path(authorized)
    path, status = (consumed, "consumed") if consumed.is_file() else (authorized, "authorized") if authorized.is_file() else (authorized, "missing")
    commitment = None
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            commitment = record.get("capability_sha256") if isinstance(record, Mapping) else None
        except (OSError, ValueError, TypeError):
            commitment = None
    return {"status": status, "path": str(path.resolve()), "sha256": file_sha256(path), "capability_sha256": commitment}


class V14ArtifactValidationError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _ledger_payload_hash(event: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in event.items() if key != "event_sha256"})


def _new_ledger(ledger: Path, commitments: Mapping[str, Any], *, run_id: str | None = None) -> tuple[dict[str, Any], str]:
    header = {"status": "run_header", "run_id": run_id or uuid.uuid4().hex, "previous_event_sha256": None, **dict(commitments)}
    header["event_sha256"] = _ledger_payload_hash(header)
    payload = (json.dumps(header, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(Path(ledger), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return header, str(header["event_sha256"])


def read_v14_ledger(ledger: Path, *, expected_commitments: Mapping[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, list[Mapping[str, Any]]], list[str], str | None]:
    try:
        records = [json.loads(line) for line in Path(ledger).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as exc:
        return None, {}, [f"invalid v1.4 append-only ledger: {exc}"], None
    if not records or records[0].get("status") != "run_header":
        return None, {}, ["v1.4 ledger lacks its exclusive run header"], None
    header, errors, previous = records[0], [], None
    run_id = header.get("run_id")
    for index, record in enumerate(records):
        if record.get("previous_event_sha256") != previous:
            errors.append(f"v1.4 ledger event {index} breaks predecessor chain")
        if record.get("event_sha256") != _ledger_payload_hash(record):
            errors.append(f"v1.4 ledger event {index} hash mismatch")
        if index and record.get("run_id") != run_id:
            errors.append(f"v1.4 ledger event {index} run_id mismatch")
        if index and record.get("run_commitment_sha256") != header.get("event_sha256"):
            errors.append(f"v1.4 ledger event {index} run commitment mismatch")
        previous = record.get("event_sha256")
    for key, value in (expected_commitments or {}).items():
        if header.get(key) != value:
            errors.append(f"v1.4 ledger header {key} commitment mismatch")
    histories: dict[str, list[Mapping[str, Any]]] = {}
    for record in records[1:]:
        decision_id = record.get("decision_id")
        if not isinstance(decision_id, str):
            errors.append("v1.4 row ledger event lacks decision_id")
            continue
        histories.setdefault(decision_id, []).append(record)
    return header, histories, sorted(set(errors)), previous


def _append_ledger_event(ledger: Path, payload: Mapping[str, Any], *, header: Mapping[str, Any], previous: str) -> str:
    event = {**dict(payload), "run_id": header["run_id"], "run_commitment_sha256": header["event_sha256"], "previous_event_sha256": previous}
    event["event_sha256"] = _ledger_payload_hash(event)
    with Path(ledger).open("ab") as handle:
        handle.write((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return str(event["event_sha256"])


def completed_receipt_errors(artifact_path: Path, arrays_path: Path, receipt: Mapping[str, Any], *, manifest_sha256: str, certificate_bundle_sha256: str | None, certificate_sha256: str | None, certificate_issuance_sha256: str | None = None, study_issuance: Mapping[str, Any] | None = None, row_commitments: Mapping[str, Any] | None = None) -> list[str]:
    expected = {
        "artifact_exists": Path(artifact_path).is_file(), "arrays_exists": Path(arrays_path).is_file(),
        "artifact_sha256": file_sha256(Path(artifact_path)), "arrays_sha256": file_sha256(Path(arrays_path)),
        "manifest_sha256": manifest_sha256, "certificate_bundle_sha256": certificate_bundle_sha256,
        "certificate_sha256": certificate_sha256, "protocol_sha256": v14_protocol_sha256(),
        "source_bundle_sha256": v14_source_bundle_sha256(), "certificate_issuance_sha256": certificate_issuance_sha256,
        "study_issuance": dict(study_issuance) if study_issuance is not None else None,
        **dict(row_commitments or {}),
    }
    errors = []
    if expected["artifact_exists"] is not True or expected["arrays_exists"] is not True:
        errors.append("completed receipt evidence is missing")
    errors.extend(
        f"completed receipt {key} changed or is missing"
        for key, value in expected.items()
        if key not in receipt or receipt.get(key) != value
    )
    return errors


def failed_receipt_errors(artifact_path: Path, arrays_path: Path, receipt: Mapping[str, Any], *, manifest_sha256: str, certificate_bundle_sha256: str | None, certificate_sha256: str | None, certificate_issuance_sha256: str | None = None, study_issuance: Mapping[str, Any] | None = None, row_commitments: Mapping[str, Any] | None = None) -> list[str]:
    expected = {
        "status": "failed", "artifact_exists": Path(artifact_path).is_file(), "arrays_exists": Path(arrays_path).is_file(),
        "artifact_sha256": file_sha256(Path(artifact_path)), "arrays_sha256": file_sha256(Path(arrays_path)),
        "manifest_sha256": manifest_sha256, "certificate_bundle_sha256": certificate_bundle_sha256,
        "certificate_sha256": certificate_sha256, "certificate_issuance_sha256": certificate_issuance_sha256,
        "study_issuance": dict(study_issuance) if study_issuance is not None else None,
        "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
        **dict(row_commitments or {}),
    }
    return [
        f"failed receipt {key} changed or is missing"
        for key, value in expected.items()
        if key not in receipt or receipt.get(key) != value
    ]


def paired_v14_inventory_errors(out_dir: Path, entries: list[Mapping[str, Any]]) -> list[str]:
    """Require exact physical/raw/J/G evidence for each paired model cell."""
    errors: list[str] = []
    groups: dict[tuple[Any, Any, Any], dict[str, tuple[Mapping[str, Any], Path]]] = {}
    for entry in entries:
        path = Path(out_dir) / f"{entry['decision_id']}.json"
        if not path.is_file():
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        groups.setdefault((entry.get("trajectory_id"), entry.get("offset"), entry.get("seed")), {})[str(entry.get("model"))] = (artifact, path)
    for cell, models in groups.items():
        if set(models) != set(V14_LOCKED_MODELS):
            continue
        loaded: dict[str, dict[str, np.ndarray]] = {}
        for model in V14_LOCKED_MODELS:
            artifact, _ = models[model]
            sidecar = artifact.get("arrays_sidecar")
            if not isinstance(sidecar, str):
                errors.append(f"{cell}: {model} lacks v1.4 sidecar")
                continue
            try:
                with np.load(sidecar, allow_pickle=False) as archive:
                    loaded[model] = {key: np.asarray(archive[key]) for key in V14_CROSS_MODEL_ARRAY_KEYS}
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{cell}: unreadable v1.4 paired sidecar ({exc})")
        if len(loaded) == 2:
            left, right = (loaded[model] for model in V14_LOCKED_MODELS)
            for key in left:
                if not np.array_equal(left[key], right[key]):
                    errors.append(f"{cell}: paired v1.4 {key} differs between JEPA and DINO")
    return sorted(set(errors))


def _bundle_errors(bundle: Mapping[str, Any]) -> list[str]:
    from experiments.official_pusht_v14_certificate_supervisor import validate_v14_bundle
    return validate_v14_bundle(bundle)


def _validated_model_specs(bundle: Mapping[str, Any], runner_kwargs: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    supplied = runner_kwargs.get("model_specs")
    if not isinstance(supplied, Mapping) or set(supplied) != set(V14_LOCKED_MODELS):
        raise ValueError("v1.4 full study requires exactly JEPA and DINO model_specs")
    certificates = bundle.get("certificates")
    if not isinstance(certificates, Mapping):
        raise ValueError("v1.4 certificate bundle lacks model certificates")
    resolved: dict[str, dict[str, str]] = {}
    for model in V14_LOCKED_MODELS:
        spec, certificate = supplied.get(model), certificates.get(model)
        if not isinstance(spec, Mapping) or not isinstance(certificate, Mapping):
            raise ValueError(f"v1.4 {model} model specification is incomplete")
        checkpoint, config = Path(str(spec.get("checkpoint"))).resolve(), Path(str(spec.get("config"))).resolve()
        if not checkpoint.is_file() or not config.is_file():
            raise ValueError(f"v1.4 {model} checkpoint/config does not exist")
        if file_sha256(checkpoint) != certificate.get("checkpoint_sha256") or file_sha256(config) != certificate.get("config_sha256"):
            raise ValueError(f"v1.4 {model} checkpoint/config differs from its certificate")
        resolved[model] = {"checkpoint": str(checkpoint), "config": str(config)}
    return resolved


def _certified_menu_hashes(bundle: Mapping[str, Any]) -> dict[int, str]:
    by_seed: dict[int, set[str]] = {seed: set() for seed in (1, 2, 3)}
    certificates = bundle.get("certificates")
    if not isinstance(certificates, Mapping):
        raise ValueError("v1.4 certificate bundle lacks certified proposal menus")
    for model in V14_LOCKED_MODELS:
        certificate = certificates.get(model)
        if not isinstance(certificate, Mapping):
            raise ValueError(f"v1.4 {model} certificate is missing")
        for record in certificate.get("contracts", []):
            if not isinstance(record, Mapping) or not isinstance(record.get("contract"), Mapping):
                continue
            signature = record.get("repeat0_signature")
            if not isinstance(signature, Mapping) or not isinstance(signature.get("array_hashes"), Mapping):
                continue
            seed = int(record["contract"].get("seed", -1))
            value = signature["array_hashes"].get("initial_actions_normalized")
            if seed in by_seed and isinstance(value, str):
                by_seed[seed].add(value)
    if any(len(values) != 1 for values in by_seed.values()):
        raise ValueError("v1.4 certified JEPA/DINO proposal-menu hashes are incomplete or disagree")
    return {seed: next(iter(values)) for seed, values in by_seed.items()}


def _study_bindings(manifest: Mapping[str, Any], manifest_hash: str, bundle: Mapping[str, Any], output_dir: Path, model_specs: Mapping[str, Mapping[str, str]], *, launcher_attestation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from experiments.official_pusht_decision_audit import _hardware_fingerprint, _runtime_fingerprint

    issuance = bundle.get("certificate_issuance")
    required = ("runtime", "hardware", "official_root", "dataset_root", "dinov2")
    if any(not isinstance(bundle.get(key), (Mapping if key in ("runtime", "hardware", "dinov2") else str)) for key in required):
        raise ValueError("v1.4 certificate bundle lacks mandatory study runtime/data bindings")
    if not isinstance(issuance, Mapping) or not isinstance(issuance.get("sha256"), str):
        raise ValueError("v1.4 certificate bundle lacks certificate issuance binding")
    current_runtime, current_hardware = _runtime_fingerprint(), _hardware_fingerprint()
    if bundle.get("runtime") != current_runtime or bundle.get("hardware") != current_hardware:
        raise ValueError("v1.4 study runtime/hardware differs from the certified execution environment")
    return {
        "manifest_sha256": manifest_hash,
        "certificate_bundle_sha256": str(bundle.get("bundle_sha256")),
        "certificate_issuance_sha256": str(issuance["sha256"]),
        "protocol_sha256": v14_protocol_sha256(),
        "source_bundle_sha256": v14_source_bundle_sha256(),
        "output_dir": str(Path(output_dir).resolve()),
        "models": {
            model: {
                "certificate_sha256": bundle.get("certificate_sha256", {}).get(model),
                "checkpoint": model_specs[model]["checkpoint"],
                "config": model_specs[model]["config"],
            }
            for model in V14_LOCKED_MODELS
        },
        "official_root": str(Path(bundle["official_root"]).resolve()),
        "dataset_root": str(Path(bundle["dataset_root"]).resolve()),
        "dataset_split": "val",
        "study_ledger_path": str(V14_STUDY_LEDGER_PATH),
        "dinov2": dict(bundle["dinov2"]),
        "runtime": dict(bundle["runtime"]),
        "hardware": dict(bundle["hardware"]),
        "canonical_project_root": str(V14_CANONICAL_PROJECT_ROOT),
        "launcher_path": str(V14_CANONICAL_LAUNCHER_PATH),
        "launcher_sha256": file_sha256(V14_CANONICAL_LAUNCHER_PATH),
        "launcher_attestation": dict(launcher_attestation) if launcher_attestation is not None else None,
    }


def _prepare_study_issuance(manifest: Mapping[str, Any], manifest_hash: str, bundle: Mapping[str, Any], output_dir: Path, model_specs: Mapping[str, Mapping[str, str]], *, ledger_exists: bool) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = V14_STUDY_ISSUANCE_PATH
    bindings = _study_bindings(manifest, manifest_hash, bundle, output_dir, model_specs)
    if path.is_file():
        if not ledger_exists:
            raise FileExistsError("v1.4 study issuance exists without its bound ledger; the one-shot study is terminal")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("v1.4 study issuance cannot be read") from exc
        if isinstance(record, Mapping):
            bindings["launcher_attestation"] = record.get("launcher_attestation")
        errors = validate_v14_study_issuance(record, expected_bindings=bindings)
        if errors:
            raise ValueError("v1.4 resumed study does not match canonical issuance: " + "; ".join(errors))
        return path, record, _study_issuance_reference(path, record)
    if ledger_exists:
        raise FileNotFoundError("v1.4 study issuance is missing for a resumed run")
    if path.exists():
        raise FileExistsError("v1.4 study issuance was already consumed; refusing a new full run")
    bindings["launcher_attestation"] = _consume_v14_launcher_attestation("experiments.official_pusht_v14_batch_study")
    path, record = _consume_v14_study_issuance(bindings)
    return path, record, _study_issuance_reference(path, record)


def _official_v14_dispatcher(*, model_specs: Mapping[str, Mapping[str, str]], model_name: str, **kwargs: Any) -> Mapping[str, Any]:
    """The only dispatcher permitted to execute a canonical 600-row study."""
    from experiments.official_pusht_v14_decision_audit import run_one_decision
    if model_name not in V14_LOCKED_MODELS or set(model_specs) != set(V14_LOCKED_MODELS):
        raise ValueError("v1.4 official dispatcher received an unsupported model map")
    spec = model_specs[model_name]
    return run_one_decision(
        **kwargs, model_name=model_name,
        checkpoint=Path(spec["checkpoint"]), config=Path(spec["config"]),
    )


def run_study(
    manifest: Mapping[str, Any], out_dir: Path, runner: Callable[..., Mapping[str, Any]],
    runner_kwargs: Mapping[str, Any] | None = None, *, planning_repeats: int = 0,
    certificate_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run rows once; failed and interrupted rows are terminal and never retried."""
    full_run = len(manifest.get("entries", [])) == 600
    errors = validate_v14_manifest(manifest, require_full=full_run)
    if errors:
        raise ValueError("invalid v1.4 manifest: " + "; ".join(errors))
    validate_v14_args(planning_repeats=planning_repeats)
    out_dir = Path(out_dir).resolve()
    ledger = V14_STUDY_LEDGER_PATH if full_run else out_dir / LEDGER_NAME
    manifest_hash = canonical_sha256(manifest)
    kwargs = dict(runner_kwargs or {})
    study_path = study_record = study_reference = None
    model_specs: dict[str, dict[str, str]] | None = None
    certified_menu_hashes: dict[int, str] = {}
    if full_run:
        from experiments.official_pusht_v14_certificate_supervisor import V14_CERTIFICATE_ISSUANCE_PATH

        protected = (V14_CERTIFICATE_ISSUANCE_PATH, V14_STUDY_ISSUANCE_PATH, V14_STUDY_LEDGER_PATH)
        if any(out_dir == path or out_dir in path.parents or path in out_dir.parents for path in protected):
            raise ValueError("v1.4 study output overlaps a canonical issuance or study-ledger path")
        if not ledger.is_file():
            if out_dir.exists() and not out_dir.is_dir():
                raise NotADirectoryError(f"v1.4 study output is not a directory: {out_dir}")
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=False)
            if any(out_dir.iterdir()):
                raise FileExistsError("v1.4 bound output directory is nonempty before study issuance")
        elif not out_dir.is_dir():
            raise NotADirectoryError("v1.4 resumed study output directory is missing or not a directory")
        if not isinstance(certificate_bundle, Mapping):
            raise ValueError("full v1.4 runs require a validated certificate bundle")
        bundle_errors = _bundle_errors(certificate_bundle)
        if bundle_errors:
            raise ValueError("invalid v1.4 certificate bundle: " + "; ".join(bundle_errors))
        if manifest.get("dataset_root") != certificate_bundle.get("dataset_root") or manifest.get("dataset_split") != "val":
            raise ValueError("v1.4 manifest and certificate bundle bind different dataset roots/splits")
        if runner is not _official_v14_dispatcher:
            raise ValueError("full v1.4 runs require the source-owned official dispatcher")
        model_specs = _validated_model_specs(certificate_bundle, kwargs)
        certified_menu_hashes = _certified_menu_hashes(certificate_bundle)
        if not isinstance(certificate_bundle.get("official_root"), str) or Path(str(kwargs.get("official_root", ""))).resolve() != Path(str(certificate_bundle["official_root"])).resolve():
            raise ValueError("v1.4 runner official_root differs from the certificate bundle")
        if not isinstance(certificate_bundle.get("dataset_root"), str) or Path(str(kwargs.get("dataset_root", ""))).resolve() != Path(str(certificate_bundle["dataset_root"])).resolve():
            raise ValueError("v1.4 runner dataset_root differs from the certificate bundle")
        for forbidden in ("checkpoint", "config", "model_name", "row_capability_path"):
            if forbidden in kwargs:
                raise ValueError(f"v1.4 full study runner_kwargs must not supply {forbidden}; the official router owns it")
        kwargs["model_specs"] = model_specs
        official_import_root = Path(str(certificate_bundle["official_root"])).resolve()
        if str(official_import_root) not in sys.path:
            sys.path.insert(0, str(official_import_root))
        environment_errors = _full_run_environment_errors(official_root=official_import_root)
        if environment_errors:
            raise RuntimeError("unsafe v1.4 full-run Python environment: " + "; ".join(environment_errors))
        import torch
        environment_errors = _full_run_environment_errors(torch_module=torch, official_root=official_import_root)
        if environment_errors:
            raise RuntimeError("unsafe v1.4 full-run Python environment: " + "; ".join(environment_errors))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        study_path, study_record, study_reference = _prepare_study_issuance(
            manifest, manifest_hash, certificate_bundle, out_dir, model_specs,
            ledger_exists=ledger.is_file(),
        )
        kwargs["parent_launcher_attestation"] = dict(study_record["launcher_attestation"])
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_hash = str(certificate_bundle.get("bundle_sha256")) if isinstance(certificate_bundle, Mapping) else None
    cert_hashes = dict(certificate_bundle.get("certificate_sha256", {})) if isinstance(certificate_bundle, Mapping) else {}
    certificate_bindings: dict[str, dict[str, Any]] = {}
    certificate_issuance = certificate_bundle.get("certificate_issuance") if isinstance(certificate_bundle, Mapping) else None
    certificate_issuance_sha256 = certificate_issuance.get("sha256") if isinstance(certificate_issuance, Mapping) else None
    if isinstance(certificate_bundle, Mapping):
        certificates = certificate_bundle.get("certificates", {})
        if isinstance(certificates, Mapping):
            dataset_bindings = certificate_bundle.get("dataset_bindings", {})
            for model in V14_LOCKED_MODELS:
                certificate = certificates.get(model, {})
                if isinstance(certificate, Mapping):
                    certificate_bindings[model] = {
                        "model": model,
                        "protocol_sha256": v14_protocol_sha256(),
                        "source_bundle_sha256": v14_source_bundle_sha256(),
                        "manifest_sha256": manifest_hash,
                        "checkpoint_sha256": certificate.get("checkpoint_sha256"),
                        "config_sha256": certificate.get("config_sha256"),
                        "official_git_commit": certificate_bundle.get("official_git_commit"),
                        "official_root": certificate_bundle.get("official_root"),
                        "dataset_root": certificate_bundle.get("dataset_root"),
                        "dataset_split": dataset_bindings.get("dataset_split") if isinstance(dataset_bindings, Mapping) else None,
                        "dataset_archive_sha256": dataset_bindings.get("dataset_archive_sha256") if isinstance(dataset_bindings, Mapping) else None,
                        "seq_lengths_sha256": dataset_bindings.get("seq_lengths_sha256") if isinstance(dataset_bindings, Mapping) else None,
                        "dataset_states_sha256": dataset_bindings.get("dataset_states_sha256") if isinstance(dataset_bindings, Mapping) else None,
                        "environment_seed": 1,
                        "runtime": certificate_bundle.get("runtime"),
                        "hardware": certificate_bundle.get("hardware"),
                        "runtime_sha256": certificate_bundle.get("runtime_sha256"),
                        "hardware_sha256": certificate_bundle.get("hardware_sha256"),
                        "dinov2": certificate_bundle.get("dinov2"),
                    }
    def expected_bindings_for(entry: Mapping[str, Any]) -> dict[str, Any] | None:
        base = certificate_bindings.get(str(entry.get("model")))
        if base is None:
            return None
        expected = {**base, "menu_seed": int(entry["seed"])}
        expected.update({key: value for key, value in _dataset_segment_hashes({**expected, "episode": int(entry["episode"]), "offset": int(entry["offset"]) }).items() if value is not None})
        return expected
    def row_commitments_for(entry: Mapping[str, Any], ordinal: int, artifact_path: Path, arrays_path: Path, *, terminal: bool = False) -> dict[str, Any]:
        row = {
            "ordinal": int(ordinal), "model": str(entry["model"]),
            "episode": int(entry["episode"]), "offset": int(entry["offset"]),
            "menu_seed": int(entry["seed"]), "environment_seed": 1,
            "artifact_path": str(artifact_path.resolve()), "arrays_path": str(arrays_path.resolve()),
            "row_bindings": expected_bindings_for(entry),
            "certified_menu_sha256": certified_menu_hashes.get(int(entry["seed"])) if full_run else None,
        }
        if full_run:
            row["row_capability_path"] = str(_row_capability_path(out_dir, ordinal, str(entry["decision_id"])))
            if terminal:
                row["row_capability"] = _row_capability_state(out_dir, ordinal, str(entry["decision_id"]))
        return row
    commitments = {"schema_version": V14_SCHEMA_VERSION, "manifest_sha256": manifest_hash, "certificate_bundle_sha256": bundle_hash, "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(), "study_issuance": study_reference}
    if ledger.is_file():
        header, histories, ledger_errors, previous = read_v14_ledger(ledger, expected_commitments=commitments)
        if ledger_errors or header is None or previous is None:
            raise ValueError("invalid v1.4 append-only ledger: " + "; ".join(ledger_errors))
        if full_run and study_record is not None and header.get("run_id") != study_record.get("run_id"):
            raise ValueError("v1.4 resumed ledger run_id differs from the study issuance")
        # An interrupted start is terminal.  Reconcile it before reading any
        # completed artifact, so a rerun cannot silently retry that row.
        for ordinal, entry in enumerate(manifest["entries"]):
            decision_id = str(entry["decision_id"])
            history = histories.get(decision_id, [])
            if [item.get("status") for item in history] != ["started"]:
                continue
            artifact_path = out_dir / f"{decision_id}.json"
            arrays_path = out_dir / f"{decision_id}.npz"
            row_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path, terminal=True)
            interrupted = {
                "decision_id": decision_id, "status": "failed",
                "failure_kind": "supervisor_interrupted",
                "error": "automatic outcome-blind finalization of orphaned started attempt",
                "artifact_exists": artifact_path.is_file(), "arrays_exists": arrays_path.is_file(),
                "artifact_sha256": file_sha256(artifact_path), "arrays_sha256": file_sha256(arrays_path),
                "manifest_sha256": manifest_hash, "certificate_bundle_sha256": bundle_hash,
                "certificate_sha256": cert_hashes.get(str(entry.get("model"))),
                "certificate_issuance_sha256": certificate_issuance_sha256,
                "study_issuance": study_reference,
                "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
                **row_commitments,
            }
            previous = _append_ledger_event(ledger, interrupted, header=header, previous=previous)
            histories[decision_id] = [*history, interrupted]
    else:
        if full_run and study_record is None:
            raise ValueError("v1.4 full run lacks its study issuance record")
        prepositioned = [path.name for entry in manifest["entries"] for path in (out_dir / f"{entry['decision_id']}.json", out_dir / f"{entry['decision_id']}.npz") if path.exists()]
        if full_run and out_dir.exists():
            unexpected = [path.name for path in out_dir.iterdir() if path.name not in prepositioned]
            if unexpected:
                raise FileExistsError("v1.4 bound output directory is nonempty before its run header: " + ", ".join(unexpected[:5]))
        if prepositioned:
            raise FileExistsError("v1.4 evidence exists before exclusive run header: " + ", ".join(prepositioned[:5]))
        header, previous = _new_ledger(
            ledger, commitments,
            run_id=str(study_record["run_id"]) if full_run and study_record is not None else None,
        )
        histories = {}
    completed = skipped = failed = 0
    if isinstance(certificate_bundle, Mapping) and isinstance(certificate_bundle.get("dinov2"), Mapping):
        dino = certificate_bundle["dinov2"]
        kwargs.setdefault("dinov2_source_root", Path(str(dino.get("source_root"))))
        kwargs.setdefault("dinov2_source_archive", Path(str(dino.get("source_archive"))))
        kwargs.setdefault("dinov2_weights", Path(str(dino.get("weights"))))
        kwargs.setdefault("torch_home", Path(str(dino.get("torch_home"))))
    for ordinal, entry in enumerate(manifest["entries"]):
        decision_id = str(entry["decision_id"])
        artifact_path, arrays_path = out_dir / f"{decision_id}.json", out_dir / f"{decision_id}.npz"
        row_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path)
        history = histories.get(decision_id, [])
        statuses = [item.get("status") for item in history]
        if statuses:
            if statuses == ["started", "completed"]:
                row_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path, terminal=True)
                receipt_errors = completed_receipt_errors(artifact_path, arrays_path, history[-1], manifest_sha256=manifest_hash, certificate_bundle_sha256=bundle_hash, certificate_sha256=cert_hashes.get(str(entry.get("model"))), certificate_issuance_sha256=certificate_issuance_sha256, study_issuance=study_reference, row_commitments=row_commitments)
                if receipt_errors:
                    raise V14ArtifactValidationError(receipt_errors)
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                validation = validate_v14_artifact(artifact, entry, arrays_path=arrays_path, expected_bindings=expected_bindings_for(entry), expected_certificate_issuance=certificate_issuance if isinstance(certificate_issuance, Mapping) else None, expected_certificate_sha256=cert_hashes.get(str(entry.get("model"))), expected_study_issuance=study_reference, expected_row_capability=row_commitments.get("row_capability"), expected_menu_sha256=certified_menu_hashes.get(int(entry["seed"])) if full_run else None)
                if validation:
                    raise V14ArtifactValidationError(validation)
                skipped += 1
                continue
            if statuses == ["started", "failed"]:
                row_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path, terminal=True)
                receipt_errors = failed_receipt_errors(artifact_path, arrays_path, history[-1], manifest_sha256=manifest_hash, certificate_bundle_sha256=bundle_hash, certificate_sha256=cert_hashes.get(str(entry.get("model"))), certificate_issuance_sha256=certificate_issuance_sha256, study_issuance=study_reference, row_commitments=row_commitments)
                if receipt_errors:
                    raise V14ArtifactValidationError(receipt_errors)
                skipped += 1
                continue
            # Any started, failed, or multiply-recorded row is terminal.
            skipped += 1
            continue
        if artifact_path.exists() or arrays_path.exists():
            raise FileExistsError(f"v1.4 evidence exists without a ledger receipt: {decision_id}")
        previous = _append_ledger_event(ledger, {"decision_id": decision_id, "status": "started", **row_commitments}, header=header, previous=previous)
        try:
            call_kwargs = dict(kwargs)
            if full_run:
                call_kwargs["row_capability_path"] = _create_row_capability(
                    out_dir=out_dir, ledger=ledger, header=header, started_event_sha256=previous,
                    study_reference=study_reference, manifest_sha256=manifest_hash, entry=entry,
                    ordinal=ordinal, artifact_path=artifact_path, arrays_path=arrays_path,
                    official_root=Path(str(certificate_bundle["official_root"])),
                    dataset_root=Path(str(certificate_bundle["dataset_root"])),
                    model_spec=model_specs[str(entry["model"])],
                )
            result = dict(runner(**call_kwargs, episode=int(entry["episode"]), offset=int(entry["offset"]), seed=int(entry["seed"]), arrays_out=arrays_path, repeat_count=1, planning_repeats=planning_repeats, model_name=entry["model"], decision_id=decision_id, trajectory_id=entry["trajectory_id"], inventory_mode="common_initial_only", manifest_sha256=manifest_hash))
            if cert_hashes:
                result["reproducibility_certificate"] = {"status": "passed", "model": entry["model"], "sha256": cert_hashes.get(str(entry["model"]))}
            if isinstance(certificate_issuance, Mapping):
                result["certificate_issuance"] = dict(certificate_issuance)
            if study_reference is not None:
                result["study_issuance"] = dict(study_reference)
            terminal_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path, terminal=True)
            validation = validate_v14_artifact(result, entry, arrays_path=arrays_path, expected_bindings=expected_bindings_for(entry), expected_certificate_issuance=certificate_issuance if isinstance(certificate_issuance, Mapping) else None, expected_certificate_sha256=cert_hashes.get(str(entry.get("model"))), expected_study_issuance=study_reference, expected_row_capability=terminal_commitments.get("row_capability"), expected_menu_sha256=certified_menu_hashes.get(int(entry["seed"])) if full_run else None)
            if validation:
                raise V14ArtifactValidationError(validation)
            artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            previous = _append_ledger_event(ledger, {"decision_id": decision_id, "status": "completed", "artifact_exists": True, "arrays_exists": True, "artifact_sha256": file_sha256(artifact_path), "arrays_sha256": file_sha256(arrays_path), "manifest_sha256": manifest_hash, "certificate_bundle_sha256": bundle_hash, "certificate_sha256": cert_hashes.get(str(entry.get("model"))), "certificate_issuance_sha256": certificate_issuance_sha256, "study_issuance": study_reference, "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(), **terminal_commitments}, header=header, previous=previous)
            completed += 1
        except Exception as exc:
            terminal_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path, terminal=True)
            previous = _append_ledger_event(ledger, {"decision_id": decision_id, "status": "failed", "failure_kind": "scientific_integrity", "error": str(exc), "artifact_exists": artifact_path.is_file(), "arrays_exists": arrays_path.is_file(), "artifact_sha256": file_sha256(artifact_path), "arrays_sha256": file_sha256(arrays_path), "manifest_sha256": manifest_hash, "certificate_bundle_sha256": bundle_hash, "certificate_sha256": cert_hashes.get(str(entry.get("model"))), "certificate_issuance_sha256": certificate_issuance_sha256, "study_issuance": study_reference, "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(), **terminal_commitments}, header=header, previous=previous)
            failed += 1
    paired = paired_v14_inventory_errors(out_dir, list(manifest["entries"]))
    if paired:
        raise V14ArtifactValidationError(paired)
    _header, final_histories, final_errors, _previous = read_v14_ledger(
        ledger, expected_commitments=commitments,
    )
    if final_errors:
        raise ValueError("invalid v1.4 append-only ledger after execution: " + "; ".join(final_errors))
    expected_ids = {str(entry["decision_id"]) for entry in manifest["entries"]}
    if set(final_histories) != expected_ids:
        raise ValueError("v1.4 ledger does not contain exactly one terminal history per manifest row")
    legal = {
        decision_id: [item.get("status") for item in history]
        for decision_id, history in final_histories.items()
    }
    illegal = {
        decision_id: statuses
        for decision_id, statuses in legal.items()
        if statuses not in (["started", "completed"], ["started", "failed"])
    }
    if illegal:
        raise ValueError("v1.4 ledger contains non-terminal or multiply-recorded rows")
    if full_run and study_record is not None and _header is not None and _header.get("run_id") != study_record.get("run_id"):
        raise ValueError("v1.4 final ledger run_id differs from the study issuance")
    final_seal_errors: list[str] = []
    for ordinal, entry in enumerate(manifest["entries"]):
        decision_id = str(entry["decision_id"])
        artifact_path, arrays_path = out_dir / f"{decision_id}.json", out_dir / f"{decision_id}.npz"
        terminal = final_histories[decision_id][-1]
        row_commitments = row_commitments_for(entry, ordinal, artifact_path, arrays_path, terminal=True)
        receipt_kwargs = {
            "manifest_sha256": manifest_hash,
            "certificate_bundle_sha256": bundle_hash,
            "certificate_sha256": cert_hashes.get(str(entry.get("model"))),
            "certificate_issuance_sha256": certificate_issuance_sha256,
            "study_issuance": study_reference,
            "row_commitments": row_commitments,
        }
        if terminal.get("status") == "completed":
            final_seal_errors.extend(completed_receipt_errors(artifact_path, arrays_path, terminal, **receipt_kwargs))
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                final_seal_errors.append(f"{decision_id}: unreadable completed artifact ({exc})")
            else:
                final_seal_errors.extend(validate_v14_artifact(
                    artifact, entry, arrays_path=arrays_path,
                    expected_bindings=expected_bindings_for(entry),
                    expected_certificate_issuance=certificate_issuance if isinstance(certificate_issuance, Mapping) else None,
                    expected_certificate_sha256=cert_hashes.get(str(entry.get("model"))),
                    expected_study_issuance=study_reference,
                    expected_row_capability=row_commitments.get("row_capability"),
                    expected_menu_sha256=certified_menu_hashes.get(int(entry["seed"])) if full_run else None,
                ))
        else:
            final_seal_errors.extend(failed_receipt_errors(artifact_path, arrays_path, terminal, **receipt_kwargs))
    if final_seal_errors:
        raise V14ArtifactValidationError(sorted(set(final_seal_errors)))
    terminal_failed_count = sum(statuses[-1] == "failed" for statuses in legal.values())
    # A properly receipted terminal failure is preregistered invalid-as-zero
    # evidence, not a broken seal. Integrity failures in the ledger/receipts
    # raise above; the analyzer separately enforces the 540/600 validity gate.
    sealed = True
    return {
        "manifest_entries": len(manifest["entries"]),
        "counts": {"completed": len(legal) - terminal_failed_count, "skipped": skipped, "failed": terminal_failed_count},
        "ledger": str(ledger), "sealed": sealed,
        "status": "sealed_with_terminal_failures" if terminal_failed_count else "sealed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or safely resume the immutable v1.4 600-row Push-T study")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--certificate-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    for model in ("jepa", "dino"):
        parser.add_argument(f"--{model}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{model}-config", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundle = json.loads(args.certificate_bundle.read_text(encoding="utf-8"))
    result = run_study(
        manifest, args.output_dir, _official_v14_dispatcher,
        runner_kwargs={
            "official_root": args.official_root, "dataset_root": args.dataset_root,
            "model_specs": {
                "official_jepa_wm": {"checkpoint": args.jepa_checkpoint, "config": args.jepa_config},
                "official_dino_wm": {"checkpoint": args.dino_checkpoint, "config": args.dino_config},
            },
        },
        planning_repeats=0, certificate_bundle=bundle,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("sealed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LEDGER_NAME", "V14_STUDY_ISSUANCE_PATH", "V14ArtifactValidationError", "validate_v14_sidecar", "validate_v14_artifact", "read_v14_ledger", "completed_receipt_errors", "failed_receipt_errors", "paired_v14_inventory_errors", "validate_v14_study_issuance", "_study_issuance_record", "_consume_v14_study_issuance", "_official_v14_dispatcher", "run_study", "main"]
