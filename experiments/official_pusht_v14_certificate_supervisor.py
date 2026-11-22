"""Fresh-process v1.4 certificate, immutable issuance, and bundle guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.official_pusht_v14_common_initial import (
    V14_BUNDLE_SCHEMA_VERSION, V14_CERTIFICATE_ISSUANCE_RELATIVE_PATH,
    V14_CERTIFICATE_ISSUANCE_SCHEMA_VERSION, V14_CERTIFICATE_SCHEMA_VERSION,
    V14_EXCLUDED_TRAJECTORY, V14_LOCKED_MODELS, V14_LOCKED_SEEDS, V14_PROTOCOL,
    V14_SOURCE_BUNDLE_VERSION, array_sha256, canonical_sha256, compare_v14_repeats,
    file_sha256, make_v14_manifest, v14_protocol_sha256, v14_source_bundle_sha256,
    validate_v14_artifact,
)
from experiments.official_pusht_v14_bootstrap import (
    V14_CANONICAL_LAUNCHER_PATH, V14_CANONICAL_PROJECT_ROOT, V14_SOURCE_MODULES,
    _consume_v14_launcher_attestation, v14_execution_preflight,
)


V14_CERTIFICATE_ISSUANCE_PATH = (V14_CANONICAL_PROJECT_ROOT / V14_CERTIFICATE_ISSUANCE_RELATIVE_PATH).resolve()
V14_CERTIFICATE_LEDGER_RELATIVE_PATH = "experiments/official_pusht_v14_certificate_ledger.jsonl"
V14_CERTIFICATE_CAPABILITY_RELATIVE_PATH = "experiments/official_pusht_v14_certificate_capabilities"
V14_CERTIFICATE_LEDGER_PATH = (V14_CANONICAL_PROJECT_ROOT / V14_CERTIFICATE_LEDGER_RELATIVE_PATH).resolve()
V14_CERTIFICATE_CAPABILITY_DIR = (V14_CANONICAL_PROJECT_ROOT / V14_CERTIFICATE_CAPABILITY_RELATIVE_PATH).resolve()
V14_CERTIFICATE_LEDGER_SCHEMA_VERSION = "PWA-PushT-certificate-attempt-ledger-v1.4-common-initial"
V14_CERTIFICATE_CAPABILITY_SCHEMA_VERSION = "PWA-PushT-certificate-attempt-capability-v1.4-common-initial"
V14_CERTIFICATE_REPEAT_COUNT = 3


def _certificate_issuance_path() -> Path:
    expected = (V14_CANONICAL_PROJECT_ROOT / V14_CERTIFICATE_ISSUANCE_RELATIVE_PATH).resolve()
    if V14_CERTIFICATE_ISSUANCE_PATH != expected:
        raise ValueError("v1.4 certificate issuance path is not canonical")
    return expected


def _canonical_certificate_issuance_path(path: Path | str | None = None) -> Path:
    return _certificate_issuance_path() if path is None else Path(path).resolve()


def _certificate_ledger_path() -> Path:
    candidate = Path(V14_CERTIFICATE_LEDGER_PATH).resolve()
    expected = (V14_CANONICAL_PROJECT_ROOT / V14_CERTIFICATE_LEDGER_RELATIVE_PATH).resolve()
    if candidate != expected:
        raise ValueError("v1.4 certificate attempt ledger path is not canonical")
    return candidate


def _certificate_capability_root() -> Path:
    candidate = Path(V14_CERTIFICATE_CAPABILITY_DIR).resolve()
    expected = (V14_CANONICAL_PROJECT_ROOT / V14_CERTIFICATE_CAPABILITY_RELATIVE_PATH).resolve()
    if candidate != expected:
        raise ValueError("v1.4 certificate capability root is not canonical")
    return candidate


def _certificate_capability_path(run_id: str, ordinal: int) -> Path:
    return (_certificate_capability_root() / str(run_id) / f"attempt_{int(ordinal):02d}.json").resolve()


def _certificate_attempt_id(model: str, contract_id: str, repeat_index: int) -> str:
    return f"{model}:{contract_id}:repeat_{int(repeat_index):02d}"


def _certificate_schedule(contracts: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"models": list(V14_LOCKED_MODELS), "contracts": [dict(item) for item in contracts], "repeats_per_contract": 3, "fresh_process": True, "comparison_anchor": "repeat_0"}


def _certificate_issuance_bindings(*, official_root: Path, dataset_root: Path, certificate_output_dir: Path, official_git_commit: str, dataset_archive_sha256: str | None, dataset_files: Mapping[str, Any], seq_lengths_sha256: str | None, dataset_states_sha256: str | None, model_specs: Mapping[str, Mapping[str, Any]], contracts: list[Mapping[str, Any]], dinov2: Mapping[str, Any], launcher_attestation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from experiments.official_pusht_decision_audit import _hardware_fingerprint, _runtime_fingerprint
    runtime, hardware = _runtime_fingerprint(), _hardware_fingerprint()
    return {
        "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
        "model_bindings": {model: {"checkpoint_sha256": model_specs[model].get("checkpoint_sha256"), "config_sha256": model_specs[model].get("config_sha256")} for model in V14_LOCKED_MODELS},
        "official_data_bindings": {"official_root": str(Path(official_root).resolve()), "official_git_commit": official_git_commit, "dataset_root": str(Path(dataset_root).resolve()), "dataset_split": "val", "dataset_archive_sha256": dataset_archive_sha256, "dataset_files": dict(dataset_files), "seq_lengths_sha256": seq_lengths_sha256, "dataset_states_sha256": dataset_states_sha256, "runtime": runtime, "hardware": hardware, "runtime_sha256": canonical_sha256(runtime), "hardware_sha256": canonical_sha256(hardware)},
        "dinov2": dict(dinov2), "certificate_schedule": _certificate_schedule(contracts),
        "certificate_output_dir": str(Path(certificate_output_dir).resolve()),
        "canonical_project_root": str(V14_CANONICAL_PROJECT_ROOT),
        "launcher_path": str(V14_CANONICAL_LAUNCHER_PATH),
        "launcher_sha256": file_sha256(V14_CANONICAL_LAUNCHER_PATH),
        "launcher_attestation": dict(launcher_attestation) if launcher_attestation is not None else None,
    }


def _certificate_issuance_record(bindings: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(bindings)
    normalized.setdefault("canonical_project_root", str(V14_CANONICAL_PROJECT_ROOT))
    normalized.setdefault("launcher_path", str(V14_CANONICAL_LAUNCHER_PATH))
    normalized.setdefault("launcher_sha256", file_sha256(V14_CANONICAL_LAUNCHER_PATH))
    normalized.setdefault("launcher_attestation", None)
    base = {"schema_version": V14_CERTIFICATE_ISSUANCE_SCHEMA_VERSION, "status": "consumed", "one_shot": True, "run_id": uuid.uuid4().hex, "certificate_attempt_ledger_path": str(_certificate_ledger_path()), **normalized}
    base["certificate_schedule_sha256"] = canonical_sha256(base["certificate_schedule"])
    return {**base, "issuance_sha256": canonical_sha256(base)}


def validate_v14_certificate_issuance(issuance: Mapping[str, Any], *, expected_bindings: Mapping[str, Any] | None = None) -> list[str]:
    required = {"schema_version", "status", "one_shot", "run_id", "protocol_sha256", "source_bundle_sha256", "model_bindings", "official_data_bindings", "dinov2", "certificate_schedule", "certificate_output_dir", "certificate_attempt_ledger_path", "certificate_schedule_sha256", "canonical_project_root", "launcher_path", "launcher_sha256", "launcher_attestation", "issuance_sha256"}
    errors: list[str] = []
    if not isinstance(issuance, Mapping) or set(issuance) != required:
        return ["v1.4 certificate issuance has an unexpected schema"]
    if issuance.get("schema_version") != V14_CERTIFICATE_ISSUANCE_SCHEMA_VERSION:
        errors.append("v1.4 issuance schema mismatch")
    if issuance.get("status") != "consumed" or issuance.get("one_shot") is not True:
        errors.append("v1.4 issuance is not a consumed one-shot record")
    if not isinstance(issuance.get("run_id"), str) or len(str(issuance.get("run_id"))) != 32:
        errors.append("v1.4 issuance lacks its immutable run_id")
    if issuance.get("protocol_sha256") != v14_protocol_sha256() or issuance.get("source_bundle_sha256") != v14_source_bundle_sha256():
        errors.append("v1.4 issuance protocol/source binding mismatch")
    if issuance.get("canonical_project_root") != str(V14_CANONICAL_PROJECT_ROOT):
        errors.append("v1.4 issuance canonical project-root binding mismatch")
    if issuance.get("launcher_path") != str(V14_CANONICAL_LAUNCHER_PATH) or issuance.get("launcher_sha256") != file_sha256(V14_CANONICAL_LAUNCHER_PATH):
        errors.append("v1.4 issuance launcher binding mismatch")
    if issuance.get("certificate_schedule_sha256") != canonical_sha256(issuance.get("certificate_schedule")):
        errors.append("v1.4 issuance schedule hash mismatch")
    if not isinstance(issuance.get("certificate_output_dir"), str) or not Path(str(issuance.get("certificate_output_dir"))).is_absolute():
        errors.append("v1.4 issuance lacks an absolute certificate output-root binding")
    try:
        canonical_ledger = str(_certificate_ledger_path())
    except ValueError:
        canonical_ledger = str(Path(V14_CERTIFICATE_LEDGER_PATH).resolve())
        errors.append("v1.4 issuance certificate attempt ledger path is not canonical")
    if issuance.get("certificate_attempt_ledger_path") != canonical_ledger:
        errors.append("v1.4 issuance certificate attempt ledger path is not canonical")
    if issuance.get("issuance_sha256") != canonical_sha256({key: value for key, value in issuance.items() if key != "issuance_sha256"}):
        errors.append("v1.4 issuance hash mismatch")
    for key, value in (expected_bindings or {}).items():
        if issuance.get(key) != value:
            errors.append(f"v1.4 issuance {key} binding mismatch")
    return sorted(set(errors))


def _consume_v14_certificate_issuance(bindings: Mapping[str, Any], *, issuance_path: Path | str | None = None) -> tuple[Path, dict[str, Any]]:
    path = _canonical_certificate_issuance_path(issuance_path)
    if path == V14_CERTIFICATE_ISSUANCE_PATH and not bindings.get("launcher_attestation"):
        raise RuntimeError("v1.4 certificate issuance requires a consumed canonical launcher attestation")
    record = _certificate_issuance_record(bindings)
    errors = validate_v14_certificate_issuance(record, expected_bindings=bindings)
    if errors:
        raise ValueError("invalid v1.4 certificate issuance: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"v1.4 certificate issuance already consumed: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return path, record


def _certificate_issuance_reference(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path)), "issuance_sha256": record.get("issuance_sha256")}


def _certificate_ledger_commitments(
    issuance_reference: Mapping[str, Any], output_dir: Path, contracts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    schedule = _certificate_schedule(contracts)
    return {
        "schema_version": V14_CERTIFICATE_LEDGER_SCHEMA_VERSION,
        "certificate_issuance": dict(issuance_reference),
        "certificate_output_dir": str(Path(output_dir).resolve()),
        "certificate_schedule_sha256": canonical_sha256(schedule),
        "attempt_count": len(V14_LOCKED_MODELS) * V14_CERTIFICATE_REPEAT_COUNT * len(contracts),
    }


def _certificate_ledger_payload_hash(event: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in event.items() if key != "event_sha256"})


def _new_certificate_ledger(
    ledger: Path, commitments: Mapping[str, Any], *, run_id: str,
) -> tuple[dict[str, Any], str]:
    header = {"status": "run_header", "run_id": str(run_id), "previous_event_sha256": None, **dict(commitments)}
    header["event_sha256"] = _certificate_ledger_payload_hash(header)
    payload = (json.dumps(header, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(Path(ledger), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return header, str(header["event_sha256"])


def _append_certificate_ledger_event(
    ledger: Path, payload: Mapping[str, Any], *, header: Mapping[str, Any], previous: str,
) -> str:
    event = {
        **dict(payload), "run_id": header["run_id"],
        "run_commitment_sha256": header["event_sha256"],
        "previous_event_sha256": previous,
    }
    event["event_sha256"] = _certificate_ledger_payload_hash(event)
    with Path(ledger).open("ab") as handle:
        handle.write((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return str(event["event_sha256"])


def read_v14_certificate_ledger(
    ledger: Path, *, expected_commitments: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, list[Mapping[str, Any]]], list[str], str | None]:
    try:
        records = [json.loads(line) for line in Path(ledger).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as exc:
        return None, {}, [f"invalid v1.4 certificate attempt ledger: {exc}"], None
    if not records or not isinstance(records[0], Mapping) or records[0].get("status") != "run_header":
        return None, {}, ["v1.4 certificate attempt ledger lacks its exclusive run header"], None
    header, errors, previous = records[0], [], None
    run_id = header.get("run_id")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"v1.4 certificate ledger event {index} is not an object")
            continue
        if record.get("previous_event_sha256") != previous:
            errors.append(f"v1.4 certificate ledger event {index} breaks predecessor chain")
        if record.get("event_sha256") != _certificate_ledger_payload_hash(record):
            errors.append(f"v1.4 certificate ledger event {index} hash mismatch")
        if index and record.get("run_id") != run_id:
            errors.append(f"v1.4 certificate ledger event {index} run_id mismatch")
        if index and record.get("run_commitment_sha256") != header.get("event_sha256"):
            errors.append(f"v1.4 certificate ledger event {index} run commitment mismatch")
        previous = record.get("event_sha256")
    for key, value in (expected_commitments or {}).items():
        if header.get(key) != value:
            errors.append(f"v1.4 certificate ledger header {key} commitment mismatch")
    histories: dict[str, list[Mapping[str, Any]]] = {}
    for record in records[1:]:
        attempt_id = record.get("attempt_id") if isinstance(record, Mapping) else None
        if not isinstance(attempt_id, str):
            errors.append("v1.4 certificate ledger event lacks attempt_id")
            continue
        histories.setdefault(attempt_id, []).append(record)
    return dict(header), histories, sorted(set(errors)), previous


def _create_certificate_attempt_capability(
    *, ledger: Path, header: Mapping[str, Any], started_event_sha256: str,
    issuance_reference: Mapping[str, Any], output_dir: Path, model: str,
    contract: Mapping[str, Any], repeat_index: int, ordinal: int,
    artifact_path: Path, arrays_path: Path, stdout_path: Path, stderr_path: Path,
    attempt_path: Path,
) -> Path:
    path = _certificate_capability_path(str(header["run_id"]), ordinal)
    record = {
        "schema_version": V14_CERTIFICATE_CAPABILITY_SCHEMA_VERSION,
        "status": "authorized", "one_shot": True,
        "certificate_issuance": dict(issuance_reference), "run_id": header["run_id"],
        "run_commitment_sha256": header["event_sha256"],
        "started_event_sha256": started_event_sha256,
        "ledger_path": str(Path(ledger).resolve()),
        "attempt_id": _certificate_attempt_id(model, str(contract["contract_id"]), repeat_index),
        "ordinal": int(ordinal), "model": str(model),
        "contract_id": str(contract["contract_id"]), "repeat_index": int(repeat_index),
        "certificate_output_dir": str(Path(output_dir).resolve()),
        "artifact_path": str(Path(artifact_path).resolve()),
        "arrays_path": str(Path(arrays_path).resolve()),
        "stdout_path": str(Path(stdout_path).resolve()),
        "stderr_path": str(Path(stderr_path).resolve()),
        "attempt_path": str(Path(attempt_path).resolve()),
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


def _consumed_certificate_capability_path(path: Path) -> Path:
    path = Path(path).resolve()
    return path.with_name(path.stem + ".consumed.json")


def _certificate_capability_state(path: Path) -> dict[str, Any]:
    authorized = Path(path).resolve()
    consumed = _consumed_certificate_capability_path(authorized)
    actual, status = (consumed, "ambiguous") if consumed.is_file() and authorized.is_file() else (consumed, "consumed") if consumed.is_file() else (authorized, "authorized") if authorized.is_file() else (authorized, "missing")
    commitment = None
    if actual.is_file():
        try:
            value = json.loads(actual.read_text(encoding="utf-8"))
            commitment = value.get("capability_sha256") if isinstance(value, Mapping) else None
        except (OSError, ValueError, TypeError):
            pass
    return {"status": status, "path": str(actual), "sha256": file_sha256(actual), "capability_sha256": commitment}


def _artifact_signature(artifact: Mapping[str, Any], arrays_path: Path, artifact_path: Path | None = None) -> dict[str, Any]:
    with np.load(Path(arrays_path), allow_pickle=False) as archive:
        array_hashes = {key: array_sha256(archive[key]) for key in sorted(archive.files)}
    binding_fields = ("model", "protocol_sha256", "source_bundle_sha256", "checkpoint_sha256", "config_sha256", "official_git_commit", "official_root", "dataset_root", "dataset_split", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "states_segment_sha256", "actions_segment_sha256", "environment_seed", "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256", "goal_observation_sha256", "dinov2", "runtime", "hardware", "runtime_sha256", "hardware_sha256")
    payload = {"array_hashes": array_hashes, "artifact_sha256": file_sha256(Path(artifact_path)) if artifact_path is not None else None, "arrays_sha256": file_sha256(Path(arrays_path)), "decision": artifact.get("decision"), "standard_outer_metrics": artifact.get("standard_outer_metrics"), "v14_reencoding_diagnostic": artifact.get("v14_reencoding_diagnostic"), "latent_coordinate_drift": artifact.get("latent_coordinate_drift"), "certificate_issuance": artifact.get("certificate_issuance"), "certificate_attempt_capability": artifact.get("certificate_attempt_capability"), "bindings": {key: artifact.get(key) for key in binding_fields}}
    return {**payload, "signature_sha256": canonical_sha256(payload)}


def make_v14_contracts(dataset_root: Path) -> list[dict[str, Any]]:
    import pickle
    with (Path(dataset_root) / "val" / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(value) for value in pickle.load(handle)]
    if len(lengths) != 21 or lengths[V14_EXCLUDED_TRAJECTORY] - 31 < 0:
        raise ValueError("v1.4 contracts require 21 validation trajectories and a usable excluded trajectory")
    offset = int((2 * (lengths[V14_EXCLUDED_TRAJECTORY] - 31)) // 4)
    return [{"contract_id": f"v14_ep007_mid_seed{seed:02d}", "trajectory_id": "val_episode_007", "episode": 7, "offset": offset, "quantile_index": 2, "segment_length": 31, "seed": seed, "inventory": "common_initial", "outcome_blind": True} for seed in V14_LOCKED_SEEDS]


def _segment_binding_values(dataset_root: Path, episode: int, offset: int) -> dict[str, str]:
    import torch
    root = Path(dataset_root) / "val"
    states = torch.load(root / "states.pth", map_location="cpu", weights_only=False).float()
    velocities = torch.load(root / "velocities.pth", map_location="cpu", weights_only=False).float()
    raw_actions = torch.load(root / "rel_actions.pth", map_location="cpu", weights_only=False).float() / 100.0
    full_states = torch.cat([states, velocities], dim=-1).numpy()
    mean = torch.tensor([-0.0087, 0.0068], dtype=torch.float32)
    std = torch.tensor([0.2019, 0.2002], dtype=torch.float32)
    normalized = ((raw_actions - mean) / std).numpy()
    return {
        "states_segment_sha256": array_sha256(full_states[episode, offset:offset + 31]),
        "actions_segment_sha256": array_sha256(normalized[episode, offset:offset + 30]),
        "initial_state_sha256": array_sha256(full_states[episode, offset].astype(np.float64)),
    }


def build_v14_certificate(model: str, model_spec: Mapping[str, Any], contracts: list[Mapping[str, Any]], results: list[Mapping[str, Any]], *, certificate_issuance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if model not in V14_LOCKED_MODELS or len(contracts) != 3 or len(results) != 3:
        raise ValueError("v1.4 certificate requires the released model and three contracts")
    records = []
    for contract, result in zip(contracts, results):
        records.append({"contract": dict(contract), "passed": result.get("passed") is True, "repeat_count": int(result.get("repeat_count", 0)), "repeat0_signature": result.get("repeat0_signature"), "repeat_signatures": list(result.get("repeat_signatures", [])), "attempts": list(result.get("attempts", [])), "comparison_errors": list(result.get("comparison_errors", []))})
    passed = all(record["passed"] and record["repeat_count"] == 3 and not record["comparison_errors"] for record in records)
    certificate = {"schema_version": V14_CERTIFICATE_SCHEMA_VERSION, "scientific_result": False, "fresh_process_supervisor": True, "supervisor_version": "PWA-PushT-fresh-process-supervisor-v1.4-common-initial", "supervisor_status": "passed" if passed else "failed", "inventory": "common_initial_only", "outcome_blind": True, "model": model, "checkpoint_sha256": model_spec.get("checkpoint_sha256"), "config_sha256": model_spec.get("config_sha256"), "protocol": V14_PROTOCOL, "protocol_sha256": v14_protocol_sha256(), "source_bundle_version": V14_SOURCE_BUNDLE_VERSION, "source_bundle_sha256": v14_source_bundle_sha256(), "contracts": records, "contract_schedule_sha256": canonical_sha256([dict(item) for item in contracts]), "repeatability": {"passed": passed, "repeat_count": 3, "fresh_process": True, "comparison_anchor": "repeat_0", "score_tolerance": 2e-5}, "cross_model_common_initial_gate": None, "certificate_issuance": dict(certificate_issuance) if certificate_issuance is not None else None}
    return certificate


def validate_v14_certificate(certificate: Mapping[str, Any], *, expected_model: str | None = None, expected_contracts: list[Mapping[str, Any]] | None = None, expected_issuance: Mapping[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if certificate.get("schema_version") != V14_CERTIFICATE_SCHEMA_VERSION:
        errors.append("certificate is not the v1.4 initial-only schema")
    if certificate.get("scientific_result") is not False or certificate.get("fresh_process_supervisor") is not True:
        errors.append("v1.4 certificate is not engineering-only fresh-process evidence")
    if certificate.get("inventory") != "common_initial_only" or certificate.get("outcome_blind") is not True:
        errors.append("v1.4 certificate inventory binding mismatch")
    if expected_model is not None and certificate.get("model") != expected_model:
        errors.append("v1.4 certificate model mismatch")
    if certificate.get("protocol") != V14_PROTOCOL or certificate.get("protocol_sha256") != v14_protocol_sha256():
        errors.append("v1.4 certificate protocol binding mismatch")
    if certificate.get("source_bundle_version") != V14_SOURCE_BUNDLE_VERSION or certificate.get("source_bundle_sha256") != v14_source_bundle_sha256():
        errors.append("v1.4 certificate source binding mismatch")
    if expected_issuance is not None and certificate.get("certificate_issuance") != dict(expected_issuance):
        errors.append("v1.4 certificate issuance binding mismatch")
    contracts = certificate.get("contracts")
    if not isinstance(contracts, list) or len(contracts) != 3:
        errors.append("v1.4 certificate must contain three contracts")
    else:
        if expected_contracts is not None and [record.get("contract") for record in contracts if isinstance(record, Mapping)] != [dict(item) for item in expected_contracts]:
            errors.append("v1.4 certificate contracts do not match canonical schedule")
        schedule_items = expected_contracts if expected_contracts is not None else [record.get("contract") if isinstance(record, Mapping) and isinstance(record.get("contract"), Mapping) else {} for record in contracts]
        if certificate.get("contract_schedule_sha256") != canonical_sha256([dict(item) for item in schedule_items]):
            errors.append("v1.4 certificate schedule hash mismatch")
        for index, record in enumerate(contracts):
            repeat_count = record.get("repeat_count", 0) if isinstance(record, Mapping) else 0
            try:
                repeat_count = int(repeat_count)
            except (TypeError, ValueError):
                repeat_count = 0
            if not isinstance(record, Mapping) or record.get("passed") is not True or repeat_count != 3:
                errors.append(f"v1.4 contract {index} lacks three passed fresh-process repeats")
            if not isinstance(record, Mapping) or not isinstance(record.get("contract"), Mapping) or record["contract"].get("inventory") != "common_initial" or record["contract"].get("outcome_blind") is not True:
                errors.append(f"v1.4 contract {index} is malformed")
            if not isinstance(record, Mapping) or not isinstance(record.get("repeat_signatures"), list) or len(record["repeat_signatures"]) != 3:
                errors.append(f"v1.4 contract {index} lacks three repeat signatures")
            if not isinstance(record, Mapping) or record.get("comparison_errors"):
                errors.append(f"v1.4 contract {index} has repeat comparison errors")
            attempts = record.get("attempts") if isinstance(record, Mapping) else None
            if not isinstance(attempts, list) or len(attempts) != 3:
                errors.append(f"v1.4 contract {index} lacks its three attempt receipts")
            elif any(not isinstance(attempt, Mapping) or attempt.get("status") != "passed" or attempt.get("returncode") != 0 for attempt in attempts):
                errors.append(f"v1.4 contract {index} contains a non-passed attempt receipt")
    if certificate.get("supervisor_status") != "passed":
        errors.append("v1.4 certificate supervisor status is not passed")
    return sorted(set(errors))


def build_v14_bundle(certificates: Mapping[str, Mapping[str, Any]], *, dataset_root: Path | str, official_root: Path | str | None, contracts: list[Mapping[str, Any]], cross_model_gate: Mapping[str, Any], attempts: Mapping[str, Any] | None = None, certificate_issuance: Mapping[str, Any] | None = None, certificate_attempt_ledger: Mapping[str, Any] | None = None, dataset_bindings: Mapping[str, Any] | None = None, dinov2: Mapping[str, Any] | None = None, launcher_attestation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if set(certificates) != set(V14_LOCKED_MODELS):
        raise ValueError("v1.4 bundle requires exactly one released certificate per model")
    attempt_map = {model: list((attempts or {}).get(model, [])) for model in V14_LOCKED_MODELS}
    runtime = hardware = official_git_commit = None
    for model in V14_LOCKED_MODELS:
        for attempt in attempt_map[model]:
            bindings = attempt.get("signature", {}).get("bindings", {}) if isinstance(attempt, Mapping) else {}
            if isinstance(bindings, Mapping) and isinstance(bindings.get("runtime"), Mapping) and isinstance(bindings.get("hardware"), Mapping):
                runtime, hardware = dict(bindings["runtime"]), dict(bindings["hardware"])
                official_git_commit = bindings.get("official_git_commit")
                break
        if runtime is not None:
            break
    bundle = {"schema_version": V14_BUNDLE_SCHEMA_VERSION, "scientific_result": False, "supervisor_version": V14_PROTOCOL["repeat_certificate"], "inventory": "common_initial_only", "outcome_blind": True, "canonical_project_root": str(V14_CANONICAL_PROJECT_ROOT), "launcher_path": str(V14_CANONICAL_LAUNCHER_PATH), "launcher_sha256": file_sha256(V14_CANONICAL_LAUNCHER_PATH), "launcher_attestation": dict(launcher_attestation) if launcher_attestation is not None else None, "dataset_root": str(Path(dataset_root).resolve()), "official_root": str(Path(official_root).resolve()) if official_root is not None else None, "official_git_commit": official_git_commit, "dataset_bindings": dict(dataset_bindings or {}), "dinov2": dict(dinov2 or {}), "runtime": runtime, "hardware": hardware, "runtime_sha256": canonical_sha256(runtime) if runtime is not None else None, "hardware_sha256": canonical_sha256(hardware) if hardware is not None else None, "protocol": V14_PROTOCOL, "protocol_sha256": v14_protocol_sha256(), "source_bundle_version": V14_SOURCE_BUNDLE_VERSION, "source_bundle_sha256": v14_source_bundle_sha256(), "contracts": [dict(item) for item in contracts], "certificates": {model: dict(certificates[model]) for model in V14_LOCKED_MODELS}, "certificate_sha256": {model: canonical_sha256(certificates[model]) for model in V14_LOCKED_MODELS}, "cross_model_common_initial_gate": dict(cross_model_gate), "attempts": attempt_map, "certificate_issuance": dict(certificate_issuance) if certificate_issuance is not None else None, "certificate_attempt_ledger": dict(certificate_attempt_ledger) if certificate_attempt_ledger is not None else None}
    bundle["status"] = "passed" if cross_model_gate.get("passed") is True and all(certificates[model].get("supervisor_status") == "passed" for model in V14_LOCKED_MODELS) else "failed"
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


def validate_v14_bundle(bundle: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(bundle, Mapping):
        return ["v1.4 certificate bundle is not an object"]
    if bundle.get("schema_version") == "PWA-PushT-certificate-bundle-v1.2-common-initial":
        errors.append("v1.4 bundle validator rejects v1.2 bundles")
    if bundle.get("schema_version") != V14_BUNDLE_SCHEMA_VERSION:
        errors.append("bundle is not the v1.4 schema")
    if bundle.get("canonical_project_root") != str(V14_CANONICAL_PROJECT_ROOT):
        errors.append("v1.4 bundle canonical project-root binding mismatch")
    if bundle.get("launcher_path") != str(V14_CANONICAL_LAUNCHER_PATH) or bundle.get("launcher_sha256") != file_sha256(V14_CANONICAL_LAUNCHER_PATH):
        errors.append("v1.4 bundle launcher binding mismatch")
    if bundle.get("scientific_result") is not False or bundle.get("inventory") != "common_initial_only" or bundle.get("outcome_blind") is not True:
        errors.append("v1.4 bundle is not an engineering-only common-initial record")
    if bundle.get("protocol") != V14_PROTOCOL or bundle.get("protocol_sha256") != v14_protocol_sha256():
        errors.append("v1.4 bundle protocol binding mismatch")
    if bundle.get("source_bundle_version") != V14_SOURCE_BUNDLE_VERSION or bundle.get("source_bundle_sha256") != v14_source_bundle_sha256():
        errors.append("v1.4 bundle source binding mismatch")
    for value_key, hash_key in (("runtime", "runtime_sha256"), ("hardware", "hardware_sha256")):
        value, advertised_hash = bundle.get(value_key), bundle.get(hash_key)
        if not isinstance(value, Mapping) or advertised_hash != canonical_sha256(value):
            errors.append(f"v1.4 bundle {value_key} binding is missing or mutable")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if bundle.get("bundle_sha256") != canonical_sha256(unsigned):
        errors.append("v1.4 bundle hash mismatch")

    dataset_root_value, official_root_value = bundle.get("dataset_root"), bundle.get("official_root")
    if bundle.get("official_git_commit") is None:
        errors.append("v1.4 bundle lacks mandatory official_git_commit binding")
    dataset_bindings = bundle.get("dataset_bindings")
    if not isinstance(dataset_root_value, str) or not isinstance(official_root_value, str) or not isinstance(dataset_bindings, Mapping):
        errors.append("v1.4 bundle lacks immutable official/dataset bindings")
    expected_contracts: list[dict[str, Any]] = []
    if isinstance(dataset_root_value, str):
        try:
            expected_contracts = make_v14_contracts(Path(dataset_root_value))
            if bundle.get("contracts") != expected_contracts:
                errors.append("v1.4 bundle contracts are not the canonical schedule")
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            errors.append(f"v1.4 bundle contracts cannot be regenerated: {exc}")

    issuance_ref = bundle.get("certificate_issuance")
    issuance_record: Mapping[str, Any] | None = None
    canonical_path = _certificate_issuance_path()
    if not isinstance(issuance_ref, Mapping) or set(issuance_ref) != {"path", "sha256", "issuance_sha256"}:
        errors.append("v1.4 bundle lacks a complete certificate-issuance reference")
    else:
        if issuance_ref.get("path") != str(canonical_path):
            errors.append("v1.4 bundle certificate issuance path is not canonical")
        try:
            if not canonical_path.is_file():
                raise OSError("canonical issuance record is missing")
            if issuance_ref.get("sha256") != file_sha256(canonical_path):
                errors.append("v1.4 bundle certificate issuance file hash mismatch")
            loaded = json.loads(canonical_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("issuance record is not an object")
            issuance_record = loaded
            errors.extend(validate_v14_certificate_issuance(loaded))
            for key in ("canonical_project_root", "launcher_path", "launcher_sha256", "launcher_attestation"):
                if bundle.get(key) != loaded.get(key):
                    errors.append(f"v1.4 bundle {key} does not match certificate issuance")
            if issuance_ref.get("issuance_sha256") != loaded.get("issuance_sha256"):
                errors.append("v1.4 bundle certificate issuance commitment mismatch")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"v1.4 bundle certificate issuance cannot be verified: {exc}")

    certificates = bundle.get("certificates")
    if not isinstance(certificates, Mapping) or set(certificates) != set(V14_LOCKED_MODELS):
        errors.append("v1.4 bundle does not bind exactly JEPA and DINO certificates")
        certificates = {}
    certificate_maps = {model: certificates.get(model) if isinstance(certificates.get(model), Mapping) else {} for model in V14_LOCKED_MODELS}
    advertised = bundle.get("certificate_sha256")
    if not isinstance(advertised, Mapping):
        errors.append("v1.4 bundle lacks certificate hashes")
        advertised = {}
    for model in V14_LOCKED_MODELS:
        certificate = certificates.get(model)
        if not isinstance(certificate, Mapping):
            continue
        errors.extend(f"{model}: {error}" for error in validate_v14_certificate(certificate, expected_model=model, expected_contracts=expected_contracts or None, expected_issuance=issuance_ref if isinstance(issuance_ref, Mapping) else None))
        if advertised.get(model) != canonical_sha256(certificate):
            errors.append(f"{model}: v1.4 certificate hash mismatch")

    # Re-open the released files and dependency binding instead of trusting
    # the JSON claims in the bundle.
    try:
        from experiments.official_pusht_batch_study import (
            RELEASED_OFFICIAL_COMMIT, _archive_matches_extracted_split,
            _dataset_files_fingerprint, _git_commit, _git_tracked_clean,
            dinov2_binding, validate_dinov2_bindings, validate_known_release_bindings,
        )
        actual_dataset = {
            "dataset_split": "val", "seq_lengths_sha256": file_sha256(Path(dataset_root_value) / "val" / "seq_lengths.pkl"),
            "dataset_states_sha256": file_sha256(Path(dataset_root_value) / "val" / "states.pth"),
            "dataset_archive_sha256": file_sha256(Path(official_root_value) / "pusht_noise.zip"),
            "dataset_files": _dataset_files_fingerprint(Path(dataset_root_value), "val"),
        }
        if dict(dataset_bindings or {}) != actual_dataset:
            errors.append("v1.4 bundle dataset bindings do not match current files")
        if not _git_tracked_clean(Path(official_root_value)) or _git_commit(Path(official_root_value)) != RELEASED_OFFICIAL_COMMIT:
            errors.append("v1.4 bundle official checkout is not the clean released commit")
        if bundle.get("official_git_commit") != RELEASED_OFFICIAL_COMMIT:
            errors.append("v1.4 bundle official_git_commit does not match the released checkout")
        if not _archive_matches_extracted_split(Path(official_root_value) / "pusht_noise.zip", Path(dataset_root_value), "val"):
            errors.append("v1.4 bundle dataset split does not match the released archive")
        if bundle.get("dinov2") != dinov2_binding():
            errors.append("v1.4 bundle DINOv2 binding is not canonical")
        errors.extend(f"v1.4 DINOv2: {error}" for error in validate_dinov2_bindings(bundle.get("dinov2", {}), require_complete=True, verify_files=True))
        release_bindings = {
            "official_git_commit": _git_commit(Path(official_root_value)),
            "dataset_archive_sha256": actual_dataset["dataset_archive_sha256"],
            "dataset_files": actual_dataset["dataset_files"],
            "models": {model: {"checkpoint_sha256": certificate_maps[model].get("checkpoint_sha256"), "config_sha256": certificate_maps[model].get("config_sha256")} for model in V14_LOCKED_MODELS},
            "dinov2": bundle.get("dinov2"),
        }
        errors.extend(f"v1.4 known-release binding: {error}" for error in validate_known_release_bindings(release_bindings, require_complete=True))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"v1.4 released input bindings cannot be verified: {exc}")

    if issuance_record is not None and isinstance(dataset_bindings, Mapping):
        expected_issuance_bindings = {
            "protocol_sha256": bundle.get("protocol_sha256"), "source_bundle_sha256": bundle.get("source_bundle_sha256"),
            "model_bindings": {model: {"checkpoint_sha256": certificate_maps[model].get("checkpoint_sha256"), "config_sha256": certificate_maps[model].get("config_sha256")} for model in V14_LOCKED_MODELS},
            "official_data_bindings": {"official_root": str(Path(str(official_root_value)).resolve()), "official_git_commit": RELEASED_OFFICIAL_COMMIT, "dataset_root": str(Path(str(dataset_root_value)).resolve()), **dict(dataset_bindings), "runtime": bundle.get("runtime"), "hardware": bundle.get("hardware"), "runtime_sha256": bundle.get("runtime_sha256"), "hardware_sha256": bundle.get("hardware_sha256")},
            "dinov2": bundle.get("dinov2"),
            "certificate_schedule": _certificate_schedule(bundle.get("contracts", []) if isinstance(bundle.get("contracts"), list) else []),
            "canonical_project_root": str(V14_CANONICAL_PROJECT_ROOT),
            "launcher_path": str(V14_CANONICAL_LAUNCHER_PATH),
            "launcher_sha256": file_sha256(V14_CANONICAL_LAUNCHER_PATH),
            "launcher_attestation": bundle.get("launcher_attestation"),
        }
        errors.extend(validate_v14_certificate_issuance(issuance_record, expected_bindings=expected_issuance_bindings))

    ledger_ref = bundle.get("certificate_attempt_ledger")
    ledger_header: Mapping[str, Any] | None = None
    ledger_histories: dict[str, list[Mapping[str, Any]]] = {}
    try:
        ledger_path = _certificate_ledger_path()
    except ValueError as exc:
        ledger_path = Path(V14_CERTIFICATE_LEDGER_PATH).resolve()
        errors.append(str(exc))
    if not isinstance(ledger_ref, Mapping) or set(ledger_ref) != {"path", "sha256", "run_id", "header_sha256"}:
        errors.append("v1.4 bundle lacks a complete certificate attempt-ledger reference")
    else:
        if ledger_ref.get("path") != str(ledger_path):
            errors.append("v1.4 bundle certificate attempt ledger path is not canonical")
        try:
            if not ledger_path.is_file():
                raise OSError("canonical certificate attempt ledger is missing")
            if ledger_ref.get("sha256") != file_sha256(ledger_path):
                errors.append("v1.4 bundle certificate attempt ledger file hash mismatch")
            expected_commitments = _certificate_ledger_commitments(
                issuance_ref if isinstance(issuance_ref, Mapping) else {},
                Path(str(issuance_record.get("certificate_output_dir"))) if isinstance(issuance_record, Mapping) else Path("."),
                expected_contracts,
            )
            ledger_header, ledger_histories, ledger_errors, _tail = read_v14_certificate_ledger(ledger_path, expected_commitments=expected_commitments)
            errors.extend(ledger_errors)
            if ledger_header is None:
                raise ValueError("certificate attempt ledger header is missing")
            if ledger_ref.get("run_id") != ledger_header.get("run_id") or ledger_ref.get("header_sha256") != ledger_header.get("event_sha256"):
                errors.append("v1.4 bundle certificate attempt ledger header commitment mismatch")
            if isinstance(issuance_record, Mapping) and ledger_header.get("run_id") != issuance_record.get("run_id"):
                errors.append("v1.4 certificate attempt ledger run_id differs from issuance")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"v1.4 certificate attempt ledger cannot be verified: {exc}")

    attempts = bundle.get("attempts")
    if not isinstance(attempts, Mapping) or set(attempts) != set(V14_LOCKED_MODELS):
        errors.append("v1.4 bundle must preserve attempts for both models")
        attempts = {}
    contract_by_id = {str(item.get("contract_id")): item for item in expected_contracts}
    contract_ordinal = {str(item.get("contract_id")): index for index, item in enumerate(expected_contracts)}
    certificate_root = Path(str(issuance_record.get("certificate_output_dir"))).resolve() if isinstance(issuance_record, Mapping) else None
    certificate_run_id = issuance_record.get("run_id") if isinstance(issuance_record, Mapping) else None
    repeat_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in V14_LOCKED_MODELS:
        model_attempts = attempts.get(model, []) if isinstance(attempts, Mapping) else []
        if not isinstance(model_attempts, list) or len(model_attempts) != 9:
            errors.append(f"v1.4 {model} must preserve exactly nine canonical attempts")
            continue
        seen: set[tuple[str, int]] = set()
        for attempt in model_attempts:
            if not isinstance(attempt, Mapping):
                errors.append(f"v1.4 {model} contains a malformed attempt")
                continue
            try:
                identity = (str(attempt["contract_id"]), int(attempt["repeat_index"]))
            except (KeyError, TypeError, ValueError):
                errors.append(f"v1.4 {model} attempt lacks canonical identity")
                continue
            if identity in seen or identity[0] not in contract_by_id or identity[1] not in (0, 1, 2):
                errors.append(f"v1.4 {model} attempt is not a unique canonical contract/repeat")
                continue
            seen.add(identity)
            expected_ordinal = V14_LOCKED_MODELS.index(model) * 9 + contract_ordinal[identity[0]] * 3 + identity[1]
            attempt_id = _certificate_attempt_id(model, identity[0], identity[1])
            history = ledger_histories.get(attempt_id, [])
            if [item.get("status") for item in history] not in (["started", "completed"], ["started", "failed"]):
                errors.append(f"v1.4 {model} {identity} has no exactly-one-start legal certificate history")
            if attempt.get("attempt_id") != attempt_id or attempt.get("attempt_ordinal") != expected_ordinal or attempt.get("certificate_run_id") != certificate_run_id:
                errors.append(f"v1.4 {model} {identity} attempt ordinal/run identity mismatch")
            if history:
                started = history[0]
                capability_path = _certificate_capability_path(str(certificate_run_id), expected_ordinal) if certificate_run_id else None
                for key, value in (("attempt_id", attempt_id), ("ordinal", expected_ordinal), ("model", model), ("contract_id", identity[0]), ("repeat_index", identity[1]), ("certificate_output_dir", str(certificate_root) if certificate_root is not None else None), ("artifact_path", str(Path(str(attempt.get("artifact", ""))).resolve())), ("arrays_path", str(Path(str(attempt.get("arrays", ""))).resolve())), ("stdout_path", str(Path(str(attempt.get("stdout", ""))).resolve())), ("stderr_path", str(Path(str(attempt.get("stderr", ""))).resolve())), ("attempt_path", str(Path(str(attempt.get("attempt", ""))).resolve()))):
                    if started.get(key) != value:
                        errors.append(f"v1.4 {model} {identity} started event {key} does not bind its receipt")
                if capability_path is not None and started.get("capability_path") != str(capability_path):
                    errors.append(f"v1.4 {model} {identity} capability path is not canonical")
                if len(history) == 2:
                    terminal = history[1]
                    expected_terminal = "completed" if attempt.get("status") == "passed" else "failed"
                    if terminal.get("status") != expected_terminal:
                        errors.append(f"v1.4 {model} {identity} terminal status differs from receipt")
                    for key, value in (("attempt_id", attempt_id), ("ordinal", expected_ordinal), ("model", model), ("contract_id", identity[0]), ("repeat_index", identity[1]), ("certificate_output_dir", str(certificate_root) if certificate_root is not None else None), ("certificate_issuance", dict(issuance_ref) if isinstance(issuance_ref, Mapping) else None)):
                        if terminal.get(key) != value:
                            errors.append(f"v1.4 {model} {identity} terminal receipt {key} does not bind its attempt")
                    for key, value in (("artifact_exists", Path(str(attempt.get("artifact", ""))).is_file()), ("arrays_exists", Path(str(attempt.get("arrays", ""))).is_file()), ("artifact_sha256", attempt.get("artifact_sha256")), ("arrays_sha256", attempt.get("arrays_sha256")), ("stdout_sha256", attempt.get("stdout_sha256")), ("stderr_sha256", attempt.get("stderr_sha256")), ("attempt_path", str(Path(str(attempt.get("attempt", ""))).resolve()))):
                        if terminal.get(key) != value:
                            errors.append(f"v1.4 {model} {identity} terminal receipt {key} does not recompute")
                    if terminal.get("certificate_capability") != attempt.get("certificate_capability"):
                        errors.append(f"v1.4 {model} {identity} capability consumption receipt differs")
                    if not isinstance(attempt.get("certificate_capability"), Mapping) or attempt["certificate_capability"].get("status") != "consumed":
                        errors.append(f"v1.4 {model} {identity} capability was not consumed exactly once")
            if attempt.get("model") != model or attempt.get("fresh_process") is not True or attempt.get("status") != "passed" or attempt.get("returncode") != 0:
                errors.append(f"v1.4 {model} {identity} is not a passed fresh process")
                continue
            try:
                artifact_path, arrays_path = Path(str(attempt["artifact"])), Path(str(attempt["arrays"]))
                attempt_dir = artifact_path.parent.resolve()
                if certificate_root is None or attempt.get("certificate_output_dir") != str(certificate_root) or not attempt_dir.is_relative_to(certificate_root):
                    errors.append(f"v1.4 {model} {identity} attempt is outside the issuance-bound certificate root")
                if artifact_path.name != "artifact.json" or arrays_path.name != "artifact.npz" or arrays_path.parent.resolve() != attempt_dir or attempt_dir.name != f"repeat_{identity[1]:02d}" or attempt_dir.parent.name != identity[0] or attempt_dir.parent.parent.name != model:
                    errors.append(f"v1.4 {model} {identity} evidence paths are not canonical")
                if attempt.get("stdout") != str(attempt_dir / "stdout.txt") or attempt.get("stderr") != str(attempt_dir / "stderr.txt") or attempt.get("attempt") != str(attempt_dir / "attempt.json") or attempt.get("certificate_issuance") != issuance_ref:
                    errors.append(f"v1.4 {model} {identity} receipt paths/issuance do not bind the canonical attempt")
                for field, path in (("artifact", artifact_path), ("arrays", arrays_path), ("stdout", attempt_dir / "stdout.txt"), ("stderr", attempt_dir / "stderr.txt"), ("attempt", attempt_dir / "attempt.json")):
                    if not path.is_file():
                        errors.append(f"v1.4 {model} {identity} {field} receipt is missing")
                if attempt.get("artifact_sha256") != file_sha256(artifact_path) or attempt.get("arrays_sha256") != file_sha256(arrays_path):
                    errors.append(f"v1.4 {model} {identity} artifact/NPZ hash binding mismatch")
                if attempt.get("stdout_sha256") != hashlib.sha256((attempt_dir / "stdout.txt").read_bytes()).hexdigest() or attempt.get("stderr_sha256") != hashlib.sha256((attempt_dir / "stderr.txt").read_bytes()).hexdigest():
                    errors.append(f"v1.4 {model} {identity} log hash binding mismatch")
                on_disk_attempt = json.loads((attempt_dir / "attempt.json").read_text(encoding="utf-8"))
                if on_disk_attempt != dict(attempt):
                    errors.append(f"v1.4 {model} {identity} attempt receipt differs from bundle")
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                contract = contract_by_id[identity[0]]
                entry = {"decision_id": f"v14_{model}_pusht_val_episode_007_offset_{int(contract['offset']):03d}_seed_{int(contract['seed']):02d}", "trajectory_id": "val_episode_007", "episode": 7, "offset": contract["offset"], "seed": contract["seed"], "model": model}
                artifact_errors = validate_v14_artifact(artifact, entry, arrays_path=arrays_path, allow_excluded_engineering=True, expected_bindings={"checkpoint_sha256": certificate_maps[model].get("checkpoint_sha256"), "config_sha256": certificate_maps[model].get("config_sha256"), "official_git_commit": RELEASED_OFFICIAL_COMMIT, "official_root": str(Path(str(official_root_value)).resolve()), "dataset_root": str(Path(str(dataset_root_value)).resolve()), "dataset_split": dataset_bindings.get("dataset_split"), "dataset_archive_sha256": dataset_bindings.get("dataset_archive_sha256"), "seq_lengths_sha256": dataset_bindings.get("seq_lengths_sha256"), "dataset_states_sha256": dataset_bindings.get("dataset_states_sha256"), "environment_seed": 1, "menu_seed": int(contract["seed"]), "runtime": bundle.get("runtime"), "hardware": bundle.get("hardware"), "runtime_sha256": bundle.get("runtime_sha256"), "hardware_sha256": bundle.get("hardware_sha256"), **_segment_binding_values(Path(str(dataset_root_value)), 7, int(contract["offset"])), "dinov2": bundle.get("dinov2")}, expected_certificate_issuance=issuance_ref if isinstance(issuance_ref, Mapping) else None)
                errors.extend(f"v1.4 {model} {identity}: {error}" for error in artifact_errors)
                signature = _artifact_signature(artifact, arrays_path, artifact_path)
                if attempt.get("signature") != signature:
                    errors.append(f"v1.4 {model} {identity} signature does not recompute")
                if not artifact_errors:
                    repeat_evidence.setdefault((model, identity[0]), []).append({"artifact": artifact, "arrays_path": arrays_path})
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"v1.4 {model} {identity} evidence cannot be verified: {exc}")
        if len(seen) != 9:
            errors.append(f"v1.4 {model} does not contain all canonical contract/repeat attempts")
        certificate_records = certificate_maps[model].get("contracts", [])
        certificate_by_contract = {str(record.get("contract", {}).get("contract_id")): record for record in certificate_records if isinstance(record, Mapping) and isinstance(record.get("contract"), Mapping)}
        for contract in expected_contracts:
            contract_id = str(contract.get("contract_id"))
            bundle_attempts = sorted([item for item in model_attempts if isinstance(item, Mapping) and str(item.get("contract_id")) == contract_id], key=lambda item: int(item.get("repeat_index", -1)))
            certificate_record = certificate_by_contract.get(contract_id)
            if not isinstance(certificate_record, Mapping):
                errors.append(f"v1.4 {model} certificate omits {contract_id}")
                continue
            if certificate_record.get("attempts") != bundle_attempts:
                errors.append(f"v1.4 {model} certificate attempts do not match canonical bundle receipts for {contract_id}")
            signatures = [item.get("signature") for item in bundle_attempts]
            if certificate_record.get("repeat_signatures") != signatures or certificate_record.get("repeat0_signature") != (signatures[0] if signatures else None):
                errors.append(f"v1.4 {model} certificate signatures do not match canonical bundle receipts for {contract_id}")
            recomputed_errors = compare_v14_repeats(repeat_evidence.get((model, contract_id), []))
            if recomputed_errors != list(certificate_record.get("comparison_errors", [])):
                errors.append(f"v1.4 {model} repeat comparison errors are not recomputed from canonical triplets for {contract_id}")
            if recomputed_errors:
                errors.extend(f"v1.4 {model} {contract_id}: {error}" for error in recomputed_errors)

    expected_attempt_ids = {
        _certificate_attempt_id(model, str(contract["contract_id"]), repeat)
        for model in V14_LOCKED_MODELS for contract in expected_contracts for repeat in range(V14_CERTIFICATE_REPEAT_COUNT)
    }
    if set(ledger_histories) != expected_attempt_ids or len(ledger_histories) != 18:
        errors.append("v1.4 certificate attempt ledger must contain exactly 18 canonical histories")
    for attempt_id, history in ledger_histories.items():
        if [item.get("status") for item in history] not in (["started", "completed"], ["started", "failed"]):
            errors.append(f"v1.4 certificate ledger history {attempt_id} is interrupted or multiply recorded")

    recomputed_cross_errors: list[str] = []
    for contract in expected_contracts:
        pair: dict[str, Mapping[str, Any]] = {}
        pair_arrays: dict[str, dict[str, np.ndarray]] = {}
        for model in V14_LOCKED_MODELS:
            matches = [item for item in attempts.get(model, []) if isinstance(item, Mapping) and item.get("contract_id") == contract.get("contract_id") and item.get("repeat_index") == 0 and item.get("status") == "passed"]
            if matches:
                try:
                    path = Path(str(matches[0]["arrays"]))
                    with np.load(path, allow_pickle=False) as archive:
                        pair_arrays[model] = {key: np.asarray(archive[key]) for key in __import__("experiments.official_pusht_v14_common_initial", fromlist=["V14_CROSS_MODEL_ARRAY_KEYS"]).V14_CROSS_MODEL_ARRAY_KEYS}
                    pair[model] = matches[0].get("signature", {}).get("bindings", {})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    recomputed_cross_errors.append(f"{contract.get('contract_id')}: cross-model evidence unreadable ({exc})")
        if set(pair_arrays) != set(V14_LOCKED_MODELS):
            recomputed_cross_errors.append(f"{contract.get('contract_id')}: missing cross-model repeat-0 evidence")
            continue
        left, right = (pair_arrays[model] for model in V14_LOCKED_MODELS)
        for key in left:
            if not np.array_equal(left[key], right[key]):
                recomputed_cross_errors.append(f"{contract.get('contract_id')}: cross-model {key} differs")
        for key in ("official_git_commit", "official_root", "dataset_root", "dataset_split", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "states_segment_sha256", "actions_segment_sha256", "environment_seed", "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256", "goal_observation_sha256", "runtime", "hardware", "runtime_sha256", "hardware_sha256", "dinov2"):
            if pair.get(V14_LOCKED_MODELS[0], {}).get(key) != pair.get(V14_LOCKED_MODELS[1], {}).get(key):
                recomputed_cross_errors.append(f"{contract.get('contract_id')}: cross-model binding {key} differs")
    gate = bundle.get("cross_model_common_initial_gate")
    expected_gate = {"passed": not recomputed_cross_errors and all(isinstance(certificates.get(model), Mapping) and certificates[model].get("supervisor_status") == "passed" for model in V14_LOCKED_MODELS), "inventory": "common_initial_only", "models": list(V14_LOCKED_MODELS), "errors": recomputed_cross_errors}
    if gate != expected_gate:
        errors.append("v1.4 cross-model common-initial gate does not recompute from preserved receipts")
    if bundle.get("status") != "passed":
        errors.append("v1.4 bundle status is not passed")
    return sorted(set(errors))


def run_v14_supervisor(*, official_root: Path, dataset_root: Path, model_specs: Mapping[str, Mapping[str, Any]], output_dir: Path, contracts: list[Mapping[str, Any]] | None = None, python_executable: str | None = None) -> dict[str, Any]:
    """Launch 3 contracts × 3 repeats × 2 models, once each."""
    resolved_python = Path(python_executable or sys.executable).resolve()
    if resolved_python != Path(sys.executable).resolve():
        raise ValueError("v1.4 certificate children must use the exact supervisor Python executable")
    if set(model_specs) != set(V14_LOCKED_MODELS):
        raise ValueError("v1.4 supervisor requires exactly JEPA and DINO model specs")
    official_root = Path(official_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    contracts = list(make_v14_contracts(dataset_root) if contracts is None else contracts)
    if contracts != make_v14_contracts(dataset_root):
        raise ValueError("v1.4 contracts must equal make_v14_contracts(dataset_root)")
    output_dir = Path(output_dir).resolve()
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    from experiments.official_pusht_v14_batch_study import V14_STUDY_ISSUANCE_PATH, V14_STUDY_LEDGER_PATH
    certificate_issuance = Path(V14_CERTIFICATE_ISSUANCE_PATH).resolve()
    certificate_ledger = _certificate_ledger_path()
    capability_root = _certificate_capability_root()
    protected = (certificate_issuance, certificate_ledger, capability_root, V14_STUDY_ISSUANCE_PATH, V14_STUDY_LEDGER_PATH)
    if any(output_dir == path or output_dir in path.parents or path in output_dir.parents for path in protected):
        raise ValueError("v1.4 certificate output overlaps a canonical issuance or study-ledger path")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"v1.4 certificate output is not a directory: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite v1.4 certificate output: {output_dir}")
    if certificate_ledger.exists():
        raise FileExistsError(f"v1.4 certificate attempt ledger already exists: {certificate_ledger}")
    repository_root = V14_CANONICAL_PROJECT_ROOT
    launcher_path = V14_CANONICAL_LAUNCHER_PATH
    if not launcher_path.is_file() or launcher_path.is_symlink():
        raise FileNotFoundError(f"v1.4 canonical launcher source is missing or symlinked: {launcher_path}")
    certificate_issuance = _certificate_issuance_path()
    preflight_errors = v14_execution_preflight(
        official_root, project_root=V14_CANONICAL_PROJECT_ROOT,
        source_modules=V14_SOURCE_MODULES,
    )
    if preflight_errors:
        raise RuntimeError("unsafe v1.4 certificate Python environment: " + "; ".join(preflight_errors))
    from experiments.official_pusht_batch_study import (
        RELEASED_OFFICIAL_COMMIT, _archive_matches_extracted_split, _dataset_files_fingerprint,
        _git_commit, _git_tracked_clean, dinov2_binding, validate_dinov2_bindings,
        validate_known_release_bindings,
    )
    if _git_commit(official_root) != RELEASED_OFFICIAL_COMMIT:
        raise ValueError("v1.4 supervisor official checkout is not the released commit")
    if not _git_tracked_clean(official_root):
        raise ValueError("v1.4 supervisor official checkout has tracked modifications")
    if not _archive_matches_extracted_split(official_root / "pusht_noise.zip", dataset_root, "val"):
        raise ValueError("v1.4 extracted validation split does not match the released archive")
    dino_binding = dinov2_binding()
    dino_errors = validate_dinov2_bindings(dino_binding, require_complete=True, verify_files=True)
    if dino_errors:
        raise ValueError("v1.4 pinned DINOv2 dependency is invalid: " + "; ".join(dino_errors))
    from experiments.official_pusht_v14_batch_study import _full_run_environment_errors
    environment_errors = _full_run_environment_errors(official_root=official_root)
    if environment_errors:
        raise RuntimeError("unsafe v1.4 certificate Python environment: " + "; ".join(environment_errors))
    # Match the deterministic state established by every child before taking
    # the runtime/hardware binding snapshot for the one-shot issuance.
    import torch
    environment_errors = _full_run_environment_errors(torch_module=torch, official_root=official_root)
    if environment_errors:
        raise RuntimeError("unsafe v1.4 certificate Python environment: " + "; ".join(environment_errors))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    specs: dict[str, dict[str, Any]] = {}
    for model in V14_LOCKED_MODELS:
        spec = dict(model_specs[model])
        checkpoint, config = Path(spec["checkpoint"]).resolve(), Path(spec["config"]).resolve()
        if not checkpoint.is_file() or not config.is_file():
            raise ValueError(f"{model} checkpoint/config does not exist")
        spec.update({"checkpoint": str(checkpoint), "config": str(config), "checkpoint_sha256": file_sha256(checkpoint), "config_sha256": file_sha256(config)})
        specs[model] = spec
    # The one-shot tombstone is consumed before output directories or child 1.
    release_bindings = {
        "official_git_commit": RELEASED_OFFICIAL_COMMIT,
        "dataset_archive_sha256": file_sha256(official_root / "pusht_noise.zip"),
        "dataset_files": _dataset_files_fingerprint(dataset_root, "val"),
        "models": {model: {"checkpoint_sha256": specs[model]["checkpoint_sha256"], "config_sha256": specs[model]["config_sha256"]} for model in V14_LOCKED_MODELS},
        "dinov2": dino_binding,
    }
    release_errors = validate_known_release_bindings(release_bindings, require_complete=True)
    if release_errors:
        raise ValueError("v1.4 supervisor inputs are not the known release: " + "; ".join(release_errors))
    launcher_attestation = _consume_v14_launcher_attestation("experiments.official_pusht_v14_certificate_supervisor")
    bindings = _certificate_issuance_bindings(official_root=official_root, dataset_root=dataset_root, certificate_output_dir=output_dir, official_git_commit=RELEASED_OFFICIAL_COMMIT, dataset_archive_sha256=release_bindings["dataset_archive_sha256"], dataset_files=release_bindings["dataset_files"], seq_lengths_sha256=file_sha256(dataset_root / "val" / "seq_lengths.pkl"), dataset_states_sha256=file_sha256(dataset_root / "val" / "states.pth"), model_specs=specs, contracts=contracts, dinov2=dino_binding, launcher_attestation=launcher_attestation)
    if certificate_issuance.exists():
        raise FileExistsError(f"v1.4 certificate issuance already consumed: {certificate_issuance}")
    issuance_path, issuance_record = _consume_v14_certificate_issuance(bindings)
    issuance_reference = _certificate_issuance_reference(issuance_path, issuance_record)
    ledger_commitments = _certificate_ledger_commitments(issuance_reference, output_dir, contracts)
    ledger_header, ledger_previous = _new_certificate_ledger(
        certificate_ledger, ledger_commitments, run_id=str(issuance_record["run_id"]),
    )
    issuance_data = bindings["official_data_bindings"]
    runtime_binding, hardware_binding = issuance_data["runtime"], issuance_data["hardware"]
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: dict[str, list[dict[str, Any]]] = {model: [] for model in V14_LOCKED_MODELS}
    certificates: dict[str, dict[str, Any]] = {}
    from experiments.official_pusht_v14_batch_study import validate_v14_artifact
    for model_index, model in enumerate(V14_LOCKED_MODELS):
        results = []
        for contract_index, contract in enumerate(contracts):
            repeat_records = []
            contract_dir = output_dir / model / str(contract["contract_id"])
            contract_dir.mkdir(parents=True, exist_ok=True)
            for repeat_index in range(3):
                attempt_ordinal = model_index * 9 + contract_index * 3 + repeat_index
                attempt_dir = contract_dir / f"repeat_{repeat_index:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                artifact_path, arrays_path = attempt_dir / "artifact.json", attempt_dir / "artifact.npz"
                attempt_id = _certificate_attempt_id(model, str(contract["contract_id"]), repeat_index)
                stdout_path, stderr_path = attempt_dir / "stdout.txt", attempt_dir / "stderr.txt"
                attempt_path = attempt_dir / "attempt.json"
                started_event = {
                    "attempt_id": attempt_id, "status": "started", "ordinal": attempt_ordinal,
                    "model": model, "contract_id": str(contract["contract_id"]),
                    "repeat_index": repeat_index, "certificate_output_dir": str(output_dir),
                    "artifact_path": str(artifact_path), "arrays_path": str(arrays_path),
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                    "attempt_path": str(attempt_path),
                    "capability_path": str(_certificate_capability_path(str(issuance_record["run_id"]), attempt_ordinal)),
                    "certificate_issuance": dict(issuance_reference),
                }
                ledger_previous = _append_certificate_ledger_event(
                    certificate_ledger, started_event, header=ledger_header, previous=ledger_previous,
                )
                capability_path = _create_certificate_attempt_capability(
                    ledger=certificate_ledger, header=ledger_header,
                    started_event_sha256=ledger_previous, issuance_reference=issuance_reference,
                    output_dir=output_dir, model=model, contract=contract,
                    repeat_index=repeat_index, ordinal=attempt_ordinal,
                    artifact_path=artifact_path, arrays_path=arrays_path,
                    stdout_path=stdout_path, stderr_path=stderr_path, attempt_path=attempt_path,
                )
                command = [str(resolved_python), "-I", "-B", str(launcher_path), "--module", "experiments.official_pusht_v14_decision_audit", "--official-root", str(official_root), "--checkpoint", specs[model]["checkpoint"], "--config", specs[model]["config"], "--dataset-root", str(dataset_root), "--episode", "7", "--offset", str(contract["offset"]), "--seed", str(contract["seed"]), "--out", str(artifact_path), "--arrays-out", str(arrays_path), "--repeat-count", "1", "--planning-repeats", "0", "--inventory-mode", "common_initial_only", "--model-name", model, "--decision-id", f"v14_{model}_pusht_val_episode_007_offset_{int(contract['offset']):03d}_seed_{int(contract['seed']):02d}", "--trajectory-id", "val_episode_007", "--dinov2-source-root", str(dino_binding["source_root"]), "--dinov2-source-archive", str(dino_binding["source_archive"]), "--dinov2-weights", str(dino_binding["weights"]), "--torch-home", str(dino_binding["torch_home"]), "--certificate-issuance-path", str(issuance_reference["path"]), "--certificate-issuance-sha256", str(issuance_reference["sha256"]), "--certificate-issuance-sha256-commitment", str(issuance_reference["issuance_sha256"]), "--certificate-attempt-capability-path", str(capability_path)]
                child_env = os.environ.copy()
                child_env.pop("PYTHONPATH", None)
                child_env["PYTHONUTF8"] = "1"
                child_env["PYTHONDONTWRITEBYTECODE"] = "1"
                child_env["TORCH_HOME"] = str(dino_binding["torch_home"])
                completed = subprocess.run(command, cwd=str(V14_CANONICAL_PROJECT_ROOT), capture_output=True, check=False, env=child_env)
                returncode = completed.returncode
                stdout_bytes = bytes(completed.stdout or b"")
                stderr_bytes = bytes(completed.stderr or b"")
                stdout_path.write_bytes(stdout_bytes)
                stderr_path.write_bytes(stderr_bytes)
                capability_state = _certificate_capability_state(capability_path)
                attempt = {"attempt_id": attempt_id, "contract_id": contract["contract_id"], "model": model, "repeat_index": repeat_index, "attempt_ordinal": attempt_ordinal, "certificate_run_id": issuance_record["run_id"], "certificate_output_dir": str(output_dir), "fresh_process": True, "returncode": returncode, "status": "failed", "artifact": str(artifact_path), "arrays": str(arrays_path), "artifact_sha256": file_sha256(artifact_path), "arrays_sha256": file_sha256(arrays_path), "stdout": str(stdout_path), "stderr": str(stderr_path), "attempt": str(attempt_path), "stdout_sha256": file_sha256(stdout_path), "stderr_sha256": file_sha256(stderr_path), "certificate_issuance": dict(issuance_reference), "certificate_capability": capability_state, "started_event_sha256": ledger_previous}
                if returncode == 0 and artifact_path.is_file() and arrays_path.is_file():
                    try:
                        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                        entry = {"decision_id": f"v14_{model}_pusht_val_episode_007_offset_{int(contract['offset']):03d}_seed_{int(contract['seed']):02d}", "trajectory_id": "val_episode_007", "episode": 7, "offset": contract["offset"], "seed": contract["seed"], "model": model}
                        validation = validate_v14_artifact(artifact, entry, arrays_path=arrays_path, allow_excluded_engineering=True, expected_bindings={"checkpoint_sha256": specs[model]["checkpoint_sha256"], "config_sha256": specs[model]["config_sha256"], "official_git_commit": RELEASED_OFFICIAL_COMMIT, "official_root": str(official_root), "dataset_root": str(dataset_root), "dataset_split": "val", "dataset_archive_sha256": release_bindings["dataset_archive_sha256"], "seq_lengths_sha256": file_sha256(dataset_root / "val" / "seq_lengths.pkl"), "dataset_states_sha256": file_sha256(dataset_root / "val" / "states.pth"), "environment_seed": 1, "menu_seed": int(contract["seed"]), "runtime": runtime_binding, "hardware": hardware_binding, "runtime_sha256": canonical_sha256(runtime_binding), "hardware_sha256": canonical_sha256(hardware_binding), **_segment_binding_values(dataset_root, 7, int(contract["offset"])), "dinov2": dino_binding}, expected_certificate_issuance=issuance_reference)
                        if not validation:
                            signature = _artifact_signature(artifact, arrays_path, artifact_path)
                            attempt.update({"status": "passed", "signature": signature, "artifact_sha256": file_sha256(artifact_path), "arrays_sha256": file_sha256(arrays_path)})
                            repeat_records.append({"artifact": artifact, "arrays_path": arrays_path, "attempt": attempt})
                        else:
                            attempt["validation_errors"] = validation
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        attempt["validation_errors"] = [str(exc)]
                attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                ledger_previous = _append_certificate_ledger_event(
                    certificate_ledger,
                    {"attempt_id": attempt_id, "status": "completed" if attempt["status"] == "passed" else "failed", "ordinal": attempt_ordinal, "model": model, "contract_id": str(contract["contract_id"]), "repeat_index": repeat_index, "certificate_output_dir": str(output_dir), "certificate_issuance": dict(issuance_reference), "returncode": returncode, "artifact_exists": artifact_path.is_file(), "arrays_exists": arrays_path.is_file(), "artifact_sha256": attempt["artifact_sha256"], "arrays_sha256": attempt["arrays_sha256"], "stdout_sha256": attempt["stdout_sha256"], "stderr_sha256": attempt["stderr_sha256"], "attempt_path": str(attempt_path), "certificate_capability": _certificate_capability_state(capability_path)},
                    header=ledger_header, previous=ledger_previous,
                )
                attempts[model].append(attempt)
            comparison_errors = compare_v14_repeats(repeat_records) if len(repeat_records) == 3 else ["three valid fresh repeats are required"]
            results.append({"passed": len(repeat_records) == 3 and not comparison_errors, "repeat_count": len(repeat_records), "repeat0_signature": repeat_records[0].get("attempt", {}).get("signature") if repeat_records else None, "repeat_signatures": [record.get("attempt", {}).get("signature") for record in repeat_records], "attempts": [record.get("attempt", {}) for record in repeat_records], "comparison_errors": comparison_errors})
        certificates[model] = build_v14_certificate(model, specs[model], contracts, results, certificate_issuance=issuance_reference)
        (output_dir / f"{model}.certificate.json").write_text(json.dumps(certificates[model], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cross_errors: list[str] = []
    for contract in contracts:
        model_arrays = {}
        for model in V14_LOCKED_MODELS:
            matches = [item for item in attempts[model] if item.get("contract_id") == contract["contract_id"] and item.get("repeat_index") == 0 and item.get("status") == "passed"]
            if matches:
                with np.load(matches[0]["arrays"], allow_pickle=False) as archive:
                    model_arrays[model] = {key: np.asarray(archive[key]) for key in __import__("experiments.official_pusht_v14_common_initial", fromlist=["V14_CROSS_MODEL_ARRAY_KEYS"]).V14_CROSS_MODEL_ARRAY_KEYS}
        if len(model_arrays) == 2:
            left, right = (model_arrays[model] for model in V14_LOCKED_MODELS)
            for key in left:
                if not np.array_equal(left[key], right[key]):
                    cross_errors.append(f"{contract['contract_id']}: cross-model {key} mismatch")
            first_artifact = next(item["signature"] for item in attempts[V14_LOCKED_MODELS[0]] if item.get("contract_id") == contract["contract_id"] and item.get("repeat_index") == 0 and item.get("status") == "passed")
            second_artifact = next(item["signature"] for item in attempts[V14_LOCKED_MODELS[1]] if item.get("contract_id") == contract["contract_id"] and item.get("repeat_index") == 0 and item.get("status") == "passed")
            for key in ("official_git_commit", "official_root", "dataset_root", "dataset_split", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "states_segment_sha256", "actions_segment_sha256", "environment_seed", "initial_state_sha256", "goal_state_sha256", "initial_observation_sha256", "goal_observation_sha256", "runtime", "hardware", "runtime_sha256", "hardware_sha256", "dinov2"):
                if first_artifact.get("bindings", {}).get(key) != second_artifact.get("bindings", {}).get(key):
                    cross_errors.append(f"{contract['contract_id']}: cross-model binding {key} mismatch")
        else:
            cross_errors.append(f"{contract['contract_id']}: missing one or both cross-model production receipts")
    gate = {"passed": not cross_errors and all(cert.get("supervisor_status") == "passed" for cert in certificates.values()), "inventory": "common_initial_only", "models": list(V14_LOCKED_MODELS), "errors": cross_errors}
    for cert in certificates.values():
        cert["cross_model_common_initial_gate"] = gate
    for model in V14_LOCKED_MODELS:
        (output_dir / f"{model}.certificate.json").write_text(json.dumps(certificates[model], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_bindings = {"dataset_split": "val", "seq_lengths_sha256": file_sha256(dataset_root / "val" / "seq_lengths.pkl"), "dataset_states_sha256": file_sha256(dataset_root / "val" / "states.pth"), "dataset_archive_sha256": file_sha256(official_root / "pusht_noise.zip"), "dataset_files": release_bindings["dataset_files"]}
    certificate_ledger_reference = {"path": str(certificate_ledger), "sha256": file_sha256(certificate_ledger), "run_id": issuance_record["run_id"], "header_sha256": ledger_header["event_sha256"]}
    bundle = build_v14_bundle(certificates, dataset_root=dataset_root, official_root=official_root, contracts=contracts, cross_model_gate=gate, attempts=attempts, certificate_issuance=issuance_reference, certificate_attempt_ledger=certificate_ledger_reference, dataset_bindings=dataset_bindings, dinov2=dino_binding, launcher_attestation=launcher_attestation)
    (output_dir / "certificate_bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the immutable v1.4 JEPA/DINO Push-T certificate")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for model in ("jepa", "dino"):
        parser.add_argument(f"--{model}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{model}-config", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = {
        "official_jepa_wm": {"checkpoint": args.jepa_checkpoint, "config": args.jepa_config},
        "official_dino_wm": {"checkpoint": args.dino_checkpoint, "config": args.dino_config},
    }
    bundle = run_v14_supervisor(
        official_root=args.official_root, dataset_root=args.dataset_root,
        model_specs=specs, output_dir=args.output_dir,
    )
    print(json.dumps({"status": bundle.get("status"), "bundle_sha256": bundle.get("bundle_sha256")}, sort_keys=True))
    return 0 if bundle.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V14_CERTIFICATE_ISSUANCE_PATH", "V14_CERTIFICATE_LEDGER_PATH", "V14_CERTIFICATE_CAPABILITY_DIR", "V14_CERTIFICATE_LEDGER_RELATIVE_PATH", "V14_CERTIFICATE_CAPABILITY_RELATIVE_PATH", "validate_v14_certificate_issuance", "_consume_v14_certificate_issuance", "read_v14_certificate_ledger", "_certificate_capability_path", "_create_certificate_attempt_capability", "_certificate_capability_state", "make_v14_contracts", "build_v14_certificate", "validate_v14_certificate", "build_v14_bundle", "validate_v14_bundle", "run_v14_supervisor", "main"]
