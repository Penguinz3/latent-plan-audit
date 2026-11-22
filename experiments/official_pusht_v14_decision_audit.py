"""Run one clean v1.4 outcome-blind Push-T initial-menu audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

from experiments.official_pusht_v14_bootstrap import (
    V14_CANONICAL_PROJECT_ROOT, V14_SOURCE_MODULES, _consume_v14_launcher_attestation,
    validate_v14_consumed_launcher_attestation, v14_execution_preflight,
)


def _canonical_stdlib(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha256_stdlib(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: Any) -> str:
    import hashlib
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _digest_array(values: list[Any]) -> np.ndarray:
    import numpy as np
    return np.asarray([np.frombuffer(bytes.fromhex(_sha256_bytes(value)), dtype=np.uint8) for value in values], dtype=np.uint8)


def _json_value(value: Any) -> Any:
    import numpy as np
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_released_helpers():
    # These are model/replay primitives from the released implementation, not
    # a v1.2 runner or artifact path.
    from experiments.official_pusht_decision_audit import (
        _encode_replays_in_chunks, _git_commit, _hardware_fingerprint,
        _load_model, _native_return, _raw_observation, _replay_one,
        _runtime_fingerprint, _sha256_file,
    )
    return (_encode_replays_in_chunks, _git_commit, _hardware_fingerprint, _load_model, _native_return, _raw_observation, _replay_one, _runtime_fingerprint, _sha256_file)


def _consume_study_row_capability(
    capability_path: Path, *, official_root: Path, checkpoint: Path, config: Path,
    dataset_root: Path, dataset_split: str, episode: int, offset: int, seed: int,
    arrays_out: Path | None, out: Path | None, model_name: str | None,
    decision_id: str | None, trajectory_id: str | None, manifest_sha256: str | None,
    expected_parent_launcher_attestation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and consume the sole started-ledger authorization for a study row."""
    from experiments.official_pusht_v14_common_initial import canonical_sha256, file_sha256, v14_protocol_sha256, v14_source_bundle_sha256
    from experiments import official_pusht_v14_batch_study as batch

    if arrays_out is None or out is not None or not decision_id or not model_name or not trajectory_id or not manifest_sha256:
        raise ValueError("v1.4 study worker requires its complete batch-owned row identity and no direct JSON output")
    path = Path(capability_path).resolve()
    try:
        payload = path.read_bytes()
        record = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("v1.4 row capability cannot be read") from exc
    required = {
        "schema_version", "status", "one_shot", "study_issuance", "run_id",
        "run_commitment_sha256", "started_event_sha256", "ledger_path",
        "decision_id", "trajectory_id", "ordinal", "model", "episode", "offset",
        "seed", "manifest_sha256", "official_root", "dataset_root", "dataset_split",
        "checkpoint", "config", "artifact_path", "arrays_path", "protocol_sha256",
        "source_bundle_sha256", "capability_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("v1.4 row capability has an unexpected schema")
    if record.get("schema_version") != batch.V14_ROW_CAPABILITY_SCHEMA_VERSION or record.get("status") != "authorized" or record.get("one_shot") is not True:
        raise ValueError("v1.4 row capability is not an unused one-shot authorization")
    if record.get("capability_sha256") != canonical_sha256({key: value for key, value in record.items() if key != "capability_sha256"}):
        raise ValueError("v1.4 row capability hash mismatch")
    issuance_ref = record.get("study_issuance")
    if not isinstance(issuance_ref, Mapping) or set(issuance_ref) != {"path", "sha256", "issuance_sha256"}:
        raise ValueError("v1.4 row capability lacks its study issuance reference")
    issuance_path = Path(str(issuance_ref["path"])).resolve()
    if issuance_path != batch.V14_STUDY_ISSUANCE_PATH or file_sha256(issuance_path) != issuance_ref.get("sha256"):
        raise ValueError("v1.4 row capability study issuance file mismatch")
    try:
        issuance = json.loads(issuance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("v1.4 canonical study issuance cannot be read") from exc
    issuance_errors = batch.validate_v14_study_issuance(issuance)
    if issuance_errors or issuance.get("issuance_sha256") != issuance_ref.get("issuance_sha256"):
        raise ValueError("v1.4 row capability has an invalid study issuance")
    if issuance.get("launcher_attestation") != dict(expected_parent_launcher_attestation):
        raise RuntimeError("v1.4 study worker launcher attestation differs from study issuance")
    output_dir = Path(str(issuance["output_dir"])).resolve()
    expected_path = batch._row_capability_path(output_dir, int(record["ordinal"]), str(decision_id))
    expected = {
        "run_id": issuance.get("run_id"), "decision_id": decision_id,
        "trajectory_id": trajectory_id, "model": model_name, "episode": int(episode),
        "offset": int(offset), "seed": int(seed), "manifest_sha256": manifest_sha256,
        "official_root": str(Path(official_root).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()), "dataset_split": dataset_split,
        "checkpoint": str(Path(checkpoint).resolve()), "config": str(Path(config).resolve()),
        "arrays_path": str(Path(arrays_out).resolve()),
        "artifact_path": str((output_dir / f"{decision_id}.json").resolve()),
        "protocol_sha256": v14_protocol_sha256(), "source_bundle_sha256": v14_source_bundle_sha256(),
    }
    if path != expected_path or any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("v1.4 row capability does not match this scientific invocation")
    model_binding = issuance.get("models", {}).get(model_name)
    if not isinstance(model_binding, Mapping) or model_binding.get("checkpoint") != expected["checkpoint"] or model_binding.get("config") != expected["config"]:
        raise ValueError("v1.4 row capability model files differ from study issuance")
    ledger = Path(str(record["ledger_path"])).resolve()
    if ledger != batch.V14_STUDY_LEDGER_PATH or issuance.get("study_ledger_path") != str(batch.V14_STUDY_LEDGER_PATH):
        raise ValueError("v1.4 row capability ledger path mismatch")
    header, histories, ledger_errors, _tail = batch.read_v14_ledger(ledger)
    history = histories.get(str(decision_id), [])
    if ledger_errors or header is None or header.get("run_id") != issuance.get("run_id") or header.get("event_sha256") != record.get("run_commitment_sha256"):
        raise ValueError("v1.4 row capability ledger header is invalid")
    if [item.get("status") for item in history] != ["started"] or history[0].get("event_sha256") != record.get("started_event_sha256"):
        raise ValueError("v1.4 row capability is not backed by exactly one live started receipt")
    for key, value in (("ordinal", int(record["ordinal"])), ("model", model_name), ("episode", int(episode)), ("offset", int(offset)), ("menu_seed", int(seed)), ("artifact_path", expected["artifact_path"]), ("arrays_path", expected["arrays_path"])):
        if history[0].get(key) != value:
            raise ValueError(f"v1.4 started receipt {key} differs from row capability")
    consumed = batch._consumed_row_capability_path(path)
    consumed.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("v1.4 row capability was already consumed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    path.unlink()
    reference = {
        "status": "consumed", "path": str(consumed.resolve()),
        "sha256": file_sha256(consumed), "capability_sha256": record["capability_sha256"],
    }
    return dict(issuance_ref), reference


def _consume_certificate_attempt_capability(
    capability_path: Path, *, issuance_path: Path, issuance_file_sha256: str,
    issuance_sha256: str, official_root: Path, checkpoint: Path, config: Path,
    dataset_root: Path, episode: int, offset: int, seed: int,
    model_name: str | None, out: Path | None, arrays_out: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consume exactly one external certificate-attempt capability."""
    from experiments import official_pusht_v14_certificate_supervisor as supervisor

    path = Path(capability_path).resolve()
    consumed = supervisor._consumed_certificate_capability_path(path)
    if consumed.is_file():
        raise FileExistsError("v1.4 certificate attempt capability was already consumed")
    try:
        payload = path.read_bytes()
        record = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("v1.4 certificate attempt capability cannot be read") from exc
    required = {
        "schema_version", "status", "one_shot", "certificate_issuance", "run_id",
        "run_commitment_sha256", "started_event_sha256", "ledger_path", "attempt_id",
        "ordinal", "model", "contract_id", "repeat_index", "certificate_output_dir",
        "artifact_path", "arrays_path", "stdout_path", "stderr_path", "attempt_path",
        "protocol_sha256", "source_bundle_sha256", "capability_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("v1.4 certificate attempt capability has an unexpected schema")
    if record.get("schema_version") != supervisor.V14_CERTIFICATE_CAPABILITY_SCHEMA_VERSION or record.get("status") != "authorized" or record.get("one_shot") is not True:
        raise ValueError("v1.4 certificate attempt capability is not an unused one-shot authorization")
    if record.get("capability_sha256") != _canonical_stdlib({key: value for key, value in record.items() if key != "capability_sha256"}):
        raise ValueError("v1.4 certificate attempt capability hash mismatch")
    issuance_ref = record.get("certificate_issuance")
    expected_issuance = {"path": str(Path(issuance_path).resolve()), "sha256": issuance_file_sha256, "issuance_sha256": issuance_sha256}
    if issuance_ref != expected_issuance:
        raise ValueError("v1.4 certificate attempt capability issuance mismatch")
    expected_issuance_path = supervisor._certificate_issuance_path()
    expected_ledger = supervisor._certificate_ledger_path()
    if Path(str(issuance_ref["path"])).resolve() != expected_issuance_path or record.get("ledger_path") != str(expected_ledger):
        raise ValueError("v1.4 certificate worker canonical path binding mismatch")
    try:
        issuance = json.loads(Path(str(issuance_ref["path"])).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("v1.4 certificate worker issuance cannot be read") from exc
    if not isinstance(issuance, Mapping) or issuance.get("status") != "consumed" or issuance.get("one_shot") is not True or issuance.get("run_id") != record.get("run_id") or issuance.get("certificate_attempt_ledger_path") != str(expected_ledger):
        raise ValueError("v1.4 certificate worker issuance is invalid")
    if supervisor.validate_v14_certificate_issuance(issuance):
        raise ValueError("v1.4 certificate worker issuance is invalid")
    if issuance.get("issuance_sha256") != issuance_sha256 or _file_sha256_stdlib(Path(str(issuance_ref["path"]))) != issuance_file_sha256:
        raise ValueError("v1.4 certificate worker issuance hash mismatch")
    data = issuance.get("official_data_bindings", {})
    models = issuance.get("model_bindings", {})
    model = models.get(model_name) if isinstance(models, Mapping) else None
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("v1.4 certificate worker model/data authorization is incomplete")
    if data.get("official_root") != str(Path(official_root).resolve()) or data.get("dataset_root") != str(Path(dataset_root).resolve()):
        raise ValueError("v1.4 certificate worker roots differ from issuance")
    if model.get("checkpoint_sha256") != _file_sha256_stdlib(Path(checkpoint)) or model.get("config_sha256") != _file_sha256_stdlib(Path(config)):
        raise ValueError("v1.4 certificate worker model files differ from issuance")
    expected_artifact = str(Path(str(out)).resolve()) if out is not None else None
    expected_arrays = str(Path(str(arrays_out)).resolve()) if arrays_out is not None else None
    if record.get("model") != model_name or record.get("artifact_path") != expected_artifact or record.get("arrays_path") != expected_arrays:
        raise ValueError("v1.4 certificate attempt capability does not match this invocation")
    try:
        contract = next(item for item in issuance["certificate_schedule"]["contracts"] if str(item.get("contract_id")) == str(record["contract_id"]))
    except (ValueError, TypeError, KeyError, StopIteration) as exc:
        raise ValueError("v1.4 certificate attempt capability contract cannot be verified") from exc
    if not (int(contract.get("episode", -1)) == int(episode) == 7 and int(contract.get("offset", -1)) == int(offset) and int(contract.get("seed", -1)) == int(seed)):
        raise ValueError("v1.4 certificate attempt capability contract mismatch")
    models_in_order = tuple(supervisor.V14_LOCKED_MODELS)
    contracts_in_order = tuple(issuance.get("certificate_schedule", {}).get("contracts", ()))
    try:
        expected_ordinal = models_in_order.index(str(model_name)) * 9 + contracts_in_order.index(contract) * 3 + int(record["repeat_index"])
    except (ValueError, TypeError):
        raise ValueError("v1.4 certificate attempt capability ordinal is not canonical")
    expected_capability_path = supervisor._certificate_capability_path(str(record["run_id"]), expected_ordinal)
    if int(record["ordinal"]) != expected_ordinal or path != expected_capability_path:
        raise ValueError("v1.4 certificate attempt capability path/ordinal is not canonical")
    records = []
    try:
        records = [json.loads(line) for line in Path(str(record["ledger_path"])).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("v1.4 certificate attempt ledger cannot be read") from exc
    if not records or not isinstance(records[0], Mapping) or records[0].get("status") != "run_header":
        raise ValueError("v1.4 certificate attempt ledger lacks its run header")
    header = records[0]
    histories: dict[str, list[Mapping[str, Any]]] = {}
    previous = None
    ledger_errors: list[str] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping) or item.get("previous_event_sha256") != previous or item.get("event_sha256") != _canonical_stdlib({key: value for key, value in item.items() if key != "event_sha256"}):
            ledger_errors.append(f"invalid certificate ledger event {index}")
        if index and (not isinstance(item, Mapping) or item.get("run_id") != header.get("run_id") or item.get("run_commitment_sha256") != header.get("event_sha256")):
            ledger_errors.append(f"invalid certificate ledger event {index} commitment")
        previous = item.get("event_sha256")
        if index:
            histories.setdefault(str(item.get("attempt_id")), []).append(item)
    if header.get("run_id") != record.get("run_id") or header.get("event_sha256") != record.get("run_commitment_sha256") or header.get("certificate_issuance") != expected_issuance or header.get("certificate_output_dir") != record.get("certificate_output_dir"):
        ledger_errors.append("certificate ledger header commitment mismatch")
    if ledger_errors:
        raise ValueError("v1.4 certificate attempt capability ledger is invalid")
    history = histories.get(str(record["attempt_id"]), [])
    if [item.get("status") for item in history] != ["started"] or history[0].get("event_sha256") != record.get("started_event_sha256"):
        raise ValueError("v1.4 certificate attempt capability is not backed by exactly one live started event")
    started = history[0]
    for key in ("attempt_id", "ordinal", "model", "contract_id", "repeat_index", "certificate_output_dir", "artifact_path", "arrays_path", "stdout_path", "stderr_path", "attempt_path", "capability_path"):
        if key == "capability_path":
            if started.get(key) != str(expected_capability_path):
                raise ValueError(f"v1.4 certificate started event {key} differs from capability")
            continue
        if started.get(key) != record.get(key):
            raise ValueError(f"v1.4 certificate started event {key} differs from capability")
    consumed_payload = {"status": "consumed", "one_shot": True, "authorized_path": str(path), "capability_sha256": record["capability_sha256"], "started_event_sha256": record["started_event_sha256"]}
    consumed.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("v1.4 certificate attempt capability was already consumed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write((json.dumps(consumed_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        handle.flush(); os.fsync(handle.fileno())
    path.unlink()
    reference = {"status": "consumed", "path": str(consumed), "sha256": _file_sha256_stdlib(consumed), "capability_sha256": record["capability_sha256"]}
    return expected_issuance, reference


def _require_module_origin(name: str, roots: tuple[Path, ...]) -> None:
    spec = importlib.util.find_spec(name)
    candidates = [] if spec is None else list(spec.submodule_search_locations or ())
    if spec is not None and spec.origin not in (None, "built-in", "frozen"):
        candidates.append(spec.origin)
    if not candidates or any(not any(Path(value).resolve() == root or root in Path(value).resolve().parents for root in roots) for value in candidates):
        raise RuntimeError(f"v1.4 import origin for {name} is missing or outside its authorized root")


def run_one_decision(
    *, official_root: Path, checkpoint: Path, config: Path, dataset_root: Path,
    dataset_split: str = "val", episode: int = 0, offset: int = 0, seed: int = 1,
    arrays_out: Path | None = None, out: Path | None = None, repeat_count: int = 1,
    planning_repeats: int = 0, inventory_mode: str = "common_initial_only",
    model_name: str | None = None, decision_id: str | None = None,
    trajectory_id: str | None = None, manifest_sha256: str | None = None,
    dinov2_source_root: Path | None = None, dinov2_source_archive: Path | None = None,
    dinov2_weights: Path | None = None, torch_home: Path | None = None,
    certificate_issuance_path: Path | None = None, certificate_issuance_sha256: str | None = None,
    certificate_issuance_sha256_commitment: str | None = None,
    certificate_attempt_capability_path: Path | None = None,
    row_capability_path: Path | None = None,
    parent_launcher_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # This is intentionally the first executable operation.  In particular,
    # the released helper module and optional model dependencies are imported
    # only after bad planning arguments have been rejected.
    if isinstance(planning_repeats, bool) or not isinstance(planning_repeats, Integral) or int(planning_repeats) < 0 or 0 < int(planning_repeats) < 3:
        raise ValueError("planning_repeats must be zero or at least three")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, Integral) or int(repeat_count) < 1:
        raise ValueError("repeat_count must be positive")
    issuance_values = (certificate_issuance_path, certificate_issuance_sha256, certificate_issuance_sha256_commitment)
    if any(value is not None for value in issuance_values) and not all(value is not None for value in issuance_values):
        raise ValueError("certificate issuance binding requires path, file hash, and issuance hash")
    certificate_mode = all(value is not None for value in issuance_values)
    if certificate_mode == (row_capability_path is not None):
        raise ValueError("v1.4 worker requires exactly one certificate or study-row authorization")
    if certificate_mode != (certificate_attempt_capability_path is not None):
        raise ValueError("v1.4 certificate worker requires exactly one attempt capability")
    if inventory_mode != "common_initial_only":
        raise ValueError("v1.4 supports only common_initial_only")
    if dataset_split != "val":
        raise ValueError("v1.4 confirmatory audit requires dataset_split=val")
    if seed not in (1, 2, 3):
        raise ValueError("v1.4 menu seed must be one of (1, 2, 3)")
    project_root = V14_CANONICAL_PROJECT_ROOT
    official = Path(official_root).resolve()
    if str(official) not in sys.path:
        sys.path.insert(0, str(official))
    preflight_errors = v14_execution_preflight(
        official_root, project_root=project_root,
        source_modules=V14_SOURCE_MODULES,
    )
    if preflight_errors:
        raise RuntimeError("unsafe v1.4 worker Python environment: " + "; ".join(preflight_errors))
    if certificate_mode:
        _consume_v14_launcher_attestation("experiments.official_pusht_v14_decision_audit")
    else:
        parent_errors = validate_v14_consumed_launcher_attestation(
            parent_launcher_attestation,
            "experiments.official_pusht_v14_batch_study",
        )
        if parent_errors:
            raise RuntimeError("v1.4 study worker parent launcher attestation is invalid: " + "; ".join(parent_errors))
    study_issuance = row_capability = None
    certificate_attempt_capability = None
    if certificate_mode:
        _certificate_issuance, certificate_attempt_capability = _consume_certificate_attempt_capability(
            Path(certificate_attempt_capability_path),
            issuance_path=Path(certificate_issuance_path),
            issuance_file_sha256=str(certificate_issuance_sha256),
            issuance_sha256=str(certificate_issuance_sha256_commitment),
            official_root=official_root, checkpoint=checkpoint, config=config,
            dataset_root=dataset_root, episode=episode, offset=offset, seed=seed,
            model_name=model_name, out=out, arrays_out=arrays_out,
        )
    else:
        study_issuance, row_capability = _consume_study_row_capability(
            row_capability_path, official_root=official_root, checkpoint=checkpoint, config=config,
            dataset_root=dataset_root, dataset_split=dataset_split, episode=episode, offset=offset,
            seed=seed, arrays_out=arrays_out, out=out, model_name=model_name,
            decision_id=decision_id, trajectory_id=trajectory_id, manifest_sha256=manifest_sha256,
            expected_parent_launcher_attestation=parent_launcher_attestation,
        )
    from experiments.official_pusht_v14_common_initial import (
        V14_LOCKED_SEEDS, V14_PROTOCOL, V14_SCORE_TOLERANCE, build_v14_artifact,
        capture_initial_only_cem, expand_model_actions, make_reencoding_diagnostic,
        make_production_pass_receipt, proposal_support_diagnostics,
        validate_p_o_arrays, canonical_sha256, file_sha256, v14_source_bundle_sha256,
        v14_protocol_sha256,
    )
    import numpy as np
    import torch
    if not sys.flags.isolated or not sys.flags.no_user_site:
        raise RuntimeError("v1.4 worker requires isolated Python with user site disabled")
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("v1.4 worker rejects ambient PYTHONPATH shadowing")
    sys.dont_write_bytecode = True
    environment_roots = (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    _require_module_origin("yaml", environment_roots)
    _require_module_origin("omegaconf", environment_roots)
    _require_module_origin("app", (official,))
    _require_module_origin("evals", (official,))
    (_encode, _git_commit, _hardware, _load_model, _native_return, _raw_observation, _replay_one, _runtime, _sha256_file) = _load_released_helpers()
    from omegaconf import OmegaConf
    from app.plan_common.datasets.pusht_dset import PushTDataset
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, AttributeError) as exc:
        raise RuntimeError("PyTorch deterministic algorithms could not be enabled") from exc

    raw_cfg, stats, model, preprocessor, dinov2_dependency = _load_model(
        Path(official_root), Path(checkpoint), Path(config),
        dinov2_source_root=dinov2_source_root, dinov2_source_archive=dinov2_source_archive,
        dinov2_weights=dinov2_weights, torch_home=torch_home,
    )
    dataset = PushTDataset(data_path=str(Path(dataset_root) / dataset_split), normalize_action=True, with_velocity=True)
    frameskip = int(raw_cfg["model_kwargs"]["data"]["custom"]["frameskip"])
    horizon = int(raw_cfg["planner"]["horizon"])
    planner_cfg, objective_cfg = raw_cfg["planner"], raw_cfg["planner"]["planning_objective"]
    task_cfg, custom_cfg = raw_cfg["task_specification"], raw_cfg["model_kwargs"]["data"]["custom"]
    pinned = {
        "planner_name": planner_cfg.get("planner_name") == "cem", "objective_type": objective_cfg.get("objective_type") == "L2",
        "sum_all_diffs": objective_cfg.get("sum_all_diffs") is False, "alpha": math.isclose(float(objective_cfg.get("alpha")), 0.1, abs_tol=1e-12),
        "horizon": horizon == 6, "num_act_stepped": planner_cfg.get("num_act_stepped") == 6, "repeat_actskip": planner_cfg.get("repeat_actskip") is False,
        "frameskip": frameskip == 5, "action_skip": custom_cfg.get("action_skip") == 1, "num_frames": task_cfg.get("num_frames") == 1,
        "num_proprios": task_cfg.get("num_proprios") == 1, "distribute_planner": planner_cfg.get("distribute_planner") is False,
        "iterations": planner_cfg.get("iterations") == 30, "num_samples": planner_cfg.get("num_samples") == 300,
        "num_elites": planner_cfg.get("num_elites") == 10, "var_scale": math.isclose(float(planner_cfg.get("var_scale")), 1.0, abs_tol=1e-12),
    }
    if not all(pinned.values()):
        raise ValueError(f"config is not the pinned released Push-T configuration: {pinned}")
    for key in ("max_norms", "max_norm_dims", "momentum_mean", "momentum_std"):
        if planner_cfg.get(key) not in (None, 0, 0.0, [], [[]]):
            raise ValueError(f"unsupported non-default CEM option: {key}")
    needed = frameskip * horizon + 1
    if not 0 <= int(episode) < len(dataset) or int(offset) + needed > dataset.get_seq_length(int(episode)):
        raise ValueError("selected validation segment is outside the dataset")
    states = dataset.states[int(episode)].numpy()
    actions = dataset.actions[int(episode)]
    init_state = np.asarray(states[int(offset)], dtype=np.float64)
    expert_normalized = actions[int(offset):int(offset) + frameskip * horizon]
    expert_actions = preprocessor.denormalize_actions(expert_normalized)
    shape = dataset.shapes[int(episode)]
    env_cfg = OmegaConf.create(raw_cfg)
    env_cfg.frameskip, env_cfg.tasks = frameskip, ["pusht-base"]
    env_cfg.task_specification.multitask = False
    env = make_env(env_cfg)
    env.update_env({"shape": shape})
    env_info = {"shape": shape}
    environment_seed = 1
    expert_rollout = _replay_one(env, environment_seed, init_state, env_info, expert_actions)
    goal_state = expert_rollout["state"]
    goal_raw = _raw_observation(expert_rollout["obs"], expert_rollout["info"])
    init_obs, init_info = env.prepare(environment_seed, init_state, env_info=env_info)
    init_raw = _raw_observation(init_obs, init_info)
    with torch.inference_mode():
        z_init, z_goal = model.encode(init_raw), model.encode(goal_raw)
    objective = ReprTargetDistMPCObjective(raw_cfg, target_enc=z_goal, sum_all_diffs=False, alpha=float(objective_cfg["alpha"]))
    captured = capture_initial_only_cem(model.unroll, objective, z_init, num_samples=300, horizon=6, action_dim=int(model.action_dim), var_scale=1.0, seed=int(seed), batch_size=8, check_batch_size=4, equivalence_tolerance=V14_SCORE_TOLERANCE)
    planning_gate: dict[str, Any] = {"status": "external_certificate_required", "passed": False, "repeat_count": 0}
    if planning_repeats:
        captures = [captured] + [capture_initial_only_cem(model.unroll, objective, z_init, num_samples=300, horizon=6, action_dim=int(model.action_dim), var_scale=1.0, seed=int(seed), batch_size=8, check_batch_size=4, equivalence_tolerance=V14_SCORE_TOLERANCE) for _ in range(int(planning_repeats) - 1)]
        inventories = [_sha256_bytes(item["initial_samples"].permute(1, 0, 2).cpu().numpy()) for item in captures]
        scores = [_sha256_bytes(item["initial_scores"].cpu().numpy()) for item in captures]
        planning_gate = {"status": "passed" if len(set(inventories)) == 1 and len(set(scores)) == 1 else "failed", "passed": len(set(inventories)) == 1 and len(set(scores)) == 1, "repeat_count": int(planning_repeats), "initial_inventory_sha256": inventories, "source_scores_sha256": scores}
        if not planning_gate["passed"]:
            raise RuntimeError("v1.4 repeated initial-only planning outputs were not identical")
    initial_model = captured["initial_samples"].permute(1, 0, 2).cpu()
    if int(model.action_dim) != 10:
        raise ValueError("v1.4 protocol requires model action width 10")
    initial_physical = preprocessor.denormalize_actions(expand_model_actions(initial_model, frameskip=frameskip))
    initial_replays = []
    for candidate_actions in initial_physical:
        replay = _replay_one(env, environment_seed, init_state, env_info, candidate_actions)
        replay.update(_native_return(env, goal_state, replay["state"], replay))
        initial_replays.append(replay)
    raw_terminal_visual = np.asarray([item["obs"] for item in initial_replays], dtype=np.uint8)
    raw_terminal_proprio = np.asarray([item["info"]["proprio"] for item in initial_replays], dtype=np.float32)
    raw_initial_visual, raw_initial_proprio = np.asarray(init_obs, dtype=np.uint8), np.asarray(init_info["proprio"], dtype=np.float32)
    raw_goal_visual, raw_goal_proprio = np.asarray(expert_rollout["obs"], dtype=np.uint8), np.asarray(expert_rollout["info"]["proprio"], dtype=np.float32)
    production_real = _encode(model, initial_replays, chunk_size=8)
    production_pred_v = captured["initial_predicted"]["visual"].float().cpu().numpy()
    production_pred_p = captured["initial_predicted"]["proprio"].float().cpu().numpy()
    production_real_v, production_real_p = production_real["visual"].numpy(), production_real["proprio"].numpy()
    production_goal_v, production_goal_p = z_goal["visual"][:, -1].float().cpu().numpy(), z_goal["proprio"][:, -1].float().cpu().numpy()
    production_initial_v, production_initial_p = z_init["visual"][:, -1].float().cpu().numpy(), z_init["proprio"][:, -1].float().cpu().numpy()
    production_decomp = validate_p_o_arrays(production_pred_v, production_real_v, production_goal_v, predicted_proprio=production_pred_p, realized_proprio=production_real_p, goal_proprio=production_goal_p)
    production_p, production_o = production_decomp["P"], production_decomp["O"]
    source_scores = captured["initial_scores"].cpu().numpy().astype(np.float64)
    native = np.asarray([item["native_return"] for item in initial_replays], dtype=np.float64)
    distances = np.asarray([item["native_state_distance"] for item in initial_replays], dtype=np.float64)
    if float(np.max(np.abs(production_p - source_scores))) > V14_SCORE_TOLERANCE:
        raise RuntimeError("v1.4 captured scores do not reproduce authoritative P")
    # Commit the production pass before any serialized-observation diagnostic
    # encoding.  These hashes are the only arrays permitted to drive P/O/G/J.
    production_receipt = make_production_pass_receipt(
        initial_actions_normalized=initial_model.numpy(), initial_physical_actions=initial_physical.numpy(),
        initial_predicted_visual=production_pred_v, initial_encoded_real_visual=production_real_v,
        goal_visual=production_goal_v, initial_predicted_proprio=production_pred_p,
        initial_encoded_real_proprio=production_real_p, goal_proprio=production_goal_p,
        initial_objective_scores=source_scores, initial_native_returns=native,
        initial_native_state_distances=distances,
        initial_final_states=np.asarray([item["state"] for item in initial_replays], dtype=np.float64),
        goal_state=np.asarray(goal_state, dtype=np.float64),
        initial_reward_traces=np.asarray([item["reward_trace"] for item in initial_replays], dtype=np.float32),
        initial_raw_visual_observations=raw_terminal_visual, initial_raw_proprio_observations=raw_terminal_proprio,
        raw_initial_visual=raw_initial_visual, raw_initial_proprio=raw_initial_proprio,
        raw_goal_visual=raw_goal_visual, raw_goal_proprio=raw_goal_proprio,
        initial_production_visual=production_initial_v, initial_production_proprio=production_initial_p,
    )
    retained = [{"obs": raw_terminal_visual[index], "info": {"proprio": raw_terminal_proprio[index]}} for index in range(300)]
    chunk8 = _encode(model, retained, chunk_size=8)
    chunk4 = _encode(model, retained, chunk_size=4)
    initial8 = _encode(model, [{"obs": raw_initial_visual, "info": {"proprio": raw_initial_proprio}}], chunk_size=8)
    initial4 = _encode(model, [{"obs": raw_initial_visual, "info": {"proprio": raw_initial_proprio}}], chunk_size=4)
    goal8 = _encode(model, [{"obs": raw_goal_visual, "info": {"proprio": raw_goal_proprio}}], chunk_size=8)
    goal4 = _encode(model, [{"obs": raw_goal_visual, "info": {"proprio": raw_goal_proprio}}], chunk_size=4)
    diagnostic = make_reencoding_diagnostic(
        production_predicted_visual=production_pred_v, production_predicted_proprio=production_pred_p,
        production_realized_visual=production_real_v, production_realized_proprio=production_real_p,
        production_goal_visual=production_goal_v, production_goal_proprio=production_goal_p,
        chunk8_realized_visual=chunk8["visual"].numpy(), chunk8_realized_proprio=chunk8["proprio"].numpy(),
        chunk8_goal_visual=goal8["visual"].numpy()[:, -1] if goal8["visual"].ndim == 3 else goal8["visual"].numpy(),
        chunk8_goal_proprio=goal8["proprio"].numpy()[:, -1] if goal8["proprio"].ndim == 3 else goal8["proprio"].numpy(),
        chunk4_realized_visual=chunk4["visual"].numpy(), chunk4_realized_proprio=chunk4["proprio"].numpy(),
        chunk4_goal_visual=goal4["visual"].numpy()[:, -1] if goal4["visual"].ndim == 3 else goal4["visual"].numpy(),
        chunk4_goal_proprio=goal4["proprio"].numpy()[:, -1] if goal4["proprio"].ndim == 3 else goal4["proprio"].numpy(),
        production_p=production_p, production_o=production_o,
        production_initial_visual=production_initial_v, production_initial_proprio=production_initial_p,
        chunk8_initial_visual=initial8["visual"].numpy()[:, -1] if initial8["visual"].ndim == 3 else initial8["visual"].numpy(),
        chunk8_initial_proprio=initial8["proprio"].numpy()[:, -1] if initial8["proprio"].ndim == 3 else initial8["proprio"].numpy(),
        chunk4_initial_visual=initial4["visual"].numpy()[:, -1] if initial4["visual"].ndim == 3 else initial4["visual"].numpy(),
        chunk4_initial_proprio=initial4["proprio"].numpy()[:, -1] if initial4["proprio"].ndim == 3 else initial4["proprio"].numpy(),
        terminal_visual_sha256=_sha256_bytes(raw_terminal_visual), terminal_proprio_sha256=_sha256_bytes(raw_terminal_proprio),
        goal_visual_sha256=_sha256_bytes(raw_goal_visual), goal_proprio_sha256=_sha256_bytes(raw_goal_proprio),
        initial_visual_sha256=_sha256_bytes(raw_initial_visual), initial_proprio_sha256=_sha256_bytes(raw_initial_proprio),
        realized_visual_sha256=_sha256_bytes(production_real_v), realized_proprio_sha256=_sha256_bytes(production_real_p),
        encoded_goal_visual_sha256=_sha256_bytes(production_goal_v), encoded_goal_proprio_sha256=_sha256_bytes(production_goal_p),
    )
    expert_outcome = _native_return(env, goal_state, goal_state, expert_rollout)
    underlying_env = getattr(env, "unwrapped", env)
    action_scale = float(getattr(underlying_env, "action_scale", 100.0))
    relative = bool(getattr(underlying_env, "relative", True))
    noop_target = np.zeros(2, dtype=np.float32) if relative else init_state[:2] / action_scale
    noop_normalized = (torch.as_tensor(noop_target, dtype=torch.float32) - preprocessor.action_mean) / preprocessor.action_std
    noop_physical = preprocessor.denormalize_actions(noop_normalized.repeat(frameskip * horizon, 1))
    noop_replay = _replay_one(env, environment_seed, init_state, env_info, noop_physical)
    noop_replay.update(_native_return(env, goal_state, noop_replay["state"], noop_replay))
    support = proposal_support_diagnostics(initial_model.numpy(), initial_physical.numpy(), released_val_actions=np.asarray(expert_normalized.numpy(), dtype=np.float32).reshape(1, -1))
    baselines = {
        "expert_goal_defining_segment": {"J": expert_outcome["native_return"], "G": -expert_outcome["native_state_distance"], "native_return": expert_outcome["native_return"], "native_state_distance": expert_outcome["native_state_distance"]},
        "hold_position_noop": {"J": noop_replay["native_return"], "G": -noop_replay["native_state_distance"], "native_return": noop_replay["native_return"], "native_state_distance": noop_replay["native_state_distance"], "physical_action_sha256": _sha256_bytes(noop_physical.numpy())},
        "repeatability": {"passed": False, "repeat_count": 1, "state_sha256": [_sha256_bytes(np.asarray(initial_replays[0]["state"]))], "terminal_observation_sha256": [_sha256_bytes(raw_terminal_visual[0])], "cumulative_reward": [float(native[0])]},
    }
    # Uniform is the exact menu expectation; it is not replaced by a lucky
    # sampled candidate in v1.4.
    baselines["random"] = {"selection": "exact uniform expectation", "candidate_id": None, "candidate_index": None, "J": float(np.mean(native)), "G": float(np.mean(-distances))}
    model_label = model_name or ("official_dino_wm" if "dino" in (Path(checkpoint).name + Path(config).name).lower() else "official_jepa_wm")
    runtime_fingerprint, hardware_fingerprint = _runtime(), _hardware()
    artifact = build_v14_artifact(
        model=model_label, decision_id=decision_id or f"v14_pusht_val_episode_{int(episode):03d}_offset_{int(offset):03d}_seed_{int(seed):02d}", episode=int(episode), offset=int(offset), seed=int(seed),
        normalized_actions=initial_model.numpy(), physical_actions=initial_physical.numpy(), predicted_visual=production_pred_v, realized_visual=production_real_v, goal_visual=production_goal_v,
        predicted_proprio=production_pred_p, realized_proprio=production_real_p, goal_proprio=production_goal_p, objective_scores=source_scores, native_returns=native, native_state_distances=distances,
        final_states=np.asarray([item["state"] for item in initial_replays], dtype=np.float64), goal_state=np.asarray(goal_state, dtype=np.float64), reward_traces=np.asarray([item["reward_trace"] for item in initial_replays], dtype=np.float32),
        expert_reward_trace=expert_rollout["reward_trace"], noop_physical_actions=noop_physical.numpy(), noop_final_state=noop_replay["state"], noop_reward_trace=noop_replay["reward_trace"], raw_visual_observations=raw_terminal_visual, raw_proprio_observations=raw_terminal_proprio,
        raw_initial_visual=raw_initial_visual, raw_initial_proprio=raw_initial_proprio, raw_goal_visual=raw_goal_visual, raw_goal_proprio=raw_goal_proprio, raw_reencoding_gate=diagnostic, raw_visual_digests=_digest_array([item["obs"] for item in initial_replays]), raw_proprio_digests=_digest_array([item["info"]["proprio"] for item in initial_replays]),
        trajectory_id=trajectory_id, arrays_out=arrays_out, proposal_diagnostics=support, baselines=baselines, planning_gate=planning_gate,
         production_pass_receipt=production_receipt,
         certificate_issuance=(
             {"path": str(Path(certificate_issuance_path).resolve()),
              "sha256": str(certificate_issuance_sha256),
              "issuance_sha256": str(certificate_issuance_sha256_commitment)}
             if certificate_issuance_path is not None and certificate_issuance_sha256 is not None and certificate_issuance_sha256_commitment is not None
             else None
         ),
         certificate_attempt_capability=certificate_attempt_capability,
         study_issuance=study_issuance,
         row_capability=row_capability,
         bindings={"runtime": runtime_fingerprint, "hardware": hardware_fingerprint, "runtime_sha256": canonical_sha256(runtime_fingerprint), "hardware_sha256": canonical_sha256(hardware_fingerprint), "official_git_commit": _git_commit(Path(official_root)), "official_root": str(Path(official_root).resolve()), "checkpoint_sha256": _sha256_file(Path(checkpoint)), "config_sha256": _sha256_file(Path(config)), "dataset_root": str(Path(dataset_root).resolve()), "dataset_split": dataset_split, "dataset_archive_sha256": _sha256_file(Path(official_root) / "pusht_noise.zip"), "seq_lengths_sha256": _sha256_file(Path(dataset_root) / dataset_split / "seq_lengths.pkl"), "dataset_states_sha256": _sha256_file(Path(dataset_root) / dataset_split / "states.pth"), "manifest_sha256": manifest_sha256, "environment_seed": environment_seed, "initial_state_sha256": _sha256_bytes(init_state), "goal_state_sha256": _sha256_bytes(goal_state), "initial_observation_sha256": canonical_sha256({"visual": _sha256_bytes(raw_initial_visual), "proprio": _sha256_bytes(raw_initial_proprio)}), "goal_observation_sha256": canonical_sha256({"visual": _sha256_bytes(raw_goal_visual), "proprio": _sha256_bytes(raw_goal_proprio)}), "states_segment_sha256": _sha256_bytes(np.asarray(states[int(offset):int(offset) + needed])), "actions_segment_sha256": _sha256_bytes(np.asarray(expert_normalized)), "dinov2": dinov2_dependency, "menu_seed": int(seed)},
         diagnostic_arrays={"initial_production_visual": production_initial_v, "initial_production_proprio": production_initial_p, "chunk8_initial_visual": initial8["visual"].numpy()[:, -1] if initial8["visual"].ndim == 3 else initial8["visual"].numpy(), "chunk8_initial_proprio": initial8["proprio"].numpy()[:, -1] if initial8["proprio"].ndim == 3 else initial8["proprio"].numpy(), "chunk4_initial_visual": initial4["visual"].numpy()[:, -1] if initial4["visual"].ndim == 3 else initial4["visual"].numpy(), "chunk4_initial_proprio": initial4["proprio"].numpy()[:, -1] if initial4["proprio"].ndim == 3 else initial4["proprio"].numpy(), "chunk8_realized_visual": chunk8["visual"].numpy(), "chunk8_realized_proprio": chunk8["proprio"].numpy(), "chunk8_goal_visual": goal8["visual"].numpy()[:, -1] if goal8["visual"].ndim == 3 else goal8["visual"].numpy(), "chunk8_goal_proprio": goal8["proprio"].numpy()[:, -1] if goal8["proprio"].ndim == 3 else goal8["proprio"].numpy(), "chunk4_realized_visual": chunk4["visual"].numpy(), "chunk4_realized_proprio": chunk4["proprio"].numpy(), "chunk4_goal_visual": goal4["visual"].numpy()[:, -1] if goal4["visual"].ndim == 3 else goal4["visual"].numpy(), "chunk4_goal_proprio": goal4["proprio"].numpy()[:, -1] if goal4["proprio"].ndim == 3 else goal4["proprio"].numpy()},
    )
    if out is not None:
        output = Path(out).resolve(); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(_json_value(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return _json_value(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-split", default="val"); parser.add_argument("--episode", type=int, default=0); parser.add_argument("--offset", type=int, default=0); parser.add_argument("--seed", type=int, default=1); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--arrays-out", type=Path, required=True); parser.add_argument("--repeat-count", type=int, default=1); parser.add_argument("--planning-repeats", type=int, default=0); parser.add_argument("--inventory-mode", default="common_initial_only"); parser.add_argument("--model-name"); parser.add_argument("--decision-id"); parser.add_argument("--trajectory-id"); parser.add_argument("--manifest-sha256"); parser.add_argument("--dinov2-source-root", type=Path); parser.add_argument("--dinov2-source-archive", type=Path); parser.add_argument("--dinov2-weights", type=Path); parser.add_argument("--torch-home", type=Path); parser.add_argument("--certificate-issuance-path", type=Path); parser.add_argument("--certificate-issuance-sha256"); parser.add_argument("--certificate-issuance-sha256-commitment"); parser.add_argument("--certificate-attempt-capability-path", type=Path); parser.add_argument("--row-capability-path", type=Path)
    args = parser.parse_args()
    artifact = run_one_decision(**vars(args))
    print(json.dumps({"status": "ok", "schema_version": artifact["schema_version"], "decision_id": artifact["decision_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
