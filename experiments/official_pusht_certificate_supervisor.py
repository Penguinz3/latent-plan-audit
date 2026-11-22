"""Fresh-process reproducibility certificate supervisor for PWA-PushT.

The supervisor is deliberately narrow: it runs three outcome-blind episode-7
contracts three times in separate processes for each released model.  It never
retries an attempt.  A certificate is emitted only when every artifact passes
the full row validator and the complete P/O/G/J-and-hash signature is identical
within each contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from experiments.official_pusht_batch_study import (
    LOCKED_MODELS,
    LOCKED_SEED,
    NEEDED_ARRAYS,
    RELEASED_CHECKPOINT_SHA256,
    RELEASED_CONFIG_SHA256,
    RELEASED_DATASET_ARCHIVE_SHA256,
    RELEASED_OFFICIAL_COMMIT,
    _array_sha256,
    _archive_matches_extracted_split,
    _canonical_sha256,
    _dataset_files_fingerprint,
    _hardware_fingerprint,
    _git_commit,
    _git_tracked_clean,
    _runtime_fingerprint,
    _sha256_file,
    _segment_data_hashes,
    dinov2_binding,
    protocol_sha256,
    study_source_bundle_sha256,
    validate_artifact,
    validate_known_release_bindings,
    validate_dinov2_bindings,
    validate_reproducibility_certificate,
)


SUPERVISOR_VERSION = "PWA-PushT-fresh-process-supervisor-v1"
CONTRACT_QUANTILES = (0, 2, 4)
CERTIFICATE_SCHEMA_VERSION = "PWA-PushT-reproducibility-certificate-v2"
BUNDLE_SCHEMA_VERSION = "PWA-PushT-certificate-bundle-v1"
COMMON_INITIAL_ARRAY_KEYS = (
    "initial_actions_normalized", "initial_physical_actions", "initial_native_returns",
    "initial_native_state_distances", "initial_final_states", "initial_native_success",
    "initial_final_coverage", "initial_raw_visual_digests", "initial_raw_proprio_digests",
)


def make_contracts(dataset_root: Path, *, seed: int = LOCKED_SEED) -> list[dict[str, Any]]:
    """Freeze three episode-7 offsets from sequence lengths, before outcomes."""
    if seed != LOCKED_SEED:
        raise ValueError("certificate contracts require seed=1")
    with (dataset_root / "val" / "seq_lengths.pkl").open("rb") as handle:
        import pickle

        lengths = [int(value) for value in pickle.load(handle)]
    if len(lengths) != 21:
        raise ValueError("certificate contracts require the 21-trajectory validation split")
    length = lengths[7]
    max_offset = length - 31
    if max_offset < 0:
        raise ValueError("episode 7 is shorter than one locked segment")
    contracts = []
    for k in CONTRACT_QUANTILES:
        offset = int((k * max_offset) // 4)
        contracts.append({
            "contract_id": f"ep007_q{k}",
            "trajectory_id": "val_episode_007",
            "episode": 7,
            "offset": offset,
            "quantile_index": k,
            "segment_length": 31,
            "seed": seed,
            "outcome_blind": True,
            **_segment_data_hashes(dataset_root, 7, offset),
        })
    return contracts


def _winner_pipeline(artifact: Mapping[str, Any]) -> dict[str, Any]:
    decision = artifact.get("decision", {})
    common = artifact.get("common_inventory", {})
    fields = (
        "predicted_choice", "source_planner_choice", "encoded_real_choice",
        "native_reward_choice", "goal_aligned_choice", "reward_comparison_abstained",
        "goal_comparison_abstained", "reward_winner_pattern", "goal_aligned_winner_pattern",
    )
    return {
        "decision": {field: decision.get(field) for field in fields},
        "common_initial": {field: common.get(field) for field in fields},
    }


def _artifact_signature(artifact: Mapping[str, Any], arrays_path: Path) -> dict[str, Any]:
    with np.load(arrays_path, allow_pickle=False) as arrays:
        array_hashes = {key: _array_sha256(arrays[key]) for key in sorted(arrays.files)}
    decision = artifact.get("decision", {})
    rows = artifact.get("candidate_rows", [])
    source_scores = np.asarray([row.get("source_planner_score") for row in rows], dtype=np.float64)
    returned_action = np.asarray(decision.get("returned_mean_action_model"), dtype=np.float32)
    gate = artifact.get("reproducibility_gate", {})
    chunk_equivalence = artifact.get("chunk_equivalence")
    chunk_equivalence_sha256 = _canonical_sha256(chunk_equivalence)
    planning = {
        "initial_inventory_sha256": artifact.get("initial_inventory", {}).get("normalized_sha256"),
        "final_inventory_sha256": array_hashes["final_actions_normalized"],
        "returned_mean_sha256": _array_sha256(returned_action),
        "source_scores_sha256": _array_sha256(source_scores),
        "source_planner_winner": decision.get("source_planner_choice"),
    }

    replay = {
        "state_sha256": (gate.get("state_sha256") or [None])[0],
        "terminal_observation_sha256": (gate.get("terminal_observation_sha256") or [None])[0],
        "cumulative_reward": (gate.get("cumulative_reward") or [None])[0],
        "native_state_distance": (gate.get("native_state_distance") or [None])[0],
        "native_success": (gate.get("native_success") or [None])[0],
    }
    return {
        "signature_sha256": _canonical_sha256({
            "bindings": {
                key: artifact.get(key)
                for key in ("model", "protocol_sha256", "checkpoint_sha256", "config_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256", "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "dataset_split", "states_segment_sha256", "actions_segment_sha256")
            },
            "dinov2": artifact.get("dinov2"),
            "array_hashes": array_hashes,
            "planning": planning,
            "replay": replay,
            "winner_pipeline": _winner_pipeline(artifact),
            "chunk_equivalence": chunk_equivalence,
            "chunk_equivalence_sha256": chunk_equivalence_sha256,
            "dataset_files": artifact.get("dataset_files"),
        }),
        "planning_signature": planning,
        "replay_signature": replay,
        "winner_pipeline": _winner_pipeline(artifact),
        "chunk_equivalence": chunk_equivalence,
        "chunk_equivalence_sha256": chunk_equivalence_sha256,
        "array_hashes": array_hashes,
        "bindings": {
            key: artifact.get(key)
            for key in ("model", "protocol_sha256", "checkpoint_sha256", "config_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256", "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "dataset_split", "states_segment_sha256", "actions_segment_sha256")
        },
        "dinov2": artifact.get("dinov2"),
        "runtime": artifact.get("runtime"),
        "hardware": artifact.get("hardware"),
        "dataset_files": artifact.get("dataset_files"),
    }


def _cross_model_common_errors(
    attempts_by_model: Mapping[str, list[Mapping[str, Any]]],
    contracts: list[Mapping[str, Any]],
) -> list[str]:
    """Require identical outcome-blind initial physical evidence for each contract."""
    errors: list[str] = []
    for contract in contracts:
        contract_id = str(contract.get("contract_id"))
        signatures: dict[str, Mapping[str, Any]] = {}
        for model_name in LOCKED_MODELS:
            passed = [
                attempt.get("signature", {})
                for attempt in attempts_by_model.get(model_name, [])
                if attempt.get("status") == "passed" and attempt.get("contract_id") == contract_id
            ]
            if passed:
                signatures[model_name] = passed[0]
        if len(signatures) != len(LOCKED_MODELS):
            continue
        first = signatures[LOCKED_MODELS[0]].get("array_hashes", {})
        second = signatures[LOCKED_MODELS[1]].get("array_hashes", {})
        for array_key in COMMON_INITIAL_ARRAY_KEYS:
            if first.get(array_key) != second.get(array_key):
                errors.append(f"{contract_id}: cross-model common initial {array_key} mismatch")
    return errors


def _process_result(result: Any) -> tuple[int, str, str]:
    if isinstance(result, Mapping):
        return int(result.get("returncode", 0)), str(result.get("stdout", "")), str(result.get("stderr", ""))
    return int(getattr(result, "returncode", 1)), str(getattr(result, "stdout", "")), str(getattr(result, "stderr", ""))


def _build_model_certificate(
    model_name: str,
    model_spec: Mapping[str, Any],
    contracts: list[Mapping[str, Any]],
    contract_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    passed = all(result.get("passed") is True for result in contract_results)
    representative = next((result.get("representative") for result in contract_results if result.get("representative")), {})
    if not representative:
        representative = {}
    planning_repeats = [result.get("planning_signature", {}) for result in contract_results]
    replay_repeats = [result.get("replay_signature", {}) for result in contract_results]
    certificate = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "scientific_result": False,
        "supervisor_version": SUPERVISOR_VERSION,
        "fresh_process_supervisor": True,
        "supervisor_status": "passed" if passed else "failed",
        "model": model_name,
        "checkpoint_sha256": model_spec.get("checkpoint_sha256"),
        "config_sha256": model_spec.get("config_sha256"),
        "protocol_sha256": representative.get("protocol_sha256"),
        "official_git_commit": representative.get("official_git_commit"),
        "runtime_sha256": representative.get("runtime_sha256"),
        "hardware_sha256": representative.get("hardware_sha256"),
        "source_bundle_sha256": representative.get("source_bundle_sha256"),
        "dataset_archive_sha256": representative.get("dataset_archive_sha256"),
        "dataset_files": representative.get("dataset_files"),
        "seq_lengths_sha256": representative.get("seq_lengths_sha256"),
        "dataset_states_sha256": representative.get("dataset_states_sha256"),
        "dataset_split": representative.get("dataset_split"),
        "dinov2": representative.get("dinov2"),
        "runtime": representative.get("runtime"),
        "hardware": representative.get("hardware"),
        "contracts": [
            {
                "contract": dict(contract),
                "passed": bool(result.get("passed")),
                "repeat_count": int(result.get("repeat_count", 0)),
                "artifact_signature_sha256": list(result.get("artifact_signature_sha256", [])),
                "planning_signature": result.get("planning_signature"),
                "replay_signature": result.get("replay_signature"),
                "winner_pipeline": result.get("winner_pipeline"),
                "chunk_equivalence": result.get("chunk_equivalence"),
                "chunk_equivalence_sha256": result.get("chunk_equivalence_sha256"),
                "array_hashes": result.get("array_hashes"),
                "bindings": result.get("bindings"),
                "dinov2": result.get("dinov2"),
                "dataset_files": result.get("dataset_files"),
            }
            for contract, result in zip(contracts, contract_results)
        ],
        "planning_reproducibility_gate": {
            "passed": passed,
            "repeat_count": 3,
            "contract_signatures": planning_repeats,
        },
        "reproducibility_gate": {
            "passed": passed,
            "repeat_count": 3,
            "contract_signatures": replay_repeats,
        },
        "decision": representative.get("decision", {}),
    }
    return certificate


def validate_certificate_bundle(bundle: Mapping[str, Any], *, bundle_path: Path | None = None) -> list[str]:
    """Recompute every receipt in a supervisor bundle before authorizing rows."""
    errors: list[str] = []
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append("unexpected certificate bundle schema version")
    if bundle.get("supervisor_version") != SUPERVISOR_VERSION:
        errors.append("unexpected certificate supervisor version")
    if bundle.get("scientific_result") is not False or bundle.get("status") != "passed":
        errors.append("certificate bundle is not a passed engineering bundle")
    try:
        if bundle.get("source_bundle_sha256") != study_source_bundle_sha256():
            errors.append("certificate bundle source hash does not match current sources")
    except ValueError as exc:
        errors.append(str(exc))
    errors.extend(validate_dinov2_bindings(bundle.get("dinov2"), require_complete=True, verify_files=True))
    contracts = bundle.get("contracts")
    expected_contracts = None
    dataset_root = bundle.get("dataset_root")
    if isinstance(dataset_root, str):
        try:
            expected_contracts = make_contracts(Path(dataset_root))
        except (OSError, ValueError, EOFError, KeyError) as exc:
            errors.append(f"certificate bundle dataset contracts cannot be recomputed: {exc}")
    else:
        errors.append("certificate bundle lacks dataset_root for contract recomputation")
    if not isinstance(contracts, list) or len(contracts) != 3:
        errors.append("certificate bundle must contain exactly three contracts")
    elif len({_canonical_sha256(contract) for contract in contracts}) != 3:
        errors.append("certificate bundle contracts are not unique")
    elif expected_contracts is not None and contracts != expected_contracts:
        errors.append("certificate bundle contracts do not equal make_contracts(dataset_root)")
    expected_contract_by_id = {
        str(contract["contract_id"]): contract
        for contract in (expected_contracts or [])
        if isinstance(contract, Mapping) and contract.get("contract_id") is not None
    }
    expected_dataset_files = _dataset_files_fingerprint(Path(dataset_root), "val") if isinstance(dataset_root, str) else None
    certificates = bundle.get("certificates")
    if not isinstance(certificates, Mapping) or set(certificates) != set(LOCKED_MODELS):
        errors.append("certificate bundle must contain JEPA and DINO certificates")
        certificates = {}
    advertised_hashes = bundle.get("certificate_sha256")
    if not isinstance(advertised_hashes, Mapping):
        errors.append("certificate bundle lacks certificate hashes")
        advertised_hashes = {}
    for model_name in LOCKED_MODELS:
        cert = certificates.get(model_name)
        if not isinstance(cert, Mapping):
            continue
        if advertised_hashes.get(model_name) != _canonical_sha256(cert):
            errors.append(f"{model_name} certificate hash does not match contents")
        errors.extend(validate_reproducibility_certificate(cert, expected_model=model_name))
        errors.extend(validate_dinov2_bindings(cert.get("dinov2"), require_complete=True, verify_files=True))
        if cert.get("dinov2") != bundle.get("dinov2"):
            errors.append(f"{model_name} certificate DINOv2 binding does not match bundle")
        if cert.get("checkpoint_sha256") != RELEASED_CHECKPOINT_SHA256[model_name]:
            errors.append(f"{model_name} certificate is not bound to the released checkpoint")
        if cert.get("official_git_commit") != RELEASED_OFFICIAL_COMMIT:
            errors.append(f"{model_name} certificate is not bound to the released checkout")
        if cert.get("protocol_sha256") != protocol_sha256():
            errors.append(f"{model_name} certificate protocol hash is not locked")
        if cert.get("dataset_archive_sha256") != RELEASED_DATASET_ARCHIVE_SHA256:
            errors.append(f"{model_name} certificate archive hash is not released")
        if expected_dataset_files is not None and cert.get("dataset_files") != expected_dataset_files:
            errors.append(f"{model_name} certificate extracted dataset hashes do not match dataset")
        if cert.get("runtime_sha256") != _canonical_sha256(_runtime_fingerprint()):
            errors.append(f"{model_name} certificate runtime hash does not match this launcher")
        if cert.get("hardware_sha256") != _canonical_sha256(_hardware_fingerprint()):
            errors.append(f"{model_name} certificate hardware hash does not match this launcher")
        if cert.get("cross_model_common_gate") is not None and cert.get("cross_model_common_gate") != bundle.get("cross_model_common_gate"):
            errors.append(f"{model_name} certificate cross-model common gate does not match bundle")
    attempts_by_model = bundle.get("attempts")
    receipts: list[tuple[str, Mapping[str, Any]]] = []
    if not isinstance(attempts_by_model, Mapping):
        errors.append("certificate bundle lacks preserved attempt receipts")
        attempts_by_model = {}
    for model_name in LOCKED_MODELS:
        attempts = attempts_by_model.get(model_name, [])
        if not isinstance(attempts, list):
            errors.append(f"{model_name} attempt receipts are not a list")
            continue
        for attempt in attempts:
            if isinstance(attempt, Mapping):
                receipts.append((model_name, attempt))
    if len(receipts) != 18:
        errors.append("certificate bundle must preserve exactly 18 attempt receipts")
    receipt_keys: set[tuple[str, str, int]] = set()
    for model_name, attempt in receipts:
        contract_id = str(attempt.get("contract_id"))
        try:
            repeat_index = int(attempt.get("repeat_index", -1))
        except (TypeError, ValueError, OverflowError):
            repeat_index = -1
        receipt_keys.add((model_name, contract_id, repeat_index))
        if attempt.get("model") != model_name:
            errors.append(f"{model_name} receipt model binding does not match its bucket")
        if contract_id not in expected_contract_by_id:
            errors.append(f"{model_name} receipt references an unbundled contract {contract_id}")
        if repeat_index not in (0, 1, 2):
            errors.append(f"{model_name} receipt has invalid repeat index {repeat_index}")
    if len(receipt_keys) != len(receipts):
        errors.append("certificate attempt receipts are not unique")
    recomputed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model_name, attempt in receipts:
        if attempt.get("status") != "passed" or not attempt.get("fresh_process"):
            errors.append(f"{model_name} has a non-passed or non-fresh receipt")
            continue
        contract_id = str(attempt.get("contract_id"))
        frozen_contract = expected_contract_by_id.get(contract_id)
        artifact_path = Path(str(attempt.get("artifact", "")))
        arrays_path = Path(str(attempt.get("arrays", "")))
        if bundle_path is not None:
            if not artifact_path.is_absolute():
                artifact_path = (bundle_path.parent / artifact_path).resolve()
            if not arrays_path.is_absolute():
                arrays_path = (bundle_path.parent / arrays_path).resolve()
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            entry = {
                "decision_id": (
                    f"{model_name}_pusht_val_episode_{int(frozen_contract['episode']):03d}_"
                    f"offset_{int(frozen_contract['offset']):03d}"
                    if frozen_contract is not None else artifact.get("decision_id")
                ),
                "trajectory_id": frozen_contract.get("trajectory_id") if frozen_contract is not None else artifact.get("trajectory_id"),
                "model": model_name,
                "episode": frozen_contract.get("episode") if frozen_contract is not None else artifact.get("episode"),
                "offset": frozen_contract.get("offset") if frozen_contract is not None else artifact.get("offset"),
                "seed": LOCKED_SEED,
                "states_segment_sha256": frozen_contract.get("states_segment_sha256") if frozen_contract is not None else None,
                "actions_segment_sha256": frozen_contract.get("actions_segment_sha256") if frozen_contract is not None else None,
            }
            certificate = certificates.get(model_name, {})
            expected_bindings = {
                field: certificate.get(field)
                for field in (
                    "model", "protocol_sha256", "checkpoint_sha256", "config_sha256",
                    "official_git_commit", "runtime_sha256", "hardware_sha256",
                    "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256",
                    "dataset_states_sha256", "dataset_split", "dinov2",
                )
            }
            validation_errors = validate_artifact(
                artifact, entry, arrays_path=arrays_path,
                require_reproducibility_gate=False, require_v2=True,
                expected_bindings=expected_bindings,
            )
            if validation_errors:
                errors.extend(f"{model_name} receipt {attempt.get('repeat_index')}: {error}" for error in validation_errors)
                continue
            signature = _artifact_signature(artifact, arrays_path)
            if attempt.get("signature", {}).get("signature_sha256") != signature["signature_sha256"]:
                errors.append(f"{model_name} receipt {attempt.get('repeat_index')} signature does not recompute")
                continue
            required_signature = (
                "planning_signature", "replay_signature", "winner_pipeline",
                "chunk_equivalence", "chunk_equivalence_sha256", "array_hashes", "bindings", "dinov2",
            )
            reported_signature = attempt.get("signature")
            if not isinstance(reported_signature, Mapping) or any(not reported_signature.get(field) for field in required_signature):
                errors.append(f"{model_name} receipt {attempt.get('repeat_index')} has incomplete signature")
            elif any(reported_signature.get(field) != signature.get(field) for field in required_signature):
                errors.append(f"{model_name} receipt {attempt.get('repeat_index')} embeds a signature payload different from its artifact")
            if len(signature["array_hashes"]) < len(NEEDED_ARRAYS):
                errors.append(f"{model_name} receipt {attempt.get('repeat_index')} does not hash every required array")
            if any(key not in signature["array_hashes"] for key in NEEDED_ARRAYS):
                errors.append(f"{model_name} receipt {attempt.get('repeat_index')} is missing required NPZ array hashes")
            if frozen_contract is not None:
                for field in ("model", "trajectory_id", "episode", "offset", "seed", "states_segment_sha256", "actions_segment_sha256"):
                    expected = model_name if field == "model" else frozen_contract.get(field, LOCKED_SEED if field == "seed" else None)
                    if field == "trajectory_id":
                        expected = frozen_contract.get("trajectory_id")
                    if artifact.get(field) != expected:
                        errors.append(f"{model_name} receipt {attempt.get('repeat_index')} artifact {field} does not match contract")
            recomputed.setdefault((model_name, str(attempt.get("contract_id"))), []).append(signature)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{model_name} receipt {attempt.get('repeat_index')} cannot be recomputed: {exc}")
    recomputed_cross_errors: list[str] = []
    for contract in expected_contracts or []:
        contract_id = str(contract.get("contract_id"))
        per_model = {
            model_name: recomputed.get((model_name, contract_id), [])
            for model_name in LOCKED_MODELS
        }
        if not all(per_model[model_name] for model_name in LOCKED_MODELS):
            continue
        first = per_model[LOCKED_MODELS[0]][0].get("array_hashes", {})
        second = per_model[LOCKED_MODELS[1]][0].get("array_hashes", {})
        for array_key in COMMON_INITIAL_ARRAY_KEYS:
            if first.get(array_key) != second.get(array_key):
                recomputed_cross_errors.append(f"{contract_id}: cross-model common initial {array_key} mismatch")
    advertised_cross_gate = bundle.get("cross_model_common_gate")
    if not isinstance(advertised_cross_gate, Mapping) or advertised_cross_gate.get("passed") is not (not recomputed_cross_errors):
        errors.append("certificate bundle cross-model common gate does not match recomputed evidence")
    elif advertised_cross_gate.get("errors") != recomputed_cross_errors:
        errors.append("certificate bundle cross-model common-gate errors do not recompute")
    if recomputed_cross_errors:
        errors.extend(recomputed_cross_errors)
    for model_name in LOCKED_MODELS:
        cert = certificates.get(model_name, {})
        cert_contract_records = cert.get("contracts", []) if isinstance(cert, Mapping) else []
        if not isinstance(cert_contract_records, list) or len(cert_contract_records) != len(expected_contracts or []):
            errors.append(f"{model_name} certificate does not contain exactly the canonical contracts")
            cert_contract_records = cert_contract_records if isinstance(cert_contract_records, list) else []
        elif expected_contracts is not None and [record.get("contract") for record in cert_contract_records if isinstance(record, Mapping)] != expected_contracts:
            errors.append(f"{model_name} certificate contracts do not equal make_contracts(dataset_root)")
        for contract_record in cert_contract_records:
            if not isinstance(contract_record, Mapping):
                continue
            contract_id = str(contract_record.get("contract", {}).get("contract_id"))
            signatures = recomputed.get((model_name, contract_id), [])
            expected = contract_record.get("artifact_signature_sha256")
            actual = [item["signature_sha256"] for item in signatures]
            if len(actual) != 3 or not isinstance(expected, list) or actual != [str(value) for value in expected]:
                errors.append(f"{model_name} certificate contract {contract_id} does not cross-match recomputed receipts")
            if isinstance(contracts, list) and contract_record.get("contract") not in contracts:
                errors.append(f"{model_name} certificate contains an unbundled contract")
            if signatures:
                representative_signature = signatures[0]
                for field in (
                    "planning_signature", "replay_signature", "winner_pipeline",
                    "chunk_equivalence", "chunk_equivalence_sha256", "array_hashes", "bindings", "dinov2", "dataset_files",
                ):
                    if contract_record.get(field) != representative_signature.get(field):
                        errors.append(f"{model_name} certificate contract {contract_id} {field} does not match recomputed receipts")
    advertised_bundle_hash = bundle.get("bundle_sha256")
    unsigned_bundle = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if advertised_bundle_hash != _canonical_sha256(unsigned_bundle):
        errors.append("certificate bundle hash does not match contents")
    return sorted(set(errors))


def run_supervisor(
    *,
    official_root: Path,
    dataset_root: Path,
    model_specs: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    contracts: list[Mapping[str, Any]] | None = None,
    python_executable: str | None = None,
    process_runner: Callable[[list[str], Path], Any] | None = None,
    require_released: bool = True,
) -> dict[str, Any]:
    expected_contracts = make_contracts(dataset_root)
    contracts = list(expected_contracts if contracts is None else contracts)
    if contracts != expected_contracts:
        raise ValueError("supervisor contracts must exactly equal make_contracts(dataset_root)")
    if set(model_specs) != set(LOCKED_MODELS):
        raise ValueError("supervisor requires one model specification for JEPA and DINO")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite supervisor output: {output_dir}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_specs: dict[str, dict[str, Any]] = {}
    for model_name in LOCKED_MODELS:
        raw = dict(model_specs[model_name])
        checkpoint = Path(raw["checkpoint"]).resolve()
        config = Path(raw["config"]).resolve()
        actual_checkpoint_sha256 = _sha256_file(checkpoint)
        actual_config_sha256 = _sha256_file(config)
        if not actual_checkpoint_sha256 or not actual_config_sha256:
            raise ValueError(f"{model_name} checkpoint/config does not exist")
        if raw.get("checkpoint_sha256") not in (None, actual_checkpoint_sha256):
            raise ValueError(f"{model_name} supplied checkpoint hash does not match the file")
        if raw.get("config_sha256") not in (None, actual_config_sha256):
            raise ValueError(f"{model_name} supplied config hash does not match the file")
        raw.update({
            "checkpoint": str(checkpoint), "config": str(config),
            "checkpoint_sha256": actual_checkpoint_sha256,
            "config_sha256": actual_config_sha256,
        })
        normalized_specs[model_name] = raw
    bindings = {
        "protocol_sha256": protocol_sha256(),
        "official_git_commit": _git_commit(official_root),
        "dataset_archive_sha256": _sha256_file(official_root / "pusht_noise.zip"),
        "dataset_files": _dataset_files_fingerprint(dataset_root, "val"),
        "source_bundle_sha256": study_source_bundle_sha256(),
        "seq_lengths_sha256": _sha256_file(dataset_root / "val" / "seq_lengths.pkl"),
        "dataset_states_sha256": _sha256_file(dataset_root / "val" / "states.pth"),
        "dataset_split": "val",
        "contracts": contracts,
        "models": {model: {"checkpoint_sha256": normalized_specs[model]["checkpoint_sha256"], "config_sha256": normalized_specs[model]["config_sha256"]} for model in LOCKED_MODELS},
        "runtime": _runtime_fingerprint(),
        "hardware": _hardware_fingerprint(),
        "dinov2": dinov2_binding(),
    }
    bindings["runtime_sha256"] = _canonical_sha256(bindings["runtime"])
    bindings["hardware_sha256"] = _canonical_sha256(bindings["hardware"])
    if require_released:
        known_errors = validate_known_release_bindings(bindings, require_complete=True)
        if known_errors:
            raise ValueError("released supervisor inputs do not match: " + "; ".join(known_errors))
        if not _git_tracked_clean(official_root):
            raise ValueError("official checkout has tracked modifications")
        if not _archive_matches_extracted_split(official_root / "pusht_noise.zip", dataset_root, "val"):
            raise ValueError("extracted validation files do not match the released archive")
    all_attempts: dict[str, list[dict[str, Any]]] = {model: [] for model in LOCKED_MODELS}
    model_certificates: dict[str, dict[str, Any]] = {}
    for model_name in LOCKED_MODELS:
        model_dir = output_dir / model_name
        model_contract_results: list[dict[str, Any]] = []
        for contract in contracts:
            contract_dir = model_dir / str(contract["contract_id"])
            contract_dir.mkdir(parents=True, exist_ok=True)
            attempts: list[dict[str, Any]] = []
            for repeat_index in range(3):
                attempt_dir = contract_dir / f"repeat_{repeat_index:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                out_path = attempt_dir / "artifact.json"
                arrays_path = attempt_dir / "artifact.npz"
                command = [
                    str(python_executable or sys.executable), "-m", "experiments.official_pusht_decision_audit",
                    "--official-root", str(official_root), "--checkpoint", normalized_specs[model_name]["checkpoint"],
                    "--config", normalized_specs[model_name]["config"], "--dataset-root", str(dataset_root),
                    "--episode", str(contract["episode"]), "--offset", str(contract["offset"]), "--seed", str(LOCKED_SEED),
                    "--out", str(out_path), "--arrays-out", str(arrays_path), "--repeat-count", "1",
                    "--planning-repeats", "0", "--model-name", model_name, "--decision-id",
                    f"{model_name}_pusht_val_episode_{int(contract['episode']):03d}_offset_{int(contract['offset']):03d}",
                    "--trajectory-id", str(contract["trajectory_id"]),
                    "--dinov2-source-root", str(bindings["dinov2"]["source_root"]),
                    "--dinov2-source-archive", str(bindings["dinov2"]["source_archive"]),
                    "--dinov2-weights", str(bindings["dinov2"]["weights"]),
                    "--torch-home", str(bindings["dinov2"]["torch_home"]),
                ]
                try:
                    if process_runner:
                        result = process_runner(command, attempt_dir)
                    else:
                        child_env = os.environ.copy()
                        child_env["TORCH_HOME"] = str(bindings["dinov2"]["torch_home"])
                        child_env["PYTHONUTF8"] = "1"
                        result = subprocess.run(
                            command, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True,
                            text=True, check=False, env=child_env,
                        )
                    returncode, stdout, stderr = _process_result(result)
                except KeyboardInterrupt as exc:
                    returncode, stdout, stderr = 130, "", str(exc)
                (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
                (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
                attempt: dict[str, Any] = {
                    "contract_id": contract["contract_id"], "model": model_name, "repeat_index": repeat_index,
                    "fresh_process": True, "command": command, "returncode": returncode,
                    "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                    "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                    "status": "failed", "failure_kind": "process_exit" if returncode else "artifact_validation",
                }
                if returncode == 0 and out_path.is_file() and arrays_path.is_file():
                    try:
                        artifact = json.loads(out_path.read_text(encoding="utf-8"))
                        entry = {
                            "decision_id": artifact.get("decision_id"), "trajectory_id": contract["trajectory_id"],
                            "episode": contract["episode"], "offset": contract["offset"], "seed": LOCKED_SEED,
                            "model": model_name,
                            "states_segment_sha256": contract.get("states_segment_sha256"),
                            "actions_segment_sha256": contract.get("actions_segment_sha256"),
                        }
                        validation_errors = validate_artifact(
                            artifact, entry, arrays_path=arrays_path, require_reproducibility_gate=False,
                            require_v2=True,
                            expected_bindings={
                                **{field: bindings.get(field) for field in (
                                    "protocol_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256",
                                    "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256",
                                    "dataset_states_sha256", "dataset_split", "dinov2",
                                )},
                                "model": model_name,
                                "checkpoint_sha256": normalized_specs[model_name]["checkpoint_sha256"],
                                "config_sha256": normalized_specs[model_name]["config_sha256"],
                            },
                        )
                        if not validation_errors:
                            signature = _artifact_signature(artifact, arrays_path)
                            attempt.update({
                                "status": "passed", "failure_kind": None,
                                "artifact": str(out_path), "arrays": str(arrays_path),
                                "signature": signature,
                            })
                        else:
                            attempt["validation_errors"] = validation_errors
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        attempt["validation_errors"] = [str(exc)]
                (attempt_dir / "attempt.json").write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                attempts.append(attempt)
                all_attempts[model_name].append(attempt)
            passed_attempts = [item for item in attempts if item.get("status") == "passed"]
            signatures = [item["signature"]["signature_sha256"] for item in passed_attempts]
            contract_result = {
                "passed": len(passed_attempts) == 3 and len(set(signatures)) == 1,
                "repeat_count": len(passed_attempts),
                "artifact_signature_sha256": signatures,
                "planning_signature": passed_attempts[0]["signature"]["planning_signature"] if passed_attempts else {},
                "replay_signature": passed_attempts[0]["signature"]["replay_signature"] if passed_attempts else {},
                "winner_pipeline": passed_attempts[0]["signature"]["winner_pipeline"] if passed_attempts else {},
                "chunk_equivalence": passed_attempts[0]["signature"].get("chunk_equivalence") if passed_attempts else {},
                "chunk_equivalence_sha256": passed_attempts[0]["signature"].get("chunk_equivalence_sha256") if passed_attempts else None,
                "array_hashes": passed_attempts[0]["signature"].get("array_hashes", {}) if passed_attempts else {},
                "bindings": passed_attempts[0]["signature"].get("bindings", {}) if passed_attempts else {},
                "dinov2": passed_attempts[0]["signature"].get("dinov2") if passed_attempts else {},
                "dataset_files": passed_attempts[0]["signature"].get("dataset_files", {}) if passed_attempts else {},
                "representative": passed_attempts[0].get("artifact_payload", {}) if passed_attempts else {},
            }
            # Reload the representative only after all attempts are preserved.
            if passed_attempts:
                contract_result["representative"] = json.loads(Path(passed_attempts[0]["artifact"]).read_text(encoding="utf-8"))
            model_contract_results.append(contract_result)
        successful_attempts = [item for item in all_attempts[model_name] if item.get("status") == "passed"]
        binding_keys = ("model", "protocol_sha256", "checkpoint_sha256", "config_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256", "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "dataset_split")
        bindings_consistent = bool(successful_attempts) and len({
            tuple(item["signature"]["bindings"].get(key) for key in binding_keys)
            for item in successful_attempts
        }) == 1
        expected_runtime_sha = _canonical_sha256(_runtime_fingerprint())
        expected_hardware_sha = _canonical_sha256(_hardware_fingerprint())
        expected_common_bindings = {
            "model": model_name,
            "protocol_sha256": protocol_sha256(),
            "checkpoint_sha256": normalized_specs[model_name]["checkpoint_sha256"],
            "config_sha256": normalized_specs[model_name]["config_sha256"],
            "official_git_commit": RELEASED_OFFICIAL_COMMIT,
            "runtime_sha256": expected_runtime_sha,
            "hardware_sha256": expected_hardware_sha,
            "source_bundle_sha256": study_source_bundle_sha256(),
            "dataset_archive_sha256": RELEASED_DATASET_ARCHIVE_SHA256,
            "seq_lengths_sha256": _sha256_file(dataset_root / "val" / "seq_lengths.pkl"),
            "dataset_states_sha256": _sha256_file(dataset_root / "val" / "states.pth"),
            "dataset_split": "val",
        }
        expected_dinov2 = dinov2_binding()
        dinov2_consistent = bool(successful_attempts) and all(
            item["signature"].get("dinov2") == expected_dinov2
            for item in successful_attempts
        )
        if not bindings_consistent or any(
            any(item["signature"]["bindings"].get(key) != value for key, value in expected_common_bindings.items())
            for item in successful_attempts
        ) or not dinov2_consistent:
            for result in model_contract_results:
                result["passed"] = False
        for result in model_contract_results:
            contract_bindings = result.get("bindings", {})
            if not isinstance(contract_bindings, Mapping) or not contract_bindings.get("states_segment_sha256") or not contract_bindings.get("actions_segment_sha256"):
                result["passed"] = False
        model_certificates[model_name] = _build_model_certificate(model_name, normalized_specs[model_name], contracts, model_contract_results)
        model_certificates[model_name]["attempt_count"] = len(all_attempts[model_name])
        (output_dir / f"{model_name}.certificate.json").write_text(
            json.dumps(model_certificates[model_name], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    cross_model_errors = _cross_model_common_errors(all_attempts, contracts)
    cross_model_gate = {
        "passed": not cross_model_errors and all(
            certificate.get("supervisor_status") == "passed"
            for certificate in model_certificates.values()
        ),
        "array_keys": list(COMMON_INITIAL_ARRAY_KEYS),
        "errors": cross_model_errors,
    }
    if not cross_model_gate["passed"]:
        for model_name, certificate in model_certificates.items():
            certificate["supervisor_status"] = "failed"
            certificate["cross_model_common_gate"] = cross_model_gate
            certificate["planning_reproducibility_gate"]["passed"] = False
            certificate["reproducibility_gate"]["passed"] = False
            for contract_record in certificate.get("contracts", []):
                contract_record["passed"] = False
            (output_dir / f"{model_name}.certificate.json").write_text(
                json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    else:
        for model_name, certificate in model_certificates.items():
            certificate["cross_model_common_gate"] = cross_model_gate
            (output_dir / f"{model_name}.certificate.json").write_text(
                json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    # Add the gate after individual certificates are finalized so that the
    # bundle is the single content-integrity record consumed by the launcher.
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "scientific_result": False,
        "supervisor_version": SUPERVISOR_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "source_bundle_sha256": study_source_bundle_sha256(),
        "dinov2": dinov2_binding(),
        "status": "passed" if all(cert.get("supervisor_status") == "passed" for cert in model_certificates.values()) else "failed",
        "contracts": [dict(contract) for contract in contracts],
        "certificates": model_certificates,
        "certificate_sha256": {model: _canonical_sha256(cert) for model, cert in model_certificates.items()},
        "attempts": all_attempts,
        "cross_model_common_gate": cross_model_gate,
    }
    bundle["bundle_sha256"] = _canonical_sha256(bundle)
    (output_dir / "certificate_bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--jepa-config", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-config", type=Path, required=True)
    args = parser.parse_args()
    result = run_supervisor(
        official_root=args.official_root, dataset_root=args.dataset_root, output_dir=args.output_dir,
        model_specs={
            "official_jepa_wm": {"checkpoint": args.jepa_checkpoint, "config": args.jepa_config},
            "official_dino_wm": {"checkpoint": args.dino_checkpoint, "config": args.dino_config},
        },
    )
    print(json.dumps({"status": result["status"], "bundle": str((args.output_dir / "certificate_bundle.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
