"""Stdlib-only v1.4 execution-boundary checks.

This module must stay free of NumPy, PyTorch, model, and task imports.  It is
used before any irreversible issuance/capability consumption.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

V14_CANONICAL_PROJECT_ROOT = Path(r"F:\stockast").resolve()
V14_LAUNCHER_RELATIVE_PATH = "experiments/official_pusht_v14_launcher.py"
V14_CANONICAL_LAUNCHER_PATH = (V14_CANONICAL_PROJECT_ROOT / V14_LAUNCHER_RELATIVE_PATH).resolve()
V14_LAUNCHER_ATTESTATION_DIR = V14_CANONICAL_PROJECT_ROOT / ".v14_launcher_attestations"
V14_SOURCE_BUNDLE_RELATIVE_FILES = (
    "experiments/__init__.py",
    "experiments/official_pusht_decision_audit.py",
    "experiments/official_pusht_batch_study.py",
    "experiments/official_pusht_v12_common_initial.py",
    "experiments/official_pusht_v14_common_initial.py",
    "experiments/official_pusht_v14_decision_audit.py",
    "experiments/official_pusht_v14_batch_study.py",
    "experiments/official_pusht_v14_bootstrap.py",
    "experiments/official_pusht_v14_launcher.py",
    "experiments/official_pusht_v14_certificate_supervisor.py",
    "experiments/analyze_official_pusht_v14.py",
    "latent_score_decomposition.py",
    "PWA_PUSHT_V14_PREREGISTRATION.md",
)
V14_SOURCE_MODULES = (
    "experiments",
    "experiments.official_pusht_decision_audit",
    "experiments.official_pusht_batch_study",
    "experiments.official_pusht_v12_common_initial",
    "experiments.official_pusht_v14_common_initial",
    "experiments.official_pusht_v14_batch_study",
    "experiments.official_pusht_v14_bootstrap",
    "experiments.official_pusht_v14_decision_audit",
    "experiments.official_pusht_v14_certificate_supervisor",
    "experiments.analyze_official_pusht_v14",
    "latent_score_decomposition",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _v14_source_bundle_sha256() -> str:
    files = {relative: _sha256_file(V14_CANONICAL_PROJECT_ROOT / relative) for relative in V14_SOURCE_BUNDLE_RELATIVE_FILES}
    if any(value is None for value in files.values()):
        raise RuntimeError("v1.4 source bundle is incomplete")
    bundle = {"version": "PWA-PushT-source-bundle-v1.4-common-initial", "files": files}
    return hashlib.sha256(_canonical_json(bundle)).hexdigest()


def _v14_scan_sha256(source_bundle_sha256: str) -> str:
    payload = {"status": "clean", "project_root": str(V14_CANONICAL_PROJECT_ROOT), "source_bundle_sha256": source_bundle_sha256}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _consume_v14_launcher_attestation(target_module: str) -> dict[str, str]:
    """Consume the one-shot launcher proof before issuance or row capability use."""
    value = os.environ.get("V14_LAUNCHER_ATTESTATION_PATH")
    if not value:
        raise RuntimeError("v1.4 execution requires a canonical launcher attestation")
    raw_path = Path(value)
    path = raw_path.resolve()
    root = V14_LAUNCHER_ATTESTATION_DIR.resolve()
    if V14_LAUNCHER_ATTESTATION_DIR.is_symlink() or raw_path.parent.resolve() != root or raw_path.is_symlink() or not path.is_file():
        raise RuntimeError("v1.4 launcher attestation path is not canonical")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("v1.4 launcher attestation cannot be read") from exc
    required = {"schema_version", "status", "one_shot", "canonical_project_root", "launcher_path", "launcher_sha256", "target_module", "pid", "executable", "nonce", "scan_sha256", "source_bundle_sha256", "attestation_sha256"}
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("v1.4 launcher attestation schema is invalid")
    unsigned = {key: value for key, value in record.items() if key != "attestation_sha256"}
    if record.get("attestation_sha256") != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise RuntimeError("v1.4 launcher attestation hash mismatch")
    executable = str(Path(sys.executable).resolve())
    if (
        record.get("status") != "authorized" or record.get("one_shot") is not True
        or record.get("canonical_project_root") != str(V14_CANONICAL_PROJECT_ROOT)
        or record.get("launcher_path") != str(V14_CANONICAL_LAUNCHER_PATH)
        or record.get("launcher_sha256") != _sha256_file(V14_CANONICAL_LAUNCHER_PATH)
        or record.get("target_module") != target_module
        or record.get("pid") != os.getpid() or record.get("executable") != executable
        or not isinstance(record.get("nonce"), str) or len(record["nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in record["nonce"])
        or record.get("source_bundle_sha256") != _v14_source_bundle_sha256()
        or record.get("scan_sha256") != _v14_scan_sha256(record["source_bundle_sha256"])
    ):
        raise RuntimeError("v1.4 launcher attestation binding mismatch")
    consumed = path.with_name(path.stem + ".consumed.json")
    consumed_record = {
        "schema_version": record["schema_version"], "status": "consumed", "one_shot": True,
        "target_module": target_module, "canonical_project_root": record["canonical_project_root"],
        "launcher_path": record["launcher_path"], "launcher_sha256": record["launcher_sha256"],
        "source_bundle_sha256": record["source_bundle_sha256"], "attestation_sha256": record["attestation_sha256"],
    }
    payload = _canonical_json(consumed_record)
    try:
        descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("v1.4 launcher attestation was already consumed") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.unlink()
    except BaseException:
        raise
    return {"path": str(consumed), "sha256": hashlib.sha256(payload).hexdigest(), **consumed_record}


consume_v14_launcher_attestation = _consume_v14_launcher_attestation


def validate_v14_consumed_launcher_attestation(record: object, target_module: str) -> list[str]:
    """Validate a consumed parent-launch receipt without mutating it."""
    errors: list[str] = []
    required = {
        "path", "sha256", "schema_version", "status", "one_shot", "target_module",
        "canonical_project_root", "launcher_path", "launcher_sha256",
        "source_bundle_sha256", "attestation_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        return ["v1.4 consumed launcher attestation schema is invalid"]
    path = Path(str(record.get("path"))).resolve()
    if (
        path.parent != V14_LAUNCHER_ATTESTATION_DIR.resolve()
        or not path.name.endswith(".consumed.json")
        or not path.is_file() or path.is_symlink()
    ):
        errors.append("v1.4 consumed launcher attestation path is invalid")
        return errors
    try:
        payload = path.read_bytes()
        stored = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return ["v1.4 consumed launcher attestation cannot be read"]
    expected_stored = {key: value for key, value in record.items() if key not in ("path", "sha256")}
    if stored != expected_stored or record.get("sha256") != hashlib.sha256(payload).hexdigest():
        errors.append("v1.4 consumed launcher attestation receipt mismatch")
    if (
        record.get("schema_version") != "PWA-PushT-launcher-attestation-v1.4-common-initial"
        or record.get("status") != "consumed" or record.get("one_shot") is not True
        or record.get("target_module") != target_module
        or record.get("canonical_project_root") != str(V14_CANONICAL_PROJECT_ROOT)
        or record.get("launcher_path") != str(V14_CANONICAL_LAUNCHER_PATH)
        or record.get("launcher_sha256") != _sha256_file(V14_CANONICAL_LAUNCHER_PATH)
        or record.get("source_bundle_sha256") != _v14_source_bundle_sha256()
    ):
        errors.append("v1.4 consumed launcher attestation binding mismatch")
    return sorted(set(errors))

def _resolved(value: str | os.PathLike[str]) -> Path:
    return Path(value).resolve()


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _source_path(project_root: Path, name: str) -> Path:
    parts = name.split(".")
    if name == "experiments":
        return project_root / "experiments" / "__init__.py"
    return project_root.joinpath(*parts[:-1], parts[-1] + ".py")


def _spec_origins(name: str) -> list[Path]:
    values: list[Path] = []
    loaded = sys.modules.get(name)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if isinstance(origin, str):
            values.append(_resolved(origin))
        paths = getattr(loaded, "__path__", None)
        if paths is not None:
            values.extend(_resolved(item) for item in paths)
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None:
        origin = spec.origin
        if isinstance(origin, str) and origin not in ("built-in", "frozen"):
            values.append(_resolved(origin))
        if spec.submodule_search_locations:
            values.extend(_resolved(item) for item in spec.submodule_search_locations)
    return list(dict.fromkeys(values))


def _source_bytecode_errors(project: Path, source_modules: Iterable[str]) -> list[str]:
    errors: list[str] = []
    source_modules = tuple(source_modules)
    for name in source_modules:
        source = _source_path(project, name)
        parent = source.parent
        stem = source.stem
        for candidate in (parent / f"{stem}.pyc", parent / f"{stem}.pyo"):
            if candidate.is_file() or candidate.is_symlink():
                errors.append(f"v1.4 source module {name} has adjacent bytecode: {candidate}")
        cache_dir = parent / "__pycache__"
        if cache_dir.is_dir():
            for candidate in cache_dir.iterdir():
                if candidate.is_file() or candidate.is_symlink():
                    errors.append(f"v1.4 source module {name} has cached bytecode: {candidate}")
        loaded = sys.modules.get(name)
        if loaded is not None:
            cached = getattr(loaded, "__cached__", None)
            spec = getattr(loaded, "__spec__", None)
            spec_cached = getattr(spec, "cached", None) if spec is not None else None
            for label, value in (("__cached__", cached), ("__spec__.cached", spec_cached)):
                if isinstance(value, str) and (Path(value).is_file() or Path(value).is_symlink()):
                    errors.append(f"v1.4 source module {name} has loaded {label}: {value}")
    return sorted(set(errors))


def v14_execution_preflight(
    official_root: Path | str,
    *,
    project_root: Path | str,
    source_modules: Iterable[str] = V14_SOURCE_MODULES,
) -> list[str]:
    """Return all isolation/origin errors without importing optional modules."""
    errors: list[str] = []
    source_modules = tuple(source_modules)
    if not sys.flags.isolated or not sys.flags.no_user_site:
        errors.append("v1.4 execution requires isolated Python with user site disabled")
    if os.environ.get("PYTHONPATH"):
        errors.append("v1.4 execution rejects ambient PYTHONPATH")
    runtime_roots = (_resolved(sys.prefix), _resolved(sys.base_prefix))
    official = _resolved(official_root)
    project = _resolved(project_root)
    for name in ("numpy", "torch", "yaml", "omegaconf"):
        origins = _spec_origins(name)
        if not origins or any(not any(_under(origin, root) for root in runtime_roots) for origin in origins):
            errors.append(f"v1.4 import origin for {name} is missing or outside its authorized root")
    for name in ("app", "evals"):
        origins = _spec_origins(name)
        if not origins or any(not _under(origin, official) for origin in origins):
            errors.append(f"v1.4 import origin for {name} is missing or outside the released official root")
    for name in source_modules:
        loaded = sys.modules.get(name)
        if loaded is not None:
            actual = getattr(loaded, "__file__", None)
            expected = _source_path(project, name)
            if not isinstance(actual, str) or _resolved(actual) != expected:
                errors.append(f"v1.4 source module {name} origin is not the exact frozen source")
    errors.extend(_source_bytecode_errors(project, source_modules))
    if not sys.dont_write_bytecode:
        errors.append("v1.4 execution requires Python no-bytecode mode (-B)")
    # Project-module origin resolution remains path-only here; the launcher
    # performs the stricter pre-import complete-surface scan.
    return sorted(set(errors))


__all__ = [
    "V14_CANONICAL_PROJECT_ROOT", "V14_CANONICAL_LAUNCHER_PATH",
    "V14_LAUNCHER_ATTESTATION_DIR", "V14_SOURCE_MODULES",
    "consume_v14_launcher_attestation", "validate_v14_consumed_launcher_attestation",
    "v14_execution_preflight",
]
