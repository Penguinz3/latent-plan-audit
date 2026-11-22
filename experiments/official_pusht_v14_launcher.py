"""Canonical isolated launcher for the v1.4 source bundle.

This file is intentionally stdlib-only.  It must be executed by its exact
source path with ``-I -B`` so that no project package is imported or resolved
until the frozen source tree has passed the adjacent-importable scan.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path


SOURCE_MODULES = (
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
SOURCE_BUNDLE_RELATIVE_FILES = (
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
LAUNCHABLE_MODULES = {
    "experiments.official_pusht_v14_certificate_supervisor",
    "experiments.official_pusht_v14_batch_study",
    "experiments.official_pusht_v14_decision_audit",
}
IMPORTABLE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".dll", ".zip", ".egg", ".whl")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _root() -> Path:
    return Path(r"F:\stockast").resolve()


def _source_path(project_root: Path, module_name: str) -> Path:
    parts = module_name.split(".")
    if module_name == "experiments":
        return project_root / "experiments" / "__init__.py"
    return project_root.joinpath(*parts[:-1], parts[-1] + ".py")


def _same(path: Path, expected: Path) -> bool:
    try:
        return path.resolve() == expected.resolve()
    except OSError:
        return False


def _scan_candidate(path: Path, errors: list[str], *, label: str) -> None:
    if path.is_symlink() or path.exists():
        errors.append(f"v1.4 launcher found forbidden adjacent importable {label}: {path}")


def _scan_tagged_native_candidates(parent: Path, stem: str, errors: list[str], *, label: str) -> None:
    """Reject ABI-tagged native alternatives such as ``name.cp311-win_amd64.pyd``."""
    for path in parent.glob(f"{stem}*"):
        if path.name == f"{stem}.py" or path.name == stem:
            continue
        if path.suffix.lower() in (".pyd", ".so", ".dll") and (path.is_file() or path.is_symlink()):
            errors.append(f"v1.4 launcher found forbidden tagged native importable {label}: {path}")


class _FrozenSourceFinder(importlib.abc.MetaPathFinder):
    """Load only the frozen project modules, without putting the project root on sys.path."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.allowed = set(SOURCE_MODULES)

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        if fullname in self.allowed:
            source = _source_path(self.project_root, fullname)
            locations = [str(source.parent)] if fullname == "experiments" else None
            return importlib.util.spec_from_file_location(fullname, source, submodule_search_locations=locations)
        if fullname.startswith("experiments."):
            raise ImportError(f"v1.4 launcher blocked non-bundled project module: {fullname}")
        return None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _source_bundle_sha256(project_root: Path) -> str:
    files = {relative: _sha256_file(project_root / relative) for relative in SOURCE_BUNDLE_RELATIVE_FILES}
    if any(value is None for value in files.values()):
        raise RuntimeError("v1.4 source bundle is incomplete")
    payload = {"version": "PWA-PushT-source-bundle-v1.4-common-initial", "files": files}
    encoded = _canonical_json(payload)
    return hashlib.sha256(encoded).hexdigest()


def _write_launcher_attestation(project_root: Path, requested: str, source_bundle_sha256: str) -> Path:
    attestation_dir = project_root / ".v14_launcher_attestations"
    if attestation_dir.is_symlink():
        raise RuntimeError("v1.4 launcher attestation directory is symlinked")
    attestation_dir.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    scan_payload = {"status": "clean", "project_root": str(project_root), "source_bundle_sha256": source_bundle_sha256}
    scan_encoded = _canonical_json(scan_payload)
    record = {
        "schema_version": "PWA-PushT-launcher-attestation-v1.4-common-initial",
        "status": "authorized", "one_shot": True,
        "canonical_project_root": str(project_root),
        "launcher_path": str((project_root / "experiments" / "official_pusht_v14_launcher.py").resolve()),
        "launcher_sha256": _sha256_file(Path(__file__).resolve()), "target_module": requested,
        "pid": os.getpid(), "executable": str(Path(sys.executable).resolve()), "nonce": nonce,
        "scan_sha256": hashlib.sha256(scan_encoded).hexdigest(),
        "source_bundle_sha256": source_bundle_sha256,
    }
    record["attestation_sha256"] = hashlib.sha256(_canonical_json(record)).hexdigest()
    path = attestation_dir / f"{requested.replace('.', '_')}_{nonce}.json"
    payload = _canonical_json(record)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def scan_frozen_source_tree(project_root: Path) -> list[str]:
    """Return path-only source-boundary violations without importing anything."""
    errors: list[str] = []
    project_root = project_root.resolve()
    if not project_root.is_dir():
        return [f"v1.4 launcher project root is not a directory: {project_root}"]
    for module_name in SOURCE_MODULES:
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            actual = getattr(loaded, "__file__", None)
            expected = _source_path(project_root, module_name)
            if not isinstance(actual, str) or not _same(Path(actual), expected):
                errors.append(f"v1.4 launcher found preloaded shadow module {module_name}: {actual}")
    for module_name in SOURCE_MODULES:
        source = _source_path(project_root, module_name)
        if not source.is_file() or source.is_symlink():
            errors.append(f"v1.4 launcher exact source is missing or symlinked: {source}")
        parent = source.parent
        stem = source.stem
        for suffix in IMPORTABLE_SUFFIXES:
            _scan_candidate(parent / f"{stem}{suffix}", errors, label=f"{module_name}{suffix}")
        _scan_tagged_native_candidates(parent, stem, errors, label=module_name)
        sibling_package = parent / stem
        if module_name != "experiments" and sibling_package.exists():
            errors.append(f"v1.4 launcher found forbidden adjacent package directory: {sibling_package}")
        cache_dir = parent / "__pycache__"
        if cache_dir.is_dir():
            for cached in cache_dir.glob(f"{stem}*.pyc"):
                if cached.is_file() or cached.is_symlink():
                    errors.append(f"v1.4 launcher found forbidden source bytecode: {cached}")
            for cached in cache_dir.glob(f"{stem}*.pyo"):
                if cached.is_file() or cached.is_symlink():
                    errors.append(f"v1.4 launcher found forbidden source bytecode: {cached}")
    for suffix in IMPORTABLE_SUFFIXES:
        _scan_candidate(project_root / f"experiments{suffix}", errors, label=f"experiments{suffix}")
    _scan_tagged_native_candidates(project_root, "experiments", errors, label="experiments")
    return sorted(set(errors))


def _verify_origins(project_root: Path, requested: str) -> None:
    for name in ("experiments", requested):
        module = sys.modules.get(name)
        expected = _source_path(project_root, name)
        actual = getattr(module, "__file__", None) if module is not None else None
        if not isinstance(actual, str) or not _same(Path(actual), expected):
            raise RuntimeError(f"v1.4 launcher origin mismatch for {name}: expected {expected}, got {actual}")
    for name in SOURCE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        expected = _source_path(project_root, name)
        actual = getattr(module, "__file__", None)
        if not isinstance(actual, str) or not _same(Path(actual), expected):
            raise RuntimeError(f"v1.4 launcher source origin mismatch for {name}: expected {expected}, got {actual}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("v1.4 launcher requires Python -I -B")
    if len(arguments) < 2 or arguments[0] != "--module" or arguments[1] not in LAUNCHABLE_MODULES:
        raise ValueError("v1.4 launcher requires --module with a supported v1.4 entrypoint")
    requested = arguments[1]
    target_arguments = arguments[2:]
    project_root = _root()
    expected_launcher = project_root / "experiments" / "official_pusht_v14_launcher.py"
    errors = scan_frozen_source_tree(project_root)
    if errors:
        raise RuntimeError("unsafe v1.4 frozen source tree: " + "; ".join(errors))
    if not _same(Path(__file__), expected_launcher):
        raise RuntimeError(f"v1.4 launcher must run from its exact source path: {expected_launcher}")
    source_bundle_sha256 = _source_bundle_sha256(project_root)
    attestation_path = _write_launcher_attestation(project_root, requested, source_bundle_sha256)
    os.environ["V14_LAUNCHER_ATTESTATION_PATH"] = str(attestation_path)
    finder = _FrozenSourceFinder(project_root)
    sys.meta_path.insert(0, finder)
    sys.argv = [str(_source_path(project_root, requested)), *target_arguments]
    try:
        module = importlib.import_module(requested)
        _verify_origins(project_root, requested)
        entrypoint = getattr(module, "main")
        result = entrypoint() if requested.endswith("decision_audit") else entrypoint(target_arguments)
        return 0 if result is None else int(result)
    finally:
        os.environ.pop("V14_LAUNCHER_ATTESTATION_PATH", None)
        if attestation_path.is_file():
            attestation_path.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


__all__ = ["LAUNCHABLE_MODULES", "SOURCE_MODULES", "scan_frozen_source_tree", "main"]
