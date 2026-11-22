"""Resumable, outcome-blind study harness for official Push-T audit rows.

The manifest is written before any model or simulator outcome is inspected.
The ledger is append-only: a failed attempt is never retried implicitly.  Only
an explicitly classified external interruption may be resumed once; scientific
integrity failures are terminal and stop the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import pickle
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


MANIFEST_VERSION = "PWA-PushT-v1.1"
LEDGER_NAME = "attempts.jsonl"
DATASET_SPLIT = "val"
LOCKED_MODELS = ("official_jepa_wm", "official_dino_wm")
LOCKED_SEED = 1
RELEASED_OFFICIAL_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
RELEASED_DATASET_ARCHIVE_SHA256 = "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08"
RELEASED_CHECKPOINT_SHA256 = {
    "official_jepa_wm": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
    "official_dino_wm": "8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f",
}
RELEASED_CONFIG_SHA256 = {
    "official_jepa_wm": "3f33b4fa67fd6a09c787e4318099e1477805938a58516a07cdd6cd2c1b31d8b3",
    "official_dino_wm": "dfc7a05a12a7830b7a014a943e233bfc68359e724ba3756d948ee5ea1fd82d1e",
}
RELEASED_DATASET_FILE_SHA256 = {
    "seq_lengths.pkl": "ed0218e14c639d254f318419e314799cc063499e6984cd4c080555a004fba40f",
    "states.pth": "7ef3ab74983415802e342642e41bf6429b927d3fff7a54db15da74745dd2fcb0",
    "rel_actions.pth": "c5e04761f261db6f96b2d16db35e5f1d7e42b8d2c507994713c98343d0f685dc",
    "velocities.pth": "b1e4694152206f34931cc5015dfefb4456608c42d9f6a7f2688d20f2fc90d59b",
    "abs_actions.pth": "141231489d7dbdc8a5b90efb6ca764d89e7679aed4a892e0e9aaad0895c3234d",
}
# DINOv2 is an input dependency of both released Push-T encoders.  It is
# pinned separately from the official JEPA-WM checkout because that checkout's
# upstream torch.hub call is otherwise network-dependent.
DINOV2_PINNED_COMMIT = "b194f00db6136677fc8a4cc2ef2168f7699dfba2"
DINOV2_SOURCE_ROOT = Path(r"F:\stockast\tmp\dinov2-release-pinned")
DINOV2_SOURCE_ARCHIVE = Path(r"F:\stockast\tmp\dinov2-b194f00db6136677fc8a4cc2ef2168f7699dfba2.tar.gz")
DINOV2_WEIGHTS = Path(r"F:\stockast\tmp\dinov2-pinned-weights\dinov2_vits14_pretrain.pth")
DINOV2_TORCH_HOME = Path(r"F:\stockast\tmp\torch-cache")
DINOV2_SOURCE_ARCHIVE_SHA256 = "d679cb1bd644cde72cc4adf496b2a8c1644cdc7d8ee6d0defdcda40dc4795f6f"
DINOV2_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_SOURCE_TREE_SHA256 = "f6331d091125a706392f6d7637e1b701b35cc3a4f465d358c7392c12333a383f"
# This is the exact deterministic serialization of the loaded ViT state
# dictionary (sorted key, dtype, shape, and contiguous tensor bytes).
DINOV2_LOADED_STATE_DICT_SHA256 = "5bb315664719c85935c4b67a0f8faef9d01b7dfb57ffbecef1c062f7b4e4aa35"
DINOV2_MODULE_RELATIVE = "dinov2/models/vision_transformer.py"
LOCKED_PROTOCOL = {
    "planner_name": "cem",
    "objective_type": "L2",
    "sum_all_diffs": False,
    "alpha": 0.1,
    "iterations": 30,
    "samples": 300,
    "elites": 10,
    "var_scale": 1.0,
    "horizon": 6,
    "num_act_stepped": 6,
    "frameskip": 5,
    "action_skip": 1,
    "num_frames": 1,
    "num_proprios": 1,
    "action_dim": 10,
    "distribute_planner": False,
    "score_reproduction_tolerance": 2e-5,
    "chunk_equivalence_tolerance": 2e-5,
    "tie_atol": 1e-10,
    "tie_rtol": 1e-8,
    "decomposition_tolerance": 1e-8,
}
SOURCE_BUNDLE_VERSION = "PWA-PushT-source-bundle-v1"
SOURCE_BUNDLE_RELATIVE_FILES = (
    "experiments/official_pusht_decision_audit.py",
    "latent_score_decomposition.py",
    "experiments/official_pusht_batch_study.py",
    "experiments/official_pusht_certificate_supervisor.py",
    "experiments/analyze_official_pusht_study.py",
    "PWA_PUSHT_PREREGISTRATION.md",
)
NEEDED_ARRAYS = (
    "initial_actions_normalized",
    "initial_physical_actions",
    "initial_objective_scores",
    "initial_predicted_visual",
    "initial_encoded_real_visual",
    "initial_predicted_proprio",
    "initial_encoded_real_proprio",
    "initial_native_returns",
    "initial_native_state_distances",
    "initial_final_states",
    "initial_native_success",
    "initial_final_coverage",
    "initial_raw_visual_digests",
    "initial_raw_proprio_digests",
    "final_raw_visual_digests",
    "final_raw_proprio_digests",
    "final_actions_normalized",
    "final_objective_scores",
    "predicted_visual",
    "encoded_real_visual",
    "goal_visual",
    "predicted_proprio",
    "encoded_real_proprio",
    "goal_proprio",
    "native_returns",
    "native_state_distances",
    "final_states",
    "native_success",
    "final_coverage",
    "returned_mean_actions_normalized",
    "returned_predicted_visual",
    "returned_encoded_real_visual",
    "returned_predicted_proprio",
    "returned_encoded_real_proprio",
    "returned_objective_score",
    "returned_native_return",
    "returned_mean_raw_visual_digests",
    "returned_mean_raw_proprio_digests",
    "returned_mean_final_state",
    "returned_mean_native_state_distance",
    "returned_mean_native_success",
    "returned_mean_final_coverage",
)
LEGACY_ARRAYS = (
    "initial_actions_normalized", "initial_predicted_visual", "initial_encoded_real_visual",
    "initial_predicted_proprio", "initial_encoded_real_proprio", "initial_native_returns",
    "initial_native_state_distances", "final_actions_normalized", "predicted_visual",
    "encoded_real_visual", "goal_visual", "predicted_proprio", "encoded_real_proprio",
    "goal_proprio", "native_returns",
)


def _git_tracked_clean(root: Path) -> bool:
    """Require clean tracked code and reject untracked importable source."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        )
        untracked_source = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", "*.py", "*.pyi", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll"],
            check=True, capture_output=True, text=True,
        )
        ignored_source = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py", "*.pyi", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not result.stdout.strip() and not untracked_source.stdout.strip() and not ignored_source.stdout.strip()


def _archive_split_file_fingerprint(archive_path: Path, dataset_split: str = DATASET_SPLIT) -> dict[str, str]:
    """Hash every extracted split file from the released archive contents."""
    cache = globals().setdefault("_ARCHIVE_SPLIT_CACHE", {})
    key = (str(archive_path.resolve()), dataset_split)
    if key in cache:
        return dict(cache[key])
    prefix = f"pusht_noise/{dataset_split}/"
    result: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[relative] = digest.hexdigest()
    cache[key] = dict(result)
    return result


def _archive_matches_extracted_split(archive_path: Path, dataset_root: Path, dataset_split: str = DATASET_SPLIT) -> bool:
    if not archive_path.is_file() or not (dataset_root / dataset_split).is_dir():
        return False
    try:
        return _archive_split_file_fingerprint(archive_path, dataset_split) == {
            key: value for key, value in _dataset_files_fingerprint(dataset_root, dataset_split).items()
            if value is not None
        }
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile):
        return False


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON value without depending on whitespace or mapping order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    """Hash the exact contiguous bytes stored in an NPZ array."""
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def dinov2_loaded_state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash the exact tensors loaded into the offline DINOv2 encoder."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not hasattr(value, "detach"):
            raise TypeError(f"DINOv2 state entry {key!r} is not a tensor")
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        header = json.dumps(
            {"key": str(key), "dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def dinov2_source_tree_sha256(source_root: Path = DINOV2_SOURCE_ROOT) -> str | None:
    """Fingerprint the pinned DINOv2 tree and reject executable cache/native additions."""
    root_path = Path(source_root)
    root = root_path.resolve()
    if root_path.is_symlink() or not root.is_dir():
        return None
    # A symlink/reparse point would let a claimed tree resolve outside the
    # pinned source directory while retaining the same visible path.
    items = list(root.rglob("*"))
    if any(item.is_symlink() for item in items):
        return None
    forbidden_suffixes = {".pyc", ".pyo", ".pyd", ".so", ".dll"}
    if any(item.is_file() and ("__pycache__" in item.parts or item.suffix.lower() in forbidden_suffixes) for item in items):
        return None
    files: dict[str, str | None] = {}
    for path in sorted((item for item in items if item.is_file()), key=lambda item: item.as_posix()):
        if ".git" in path.parts:
            continue
        files[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not files or any(value is None for value in files.values()):
        return None
    return _canonical_sha256({"version": "dinov2-source-tree-v1", "files": files})


def dinov2_binding(
    *, source_root: Path = DINOV2_SOURCE_ROOT,
    source_archive: Path = DINOV2_SOURCE_ARCHIVE,
    weights: Path = DINOV2_WEIGHTS,
    torch_home: Path = DINOV2_TORCH_HOME,
) -> dict[str, Any]:
    """Resolve and fingerprint the exact offline DINOv2 dependency inputs."""
    return {
        "commit": DINOV2_PINNED_COMMIT,
        "source_root": str(Path(source_root).resolve()),
        "source_archive": str(Path(source_archive).resolve()),
        "source_archive_sha256": _sha256_file(Path(source_archive)),
        "source_tree_sha256": dinov2_source_tree_sha256(Path(source_root)),
        "weights": str(Path(weights).resolve()),
        "weights_sha256": _sha256_file(Path(weights)),
        "torch_home": str(Path(torch_home).resolve()),
        "model_name": "dinov2_vits14",
        "module_file": str((Path(source_root).resolve() / DINOV2_MODULE_RELATIVE).resolve()),
        "loaded_state_dict_sha256": DINOV2_LOADED_STATE_DICT_SHA256,
        "loaded_model_name": "dinov2_vits14",
        "patch_size": 14,
        "feature_dim": 384,
        "missing_keys": [],
        "unexpected_keys": [],
    }


def validate_dinov2_bindings(
    binding: Mapping[str, Any] | None, *, require_complete: bool = False,
    verify_files: bool = True,
) -> list[str]:
    """Fail closed unless the exact pinned offline DINOv2 inputs are bound."""
    errors: list[str] = []
    if not isinstance(binding, Mapping):
        return ["DINOv2 dependency binding is missing"] if require_complete else []
    required = (
        "commit", "source_root", "source_archive", "source_archive_sha256",
        "source_tree_sha256", "weights", "weights_sha256", "torch_home", "model_name",
        "module_file", "loaded_state_dict_sha256", "loaded_model_name",
        "patch_size", "feature_dim", "missing_keys", "unexpected_keys",
    )
    if require_complete:
        errors.extend(
            f"DINOv2 binding is missing {field}"
            for field in required
            if field not in binding or binding.get(field) is None or binding.get(field) == ""
        )
    if binding.get("commit") != DINOV2_PINNED_COMMIT:
        errors.append("DINOv2 source commit is not the pinned release")
    if binding.get("model_name") != "dinov2_vits14":
        errors.append("DINOv2 binding model name is not dinov2_vits14")
    expected_paths = {
        "source_root": DINOV2_SOURCE_ROOT,
        "source_archive": DINOV2_SOURCE_ARCHIVE,
        "weights": DINOV2_WEIGHTS,
        "torch_home": DINOV2_TORCH_HOME,
    }
    for field, expected in expected_paths.items():
        value = binding.get(field)
        if value and Path(str(value)).resolve() != expected.resolve():
            errors.append(f"DINOv2 {field} is not the controlled pinned path")
    source_root_value = binding.get("source_root")
    module_file = binding.get("module_file")
    if source_root_value and module_file:
        expected_module = (Path(str(source_root_value)).resolve() / DINOV2_MODULE_RELATIVE).resolve()
        if Path(str(module_file)).resolve() != expected_module:
            errors.append("DINOv2 module file is not under the pinned source tree")
    if binding.get("loaded_state_dict_sha256") != DINOV2_LOADED_STATE_DICT_SHA256:
        errors.append("DINOv2 loaded encoder state hash is not the pinned state")
    if binding.get("loaded_model_name") != "dinov2_vits14":
        errors.append("loaded DINOv2 model name is not dinov2_vits14")
    if binding.get("patch_size") != 14 or binding.get("feature_dim") != 384:
        errors.append("loaded DINOv2 architecture does not match ViT-S/14")
    if binding.get("missing_keys") != [] or binding.get("unexpected_keys") != []:
        errors.append("loaded DINOv2 state dictionary was not an exact match")
    if binding.get("source_archive_sha256") != DINOV2_SOURCE_ARCHIVE_SHA256:
        errors.append("DINOv2 source archive hash is not the pinned archive")
    if binding.get("source_tree_sha256") != DINOV2_SOURCE_TREE_SHA256:
        errors.append("DINOv2 source tree fingerprint is not the pinned tree")
    if binding.get("weights_sha256") != DINOV2_WEIGHTS_SHA256:
        errors.append("DINOv2 weights hash is not the pinned weights")
    if verify_files and not errors:
        actual = dinov2_binding()
        for field in ("source_archive_sha256", "source_tree_sha256", "weights_sha256", "module_file", "loaded_state_dict_sha256"):
            if actual.get(field) != binding.get(field):
                errors.append(f"DINOv2 {field} does not match the resolved file")
    return sorted(set(errors))


def study_source_bundle(root: Path | None = None) -> dict[str, Any]:
    """Return the immutable source files committed to the study protocol."""
    project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    files = {
        relative: _sha256_file(project_root / relative)
        for relative in SOURCE_BUNDLE_RELATIVE_FILES
    }
    return {"version": SOURCE_BUNDLE_VERSION, "files": files}


def study_source_bundle_sha256(root: Path | None = None) -> str:
    bundle = study_source_bundle(root)
    missing = [name for name, digest in bundle["files"].items() if not digest]
    if missing:
        raise ValueError("study source bundle is incomplete: " + ", ".join(missing))
    return _canonical_sha256(bundle)


def protocol_sha256() -> str:
    """Return the commitment for the pinned planner protocol."""
    return _canonical_sha256(LOCKED_PROTOCOL)


def _git_commit(root: Path) -> str | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _runtime_fingerprint() -> dict[str, Any]:
    try:
        from importlib import metadata

        packages = sorted(
            f"{dist.metadata.get('Name', '').lower()}=={dist.version}"
            for dist in metadata.distributions()
            if dist.metadata.get("Name")
        )
    except Exception:
        packages = []
    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "installed_distributions_sha256": hashlib.sha256("\n".join(packages).encode("utf-8")).hexdigest(),
        "installed_distribution_count": len(packages),
        "deterministic_flags_requested": {
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "deterministic_algorithms": True,
        },
    }
    try:
        import torch

        # Fingerprinting a confirmatory runtime also establishes the locked
        # numerical mode, so the manifest/supervisor and child processes hash
        # the same actual state.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, AttributeError):
            pass
        runtime.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_flags_actual": {
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cuda_matmul_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
                "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            },
        })
    except Exception:
        runtime.update({"torch": None, "cuda_runtime": None, "cudnn": None, "deterministic_flags_actual": None})
    return runtime


def _hardware_fingerprint() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hardware.update({
                "gpu_name": props.name,
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_count": int(torch.cuda.device_count()),
            })
        else:
            hardware.update({"gpu_name": None, "gpu_total_memory_bytes": None, "gpu_count": 0})
    except Exception:
        hardware.update({"gpu_name": None, "gpu_total_memory_bytes": None, "gpu_count": 0})
    return hardware


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append and fsync one event; this is the permanent failure log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _latest_events(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                decision_id = str(event["decision_id"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid ledger event on line {line_number}") from exc
            latest[decision_id] = event
    return latest


def _finite_tree(value: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            errors.extend(_finite_tree(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            errors.extend(_finite_tree(child, f"{path}[{index}]"))
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            errors.append(f"non-finite value at {path}")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        errors.append(f"non-finite value at {path}")
    return errors


def _strict_choice(values: Any, *, lower_better: bool, allow_tie: bool = False) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return None
    best = float(np.min(values) if lower_better else np.max(values))
    ties = np.isclose(
        values, best,
        atol=float(LOCKED_PROTOCOL["tie_atol"]),
        rtol=float(LOCKED_PROTOCOL["tie_rtol"]),
    )
    if not allow_tie and int(ties.sum()) != 1:
        return None
    return int(np.flatnonzero(ties)[0])


def _validate_chunk_equivalence(chunk: Mapping[str, Any] | None) -> list[str]:
    """Validate the complete full-population CEM chunk trace.

    The summary fields are derived claims; the 30 per-iteration records are
    the evidence. Confirmatory artifacts must retain all iterations so a
    truncated or hand-edited trace cannot pass on top-level booleans.
    """
    errors: list[str] = []
    if not isinstance(chunk, Mapping):
        return ["real-model chunk equivalence record is missing"]
    if chunk.get("full_population_checked") is not True:
        errors.append("chunk equivalence did not check the full population")
    if chunk.get("iterations_checked") != int(LOCKED_PROTOCOL["iterations"]):
        errors.append("chunk equivalence must report all 30 iterations")
    if chunk.get("population_size") != int(LOCKED_PROTOCOL["samples"]):
        errors.append("chunk equivalence population size is not 300")
    if chunk.get("full_batch_size") != 8 or chunk.get("chunk_batch_size") != 4:
        errors.append("chunk equivalence batches must be primary=8 and reference=4")
    tolerance = chunk.get("tolerance")
    if not isinstance(tolerance, (int, float)) or float(tolerance) != float(LOCKED_PROTOCOL["chunk_equivalence_tolerance"]):
        errors.append("chunk equivalence tolerance is not locked")
        tolerance = float(LOCKED_PROTOCOL["chunk_equivalence_tolerance"])
    trace = chunk.get("trace")
    if not isinstance(trace, list) or len(trace) != int(LOCKED_PROTOCOL["iterations"]):
        errors.append("chunk equivalence trace must contain exactly 30 iterations")
        trace = []
    expected_iterations = list(range(int(LOCKED_PROTOCOL["iterations"])))
    actual_iterations = [item.get("iteration") for item in trace if isinstance(item, Mapping)]
    if actual_iterations != expected_iterations:
        errors.append("chunk equivalence trace iteration IDs are not exactly 0..29")
    drifts: list[float] = []
    row_sets: list[bool] = []
    row_orders: list[bool] = []
    for index, item in enumerate(trace):
        if not isinstance(item, Mapping):
            errors.append(f"chunk equivalence trace row {index} is not an object")
            continue
        if item.get("population_size") != int(LOCKED_PROTOCOL["samples"]):
            errors.append(f"chunk equivalence trace row {index} population is not 300")
        if item.get("primary_batch_size") != 8 or item.get("reference_batch_size") != 4:
            errors.append(f"chunk equivalence trace row {index} batches are not 8/4")
        if item.get("elite_set_agrees") is not True:
            errors.append(f"chunk equivalence elite membership disagrees at iteration {index}")
        row_sets.append(item.get("elite_set_agrees") is True)
        row_orders.append(item.get("elite_order_agrees") is True)
        try:
            drift = float(item["max_abs_score_difference"])
        except (KeyError, TypeError, ValueError, OverflowError):
            drift = math.inf
        if not math.isfinite(drift):
            errors.append(f"chunk equivalence drift is non-finite at iteration {index}")
        elif drift > float(tolerance):
            errors.append(f"chunk equivalence drift exceeds tolerance at iteration {index}")
        drifts.append(drift)
        if item.get("tolerance") != tolerance:
            errors.append(f"chunk equivalence trace tolerance differs at iteration {index}")
    expected_max = max(drifts) if drifts else math.inf
    try:
        reported_max = float(chunk["max_abs_score_difference"])
    except (KeyError, TypeError, ValueError, OverflowError):
        reported_max = math.inf
    if not math.isfinite(reported_max) or not math.isclose(reported_max, expected_max, rel_tol=0.0, abs_tol=1e-12):
        errors.append("chunk equivalence maximum drift does not match its trace")
    if chunk.get("elite_set_agrees") is not all(row_sets) or len(row_sets) != len(trace):
        errors.append("chunk equivalence summary elite membership does not match its trace")
    if chunk.get("elite_order_agrees") is not all(row_orders) or len(row_orders) != len(trace):
        errors.append("chunk equivalence summary elite order does not match its trace")
    return errors


def _winner_pattern(predicted: int | None, observed: int | None, native: int | None) -> str:
    if predicted is None or observed is None or native is None:
        return "indeterminate_abstain"
    if predicted == observed == native:
        return "all_winners_agree"
    if predicted != native and observed == native:
        return "encoded_real_recovers_native_winner"
    if predicted == observed and observed != native:
        return "predicted_and_encoded_real_share_non_native_winner"
    if predicted == native and observed != native:
        return "predicted_and_native_agree_encoded_real_differs"
    if predicted != observed and observed != native and predicted != native:
        return "all_winners_differ"
    return "indeterminate"


def _seq_lengths(dataset_root: Path, dataset_split: str = DATASET_SPLIT) -> list[int]:
    path = dataset_root / dataset_split / "seq_lengths.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        values = pickle.load(handle)
    result = [int(value) for value in values]
    if not result:
        raise ValueError("validation sequence-length file is empty")
    return result


def _dataset_files_fingerprint(dataset_root: Path, dataset_split: str = DATASET_SPLIT) -> dict[str, str | None]:
    """Hash every extracted file used/available in the frozen split."""
    split_root = (dataset_root / dataset_split).resolve()
    if not split_root.is_dir():
        return {}
    paths = sorted(
        (item for item in split_root.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    )
    # Artifact validation runs once per row.  Reuse a digest only while the
    # complete relative-path/size/mtime inventory is unchanged; a modified or
    # added extracted file invalidates the key and is hashed again.
    signature = tuple(
        (path.relative_to(split_root).as_posix(), int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in paths
    )
    cache = globals().setdefault("_DATASET_FILES_FINGERPRINT_CACHE", {})
    cache_key = (str(split_root), signature)
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached)
    files: dict[str, str | None] = {
        path.relative_to(split_root).as_posix(): _sha256_file(path)
        for path in paths
    }
    cache[cache_key] = dict(files)
    return files


def _segment_data_hashes(
    dataset_root: Path, episode: int, offset: int, *, dataset_split: str = DATASET_SPLIT,
    segment_length: int = 31,
) -> dict[str, str | None]:
    """Hash the exact state/action segment consumed by the official runner."""
    try:
        import torch

        split_root = dataset_root / dataset_split
        cache_key = str(split_root.resolve())
        cached = globals().setdefault("_DATASET_SEGMENT_CACHE", {}).get(cache_key)
        if cached is None:
            states = torch.load(split_root / "states.pth", map_location="cpu", weights_only=False).float()
            velocities = torch.load(split_root / "velocities.pth", map_location="cpu", weights_only=False).float()
            raw_actions = torch.load(split_root / "rel_actions.pth", map_location="cpu", weights_only=False).float() / 100.0
            cached = (states, velocities, raw_actions)
            globals()["_DATASET_SEGMENT_CACHE"][cache_key] = cached
        states, velocities, raw_actions = cached
        action_mean = torch.tensor([-0.0087, 0.0068], dtype=torch.float32)
        action_std = torch.tensor([0.2019, 0.2002], dtype=torch.float32)
        full_states = torch.cat([states, velocities], dim=-1)
        normalized_actions = (raw_actions - action_mean) / action_std
        return {
            "states_segment_sha256": _array_sha256(full_states[episode, offset : offset + segment_length].numpy()),
            "actions_segment_sha256": _array_sha256(normalized_actions[episode, offset : offset + segment_length - 1].numpy()),
        }
    except (OSError, RuntimeError, ValueError, IndexError, TypeError):
        return {"states_segment_sha256": None, "actions_segment_sha256": None}


def _manifest_bindings(
    dataset_root: Path,
    *,
    dataset_split: str,
    seed: int,
    official_root: Path | None = None,
    model_specs: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_binding: Mapping[str, Any] | None = None,
    hardware_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect immutable inputs that can be checked again immediately before execution."""
    split_root = dataset_root / dataset_split
    bindings: dict[str, Any] = {
        "protocol_sha256": protocol_sha256(),
        "seq_lengths_sha256": _sha256_file(split_root / "seq_lengths.pkl"),
        "dataset_states_sha256": _sha256_file(split_root / "states.pth"),
        "dataset_archive_sha256": _sha256_file((official_root or dataset_root) / "pusht_noise.zip"),
        "dataset_files": _dataset_files_fingerprint(dataset_root, dataset_split),
        "source_bundle": study_source_bundle(),
        "source_bundle_sha256": study_source_bundle_sha256(),
        "official_git_commit": _git_commit(official_root) if official_root else None,
        "models": {},
        "runtime": dict(runtime_binding or _runtime_fingerprint()),
        "hardware": dict(hardware_binding or _hardware_fingerprint()),
        "dataset_split": dataset_split,
        "seed": seed,
        "dinov2": dinov2_binding(),
    }
    bindings["runtime_sha256"] = _canonical_sha256(bindings["runtime"])
    bindings["hardware_sha256"] = _canonical_sha256(bindings["hardware"])
    for model_name, spec in (model_specs or {}).items():
        bindings["models"][str(model_name)] = {
            "checkpoint_sha256": spec.get("checkpoint_sha256"),
            "config_sha256": spec.get("config_sha256"),
        }
    return bindings


def validate_known_release_bindings(bindings: Mapping[str, Any], *, require_complete: bool = False) -> list[str]:
    errors: list[str] = []
    if bindings.get("official_git_commit") is not None and bindings.get("official_git_commit") != RELEASED_OFFICIAL_COMMIT:
        errors.append("official checkout commit is not the known released commit")
    archive = bindings.get("dataset_archive_sha256")
    if archive is not None and archive != RELEASED_DATASET_ARCHIVE_SHA256:
        errors.append("dataset archive is not the known released Push-T archive")
    models = bindings.get("models")
    if isinstance(models, Mapping):
        for model_name, expected in RELEASED_CHECKPOINT_SHA256.items():
            bound = models.get(model_name, {})
            if bound.get("checkpoint_sha256") is not None and bound.get("checkpoint_sha256") != expected:
                errors.append(f"{model_name} checkpoint is not the known released checkpoint")
            config_expected = RELEASED_CONFIG_SHA256[model_name]
            if bound.get("config_sha256") is not None and bound.get("config_sha256") != config_expected:
                errors.append(f"{model_name} config is not the known released config")
            if require_complete and bound.get("checkpoint_sha256") != expected:
                errors.append(f"{model_name} checkpoint hash binding is required")
            if require_complete and bound.get("config_sha256") != config_expected:
                errors.append(f"{model_name} config hash binding is required")
    elif require_complete:
        errors.append("released model bindings are missing")
    dataset_files = bindings.get("dataset_files")
    for relative, expected in RELEASED_DATASET_FILE_SHA256.items():
        if dataset_files is not None and dataset_files.get(relative) is not None and dataset_files.get(relative) != expected:
            errors.append(f"extracted dataset file {relative} is not from the released archive")
        if require_complete and (not isinstance(dataset_files, Mapping) or dataset_files.get(relative) != expected):
            errors.append(f"extracted dataset file {relative} hash binding is required")
    if require_complete:
        if bindings.get("official_git_commit") != RELEASED_OFFICIAL_COMMIT:
            errors.append("known released official commit binding is required")
        if bindings.get("dataset_archive_sha256") != RELEASED_DATASET_ARCHIVE_SHA256:
            errors.append("known released dataset archive binding is required")
    errors.extend(validate_dinov2_bindings(bindings.get("dinov2"), require_complete=require_complete))
    return sorted(set(errors))


def make_manifest(
    dataset_root: Path,
    *,
    seed: int = 1,
    frameskip: int = 5,
    horizon: int = 6,
    models: tuple[str, ...] = ("official_jepa_wm", "official_dino_wm"),
    dataset_split: str = DATASET_SPLIT,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the locked 20-trajectory x 5-offset x 2-model inventory."""
    if seed != LOCKED_SEED:
        raise ValueError("PWA-PushT-v1.1 fixes seed=1")
    if dataset_split != DATASET_SPLIT:
        raise ValueError("PWA-PushT-v1.1 confirmatory data must use the val split")
    if frameskip != int(LOCKED_PROTOCOL["frameskip"]) or horizon != int(LOCKED_PROTOCOL["horizon"]):
        raise ValueError("PWA-PushT-v1.1 fixes frameskip=5 and horizon=6")
    lengths = _seq_lengths(dataset_root, dataset_split)
    needed = frameskip * horizon + 1
    if len(lengths) != 21:
        raise ValueError(f"PWA-PushT-v1.1 expects 21 validation trajectories, found {len(lengths)}")
    if tuple(models) != LOCKED_MODELS:
        raise ValueError("PWA-PushT-v1.1 requires the released JEPA-WM and DINO-WM pair")
    if bindings is not None:
        known_errors = validate_known_release_bindings(bindings)
        if known_errors:
            raise ValueError("invalid released input binding: " + "; ".join(known_errors))
    rows: list[dict[str, Any]] = []
    for episode, length in enumerate(lengths):
        if episode == 7:
            continue
        max_offset = length - needed
        if max_offset < 0:
            raise ValueError(f"validation trajectory {episode} is shorter than {needed} frames")
        offsets = [int((k * max_offset) // 4) for k in range(5)]
        for offset in offsets:
            segment_hashes = _segment_data_hashes(dataset_root, episode, offset, dataset_split=dataset_split, segment_length=needed)
            for model in models:
                rows.append(
                    {
                        "decision_id": f"{model}_pusht_val_episode_{episode:03d}_offset_{offset:03d}",
                        "trajectory_id": f"val_episode_{episode:03d}",
                        "model": model,
                        "episode": episode,
                        "offset": offset,
                        "seed": seed,
                        **segment_hashes,
                    }
                )
    return {
        "schema_version": MANIFEST_VERSION,
        "selection": {
            "outcome_blind": True,
            "seed": seed,
            "frameskip": frameskip,
            "horizon": horizon,
            "segment_length": needed,
            "offset_rule": "floor(k * (L - 31) / 4), k=0,...,4",
            "offset_count_per_trajectory": 5,
            "excluded_trajectories": [7],
            "models": list(models),
            "dataset_split": dataset_split,
            "selection_source": "val/seq_lengths.pkl only; no model or outcome access",
        },
        "dataset_root": str(dataset_root.resolve()),
        "dataset_split": dataset_split,
        "dataset_states_sha256": _sha256_file(dataset_root / dataset_split / "states.pth"),
        "dataset_files": _dataset_files_fingerprint(dataset_root, dataset_split),
        "source_bundle_sha256": study_source_bundle_sha256(),
        "bindings": dict(bindings or _manifest_bindings(
            dataset_root, dataset_split=dataset_split, seed=seed,
        )),
        "trajectory_count": 20,
        "segment_count": 100,
        "decision_count": len(rows),
        "entries": rows,
    }


def make_train_calibration_manifest(
    dataset_root: Path, *, seed: int = LOCKED_SEED, frameskip: int = 5, horizon: int = 6,
) -> dict[str, Any]:
    """Create a clearly separate, outcome-blind engineering inventory for train data.

    This inventory is intentionally not accepted by ``run_study``: train rows can
    calibrate code and thresholds, but cannot contribute to the locked validation
    denominator.
    """
    if seed != LOCKED_SEED:
        raise ValueError("train calibration uses seed=1")
    lengths = _seq_lengths(dataset_root, "train")
    needed = frameskip * horizon + 1
    entries: list[dict[str, Any]] = []
    for episode, length in enumerate(lengths):
        max_offset = length - needed
        if max_offset < 0:
            continue
        offset = int(max_offset // 2)
        for model in LOCKED_MODELS:
            entries.append({
                "calibration_id": f"{model}_train_episode_{episode:03d}_offset_{offset:03d}",
                "trajectory_id": f"train_episode_{episode:03d}",
                "model": model,
                "episode": episode,
                "offset": offset,
                "seed": seed,
            })
    return {
        "schema_version": "PWA-PushT-train-calibration-v1",
        "scientific_result": False,
        "outcome_blind": True,
        "dataset_split": "train",
        "dataset_root": str(dataset_root.resolve()),
        "selection": {
            "frameskip": frameskip,
            "horizon": horizon,
            "segment_length": needed,
            "offset_rule": "floor((L - 31) / 2)",
            "selection_source": "train/seq_lengths.pkl only; no model or outcome access",
            "models": list(LOCKED_MODELS),
        },
        "entries": entries,
    }


def validate_manifest(
    manifest: Mapping[str, Any], *, minimum_entries: int = 3, require_full: bool = False,
    verify_sources: bool = True,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_VERSION:
        errors.append("unexpected manifest schema_version")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("outcome_blind") is not True:
        errors.append("manifest is not marked outcome_blind")
    elif (
        selection.get("frameskip") != int(LOCKED_PROTOCOL["frameskip"])
        or selection.get("horizon") != int(LOCKED_PROTOCOL["horizon"])
        or selection.get("segment_length")
        != int(LOCKED_PROTOCOL["frameskip"]) * int(LOCKED_PROTOCOL["horizon"]) + 1
        or selection.get("offset_count_per_trajectory") != 5
        or selection.get("excluded_trajectories") != [7]
        or selection.get("models") != list(LOCKED_MODELS)
        or selection.get("dataset_split") != DATASET_SPLIT
        or selection.get("seed") != LOCKED_SEED
    ):
        errors.append("manifest has an invalid frameskip/horizon segment length")
    if manifest.get("dataset_split") != DATASET_SPLIT:
        errors.append("confirmatory manifest must use dataset_split=val")
    if manifest.get("dataset_root") and verify_sources:
        dataset_root = Path(str(manifest["dataset_root"]))
        try:
            lengths = _seq_lengths(dataset_root, DATASET_SPLIT)
            if len(lengths) != 21:
                errors.append("validation sequence lengths must contain 21 trajectories")
            if manifest.get("bindings", {}).get("seq_lengths_sha256") not in (None, _sha256_file(dataset_root / DATASET_SPLIT / "seq_lengths.pkl")):
                errors.append("manifest seq_lengths SHA256 does not match dataset")
            if manifest.get("dataset_states_sha256") not in (None, _sha256_file(dataset_root / DATASET_SPLIT / "states.pth")):
                errors.append("manifest dataset states SHA256 does not match dataset")
            bound_files = (manifest.get("bindings", {}) or {}).get("dataset_files")
            if bound_files is not None and bound_files != _dataset_files_fingerprint(dataset_root, DATASET_SPLIT):
                errors.append("manifest extracted dataset file hashes do not match dataset")
        except (OSError, ValueError, EOFError, pickle.PickleError) as exc:
            errors.append(f"manifest dataset binding cannot be verified: {exc}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) < minimum_entries:
        errors.append(f"manifest needs at least {minimum_entries} entries")
        return errors
    ids = [entry.get("decision_id") for entry in entries if isinstance(entry, Mapping)]
    if len(ids) != len(set(ids)):
        errors.append("manifest contains duplicate decision_id")
    keys = []
    trajectories = []
    episodes = []
    models = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"entry {index} is not an object")
            continue
        required = {"decision_id", "trajectory_id", "model", "episode", "offset", "seed"}
        if not required.issubset(entry):
            errors.append(f"entry {index} is missing required fields")
            continue
        key = (entry["trajectory_id"], entry["offset"], entry["model"])
        keys.append(key)
        trajectories.append(entry["trajectory_id"])
        episodes.append(entry["episode"])
        models.append(entry["model"])
        if entry["episode"] == 7:
            errors.append("manifest includes excluded validation trajectory 7")
        if entry["seed"] != LOCKED_SEED:
            errors.append(f"entry {index} does not use locked seed=1")
    if len(keys) != len(set(keys)):
        errors.append("manifest contains duplicate trajectory/offset/model rows")
    if len(set(episodes)) != len(set(trajectories)):
        errors.append("manifest trajectory and episode IDs are inconsistent")
    if manifest.get("trajectory_count") != len(set(trajectories)):
        errors.append("manifest trajectory_count does not match unique trajectories")
    if manifest.get("segment_count") != len({(row[0], row[1]) for row in keys}):
        errors.append("manifest segment_count does not match unique segments")
    if manifest.get("decision_count") != len(entries):
        errors.append("manifest decision_count does not match entries")
    if require_full:
        expected = 20 * 5 * 2
        if len(entries) != expected or len(set(trajectories)) != 20:
            errors.append("PWA-PushT-v1.1 requires exactly 200 decisions across 20 trajectories")
        for trajectory_id in sorted(set(trajectories)):
            trajectory_rows = [row for row in entries if row.get("trajectory_id") == trajectory_id]
            if len(trajectory_rows) != 10 or {row.get("model") for row in trajectory_rows} != set(selection["models"]):
                errors.append(f"trajectory {trajectory_id} does not have five offsets for both models")
        if verify_sources and manifest.get("dataset_root"):
            dataset_root = Path(str(manifest["dataset_root"]))
            lengths = _seq_lengths(dataset_root, DATASET_SPLIT)
            needed = int(selection["frameskip"]) * int(selection["horizon"]) + 1
            expected = {
                (f"val_episode_{episode:03d}", int((k * (length - needed)) // 4))
                for episode, length in enumerate(lengths) if episode != 7
                for k in range(5)
            }
            observed = {(str(row["trajectory_id"]), int(row["offset"])) for row in entries}
            if not observed.issubset(expected) or observed != expected:
                errors.append("manifest offsets do not match quantiles recomputed from val/seq_lengths.pkl")
            try:
                regenerated = make_manifest(
                    dataset_root, seed=LOCKED_SEED,
                    frameskip=int(selection["frameskip"]), horizon=int(selection["horizon"]),
                )
                if manifest.get("entries") != regenerated.get("entries"):
                    errors.append("manifest entries do not equal canonical regeneration")
                if manifest.get("selection") != regenerated.get("selection"):
                    errors.append("manifest selection does not equal canonical regeneration")
            except (OSError, ValueError, EOFError, pickle.PickleError) as exc:
                errors.append(f"manifest canonical regeneration failed: {exc}")
    if set(models) - set(LOCKED_MODELS):
        errors.append("manifest contains an unsupported model")
    if verify_sources:
        bound_source = (manifest.get("bindings", {}) or {}).get("source_bundle_sha256", manifest.get("source_bundle_sha256"))
        if bound_source is not None:
            try:
                if bound_source != study_source_bundle_sha256():
                    errors.append("manifest source bundle hash does not match current sources")
            except ValueError as exc:
                errors.append(str(exc))
        if manifest.get("source_bundle_sha256") is not None and manifest.get("source_bundle_sha256") != bound_source:
            errors.append("manifest top-level source bundle hash disagrees with bindings")
    return errors


def validate_manifest_bindings(
    manifest: Mapping[str, Any], *,
    official_root: Path,
    model_specs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Verify the released inputs and execution fingerprints before a full run."""
    errors: list[str] = []
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        return ["full confirmatory manifest has no immutable bindings"]
    errors.extend(validate_known_release_bindings(bindings, require_complete=True))
    errors.extend(validate_dinov2_bindings(bindings.get("dinov2"), require_complete=True, verify_files=True))
    if bindings.get("protocol_sha256") != protocol_sha256():
        errors.append("manifest protocol hash is not the locked v1.1 protocol")
    dataset_root = Path(str(manifest.get("dataset_root", "")))
    if not dataset_root.is_dir():
        errors.append("manifest dataset_root does not exist")
    else:
        expected_lengths = _sha256_file(dataset_root / DATASET_SPLIT / "seq_lengths.pkl")
        expected_states = _sha256_file(dataset_root / DATASET_SPLIT / "states.pth")
        if bindings.get("seq_lengths_sha256") != expected_lengths:
            errors.append("manifest seq_lengths binding does not match current dataset")
        if bindings.get("dataset_states_sha256") != expected_states:
            errors.append("manifest states binding does not match current dataset")
    archive_hash = _sha256_file(official_root / "pusht_noise.zip")
    if bindings.get("dataset_archive_sha256") != archive_hash:
        errors.append("manifest dataset archive binding does not match released archive")
    commit = _git_commit(official_root)
    if not commit or bindings.get("official_git_commit") != commit:
        errors.append("manifest official checkout commit binding does not match")
    if not _git_tracked_clean(official_root):
        errors.append("official checkout has tracked modifications")
    manifest_models = bindings.get("models")
    if not isinstance(manifest_models, Mapping) or set(manifest_models) != set(LOCKED_MODELS):
        errors.append("manifest must bind both released model checkpoints and configs")
    else:
        for model_name in LOCKED_MODELS:
            expected = model_specs.get(model_name, {})
            bound = manifest_models.get(model_name, {})
            if bound.get("checkpoint_sha256") != expected.get("checkpoint_sha256"):
                errors.append(f"manifest {model_name} checkpoint binding does not match")
            if bound.get("config_sha256") != expected.get("config_sha256"):
                errors.append(f"manifest {model_name} config binding does not match")
    if bindings.get("dataset_split") != DATASET_SPLIT or bindings.get("seed") != LOCKED_SEED:
        errors.append("manifest binding split/seed is not val/1")
    if bindings.get("runtime_sha256") != _canonical_sha256(_runtime_fingerprint()):
        errors.append("manifest runtime fingerprint does not match execution runtime")
    if bindings.get("hardware_sha256") != _canonical_sha256(_hardware_fingerprint()):
        errors.append("manifest hardware fingerprint does not match execution host")
    try:
        if bindings.get("source_bundle_sha256") != study_source_bundle_sha256():
            errors.append("manifest source bundle hash does not match current sources")
    except ValueError as exc:
        errors.append(str(exc))
    expected_files = _dataset_files_fingerprint(dataset_root, DATASET_SPLIT)
    if bindings.get("dataset_files") != expected_files:
        errors.append("manifest extracted dataset file hashes do not match current dataset")
    if not _archive_matches_extracted_split(official_root / "pusht_noise.zip", dataset_root, DATASET_SPLIT):
        errors.append("manifest extracted validation files do not match the released archive")
    return errors


def validate_reproducibility_certificate(
    certificate: Mapping[str, Any], *, expected_model: str | None = None,
    expected_checkpoint_sha256: str | None = None, expected_config_sha256: str | None = None,
    expected_runtime_sha256: str | None = None, expected_hardware_sha256: str | None = None,
) -> list[str]:
    """Validate the one-time engineering certificate used by confirmatory rows."""
    errors = _finite_tree(certificate)
    if certificate.get("scientific_result") is not False:
        errors.append("reproducibility certificate must remain engineering-only")
    if expected_model is not None and certificate.get("model") != expected_model:
        errors.append("certificate model does not match the authorized model")
    if expected_checkpoint_sha256 is not None and certificate.get("checkpoint_sha256") != expected_checkpoint_sha256:
        errors.append("certificate checkpoint hash does not match the authorized checkpoint")
    if expected_model in RELEASED_CHECKPOINT_SHA256 and certificate.get("checkpoint_sha256") != RELEASED_CHECKPOINT_SHA256[expected_model]:
        errors.append("certificate checkpoint hash is not the known released checkpoint")
    if expected_config_sha256 is not None and certificate.get("config_sha256") != expected_config_sha256:
        errors.append("certificate config hash does not match the authorized config")
    if expected_model in RELEASED_CONFIG_SHA256 and certificate.get("config_sha256") != RELEASED_CONFIG_SHA256[expected_model]:
        errors.append("certificate config hash is not the known released config")
    if expected_runtime_sha256 is not None and certificate.get("runtime_sha256") != expected_runtime_sha256:
        errors.append("certificate runtime fingerprint does not match the authorized runtime")
    if expected_hardware_sha256 is not None and certificate.get("hardware_sha256") != expected_hardware_sha256:
        errors.append("certificate hardware fingerprint does not match the authorized host")
    if certificate.get("fresh_process_supervisor") is True:
        if certificate.get("schema_version") != "PWA-PushT-reproducibility-certificate-v2":
            errors.append("fresh-process certificate has an invalid v2 schema")
        if certificate.get("supervisor_version") != "PWA-PushT-fresh-process-supervisor-v1":
            errors.append("fresh-process certificate has an invalid supervisor version")
        for field in ("source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256", "dataset_states_sha256", "dataset_split"):
            if not certificate.get(field):
                errors.append(f"fresh-process certificate is missing {field}")
        if certificate.get("dataset_archive_sha256") != RELEASED_DATASET_ARCHIVE_SHA256:
            errors.append("fresh-process certificate is not bound to the released dataset archive")
        if not isinstance(certificate.get("dataset_files"), Mapping) or not certificate.get("dataset_files"):
            errors.append("fresh-process certificate lacks extracted dataset file hashes")
        else:
            for relative, expected in RELEASED_DATASET_FILE_SHA256.items():
                if certificate["dataset_files"].get(relative) != expected:
                    errors.append(f"fresh-process certificate dataset file {relative} is not the released file")
        if certificate.get("official_git_commit") != RELEASED_OFFICIAL_COMMIT:
            errors.append("fresh-process certificate is not bound to the known released official commit")
        if certificate.get("protocol_sha256") != protocol_sha256():
            errors.append("fresh-process certificate is not bound to the locked protocol")
        errors.extend(validate_dinov2_bindings(certificate.get("dinov2"), require_complete=True, verify_files=True))
        runtime = certificate.get("runtime")
        required_runtime = {
            "python", "numpy", "torch", "cuda_runtime", "cudnn",
            "installed_distributions_sha256", "installed_distribution_count",
            "deterministic_flags_requested", "deterministic_flags_actual",
        }
        if not isinstance(runtime, Mapping) or not required_runtime.issubset(runtime):
            errors.append("fresh-process certificate lacks the full runtime/package fingerprint")
        elif certificate.get("runtime_sha256") != _canonical_sha256(runtime):
            errors.append("fresh-process certificate runtime hash does not match its fingerprint")
        elif not isinstance(runtime.get("deterministic_flags_actual"), Mapping) or runtime["deterministic_flags_actual"].get("deterministic_algorithms") is not True:
            errors.append("fresh-process certificate does not prove deterministic execution")
        hardware = certificate.get("hardware")
        required_hardware = {"system", "release", "machine", "gpu_name", "gpu_total_memory_bytes", "gpu_count"}
        if not isinstance(hardware, Mapping) or not required_hardware.issubset(hardware):
            errors.append("fresh-process certificate lacks the full hardware fingerprint")
        elif certificate.get("hardware_sha256") != _canonical_sha256(hardware):
            errors.append("fresh-process certificate hardware hash does not match its fingerprint")
        contracts = certificate.get("contracts")
        if not isinstance(contracts, list) or len(contracts) != 3:
            errors.append("fresh-process certificate must contain three predeclared contracts")
        else:
            for index, contract in enumerate(contracts):
                if not isinstance(contract, Mapping):
                    errors.append(f"certificate contract {index} is not an object")
                    continue
                if contract.get("passed") is not True or int(contract.get("repeat_count", 0)) != 3:
                    errors.append(f"certificate contract {index} lacks three successful fresh-process repeats")
                frozen_contract = contract.get("contract")
                if (
                    not isinstance(frozen_contract, Mapping)
                    or int(frozen_contract.get("episode", -1)) != 7
                    or frozen_contract.get("outcome_blind") is not True
                ):
                    errors.append(f"certificate contract {index} is not an outcome-blind episode-7 contract")
                signatures = contract.get("artifact_signature_sha256")
                if not isinstance(signatures, list) or len(signatures) != 3 or len(set(map(str, signatures))) != 1:
                    errors.append(f"certificate contract {index} repeats are not identical")
                if not isinstance(contract.get("array_hashes"), Mapping) or not contract.get("array_hashes"):
                    errors.append(f"certificate contract {index} lacks NPZ array hashes")
                if not isinstance(contract.get("bindings"), Mapping) or not contract.get("bindings"):
                    errors.append(f"certificate contract {index} lacks immutable bindings")
                if not isinstance(contract.get("planning_signature"), Mapping) or not contract.get("planning_signature"):
                    errors.append(f"certificate contract {index} lacks a planning signature")
                if not isinstance(contract.get("replay_signature"), Mapping) or not contract.get("replay_signature"):
                    errors.append(f"certificate contract {index} lacks a replay signature")
                if not isinstance(contract.get("winner_pipeline"), Mapping) or not contract.get("winner_pipeline"):
                    errors.append(f"certificate contract {index} lacks a winner pipeline")
                chunk = contract.get("chunk_equivalence")
                if not isinstance(chunk, Mapping) or not chunk:
                    errors.append(f"certificate contract {index} lacks the canonical chunk trace")
                else:
                    errors.extend(
                        f"certificate contract {index}: {error}"
                        for error in _validate_chunk_equivalence(chunk)
                    )
                    if contract.get("chunk_equivalence_sha256") != _canonical_sha256(chunk):
                        errors.append(f"certificate contract {index} chunk trace hash does not match")
                for field in ("planning_signature", "replay_signature", "winner_pipeline", "chunk_equivalence", "chunk_equivalence_sha256"):
                    if field not in contract:
                        errors.append(f"certificate contract {index} missing {field}")
        if certificate.get("supervisor_status") != "passed":
            errors.append("fresh-process supervisor certificate is not passed")
        if not isinstance(certificate.get("runtime_sha256"), str) or not isinstance(certificate.get("hardware_sha256"), str):
            errors.append("fresh-process certificate is missing runtime/hardware binding")
        decision = certificate.get("decision")
        if not isinstance(decision, Mapping):
            errors.append("fresh-process certificate has no representative decision")
        else:
            for key in ("predicted_choice", "encoded_real_choice", "native_reward_choice", "goal_aligned_choice"):
                if key not in decision:
                    errors.append(f"certificate missing winner identity {key}")
        return errors
    planning = certificate.get("planning_reproducibility_gate")
    if not isinstance(planning, Mapping) or planning.get("passed") is not True or int(planning.get("repeat_count", 0)) < 3:
        errors.append("certificate lacks three identical full CEM planning repeats")
    else:
        for key in ("initial_inventory_sha256", "final_inventory_sha256", "returned_mean_sha256", "source_scores_sha256", "source_planner_winner_index"):
            values = planning.get(key)
            if not isinstance(values, list) or len(values) < 3 or len(set(map(str, values))) != 1:
                errors.append(f"certificate planning gate is not identical for {key}")
    replay = certificate.get("reproducibility_gate")
    if not isinstance(replay, Mapping) or replay.get("passed") is not True or int(replay.get("repeat_count", 0)) < 3:
        errors.append("certificate lacks three identical simulator replay repeats")
    else:
        for key in ("state_sha256", "terminal_observation_sha256", "cumulative_reward", "native_state_distance", "native_success"):
            values = replay.get(key)
            if not isinstance(values, list) or len(values) < 3 or len(set(map(str, values))) != 1:
                errors.append(f"certificate replay gate is not identical for {key}")
        if not isinstance(replay.get("winner_identities"), Mapping):
            errors.append("certificate replay gate has no winner identities")
    decision = certificate.get("decision")
    if not isinstance(decision, Mapping):
        errors.append("certificate has no decision winner identities")
    else:
        for key in ("predicted_choice", "encoded_real_choice", "native_reward_choice", "goal_aligned_choice"):
            if key not in decision:
                errors.append(f"certificate missing winner identity {key}")
    return errors


def _validate_sidecar(path: Path, *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing arrays sidecar: {path}"]
    try:
        with np.load(path, allow_pickle=False) as archive:
            required_arrays = NEEDED_ARRAYS if strict else LEGACY_ARRAYS
            missing = [key for key in required_arrays if key not in archive.files]
            errors.extend(f"sidecar missing {key}" for key in missing)
            if missing:
                return errors
            if tuple(archive["initial_actions_normalized"].shape) != (300, 6, 10):
                errors.append("initial_actions_normalized must have shape (300, 6, 10)")
            if tuple(archive["final_actions_normalized"].shape) != (300, 6, 10):
                errors.append("final_actions_normalized must have shape (300, 6, 10)")
            if strict and archive["initial_physical_actions"].shape[0] != 300:
                errors.append("initial_physical_actions must have 300 candidates")
            if strict and archive["initial_objective_scores"].shape != (300,):
                errors.append("initial_objective_scores must have shape (300,)")
            if strict and archive["final_objective_scores"].shape != (300,):
                errors.append("final_objective_scores must have shape (300,)")
            if archive["initial_predicted_visual"].shape[0] != 300 or archive["initial_encoded_real_visual"].shape[0] != 300:
                errors.append("initial visual latent arrays must have 300 candidates")
            if archive["predicted_visual"].shape[0] != 300 or archive["encoded_real_visual"].shape[0] != 300:
                errors.append("visual latent arrays must have 300 candidates")
            if archive["initial_predicted_proprio"].shape[0] != 300 or archive["initial_encoded_real_proprio"].shape[0] != 300:
                errors.append("initial proprio latent arrays must have 300 candidates")
            if archive["predicted_proprio"].shape[0] != 300 or archive["encoded_real_proprio"].shape[0] != 300:
                errors.append("proprio latent arrays must have 300 candidates")
            if archive["native_returns"].shape != (300,):
                errors.append("native_returns must have shape (300,)")
            if archive["initial_native_returns"].shape != (300,) or archive["initial_native_state_distances"].shape != (300,):
                errors.append("initial native outcome arrays must have shape (300,)")
            for key in (("initial_final_states", "final_states") if strict else ()):
                if archive[key].shape[0] != 300:
                    errors.append(f"{key} must have 300 candidates")
            for key in (("initial_native_success", "initial_final_coverage", "native_state_distances", "native_success", "final_coverage") if strict else ("native_returns",)):
                if archive[key].shape != (300,):
                    errors.append(f"{key} must have shape (300,)")
            if strict:
                for key in ("initial_raw_visual_digests", "initial_raw_proprio_digests"):
                    if tuple(archive[key].shape) != (300, 32):
                        errors.append(f"{key} must have shape (300, 32)")
                for key in ("final_raw_visual_digests", "final_raw_proprio_digests"):
                    if tuple(archive[key].shape) != (300, 32):
                        errors.append(f"{key} must have shape (300, 32)")
                for key in ("returned_mean_raw_visual_digests", "returned_mean_raw_proprio_digests"):
                    if tuple(archive[key].shape) != (1, 32):
                        errors.append(f"{key} must have shape (1, 32)")
                if archive["returned_mean_final_state"].shape[0] != 1:
                    errors.append("returned_mean_final_state must contain one state")
                for key in ("returned_mean_native_state_distance", "returned_mean_native_success", "returned_mean_final_coverage"):
                    if tuple(archive[key].shape) != (1,):
                        errors.append(f"{key} must have shape (1,)")
                if tuple(archive["returned_mean_actions_normalized"].shape) != (6, 10):
                    errors.append("returned_mean_actions_normalized must have shape (6, 10)")
                if tuple(archive["returned_native_return"].shape) != (1,):
                    errors.append("returned_native_return must have shape (1,)")
                if tuple(archive["returned_objective_score"].shape) != (1,):
                    errors.append("returned_objective_score must have shape (1,)")
                for key in (
                    "returned_predicted_visual", "returned_encoded_real_visual",
                    "returned_predicted_proprio", "returned_encoded_real_proprio",
                ):
                    if archive[key].shape[0] != 1:
                        errors.append(f"{key} must contain one returned-mean encoding")
            for key in required_arrays:
                if not np.isfinite(np.asarray(archive[key], dtype=np.float64)).all():
                    errors.append(f"sidecar contains non-finite {key}")
    except (OSError, TypeError, ValueError, EOFError) as exc:
        errors.append(f"sidecar unreadable: {exc}")
    return errors


def validate_artifact(
    artifact: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    arrays_path: Path | None = None,
    certificate_sha256: str | None = None,
    manifest_sha256: str | None = None,
    require_reproducibility_gate: bool = True,
    require_v2: bool = False,
    expected_bindings: Mapping[str, Any] | None = None,
) -> list[str]:
    errors = _finite_tree(artifact)
    strict_schema = artifact.get("schema_version") == "official-pusht-decision-audit-v2"
    if require_v2 and not strict_schema:
        errors.append("confirmatory artifact must use schema_version official-pusht-decision-audit-v2")
    if artifact.get("scientific_result") is not False:
        errors.append("scientific_result must remain false for a one-decision pilot")
    if strict_schema and (artifact.get("seed") != entry.get("seed") or artifact.get("seed") != LOCKED_SEED):
        errors.append("artifact seed does not match locked manifest seed=1")
    if strict_schema and artifact.get("dataset_split") != DATASET_SPLIT:
        errors.append("artifact dataset_split must be val")
    if strict_schema and manifest_sha256 is not None and artifact.get("manifest_sha256") != manifest_sha256:
        errors.append("artifact manifest SHA256 does not match execution manifest")
    if strict_schema:
        for field in ("protocol_sha256", "checkpoint_sha256", "config_sha256", "official_git_commit", "seq_lengths_sha256", "dataset_states_sha256", "runtime_sha256", "hardware_sha256"):
            if not isinstance(artifact.get(field), str) or not artifact.get(field):
                errors.append(f"artifact is missing immutable binding {field}")
        errors.extend(validate_dinov2_bindings(artifact.get("dinov2"), require_complete=True, verify_files=True))
        if artifact.get("protocol_sha256") != protocol_sha256():
            errors.append("artifact protocol hash does not match the locked protocol")
        if artifact.get("protocol") != LOCKED_PROTOCOL:
            errors.append("artifact protocol constants do not match the locked protocol")
        if artifact.get("official_git_commit") != RELEASED_OFFICIAL_COMMIT:
            errors.append("artifact official checkout is not the known released commit")
        if artifact.get("dataset_archive_sha256") != RELEASED_DATASET_ARCHIVE_SHA256:
            errors.append("artifact dataset archive is not the known released archive")
        if artifact.get("model") in RELEASED_CHECKPOINT_SHA256 and artifact.get("checkpoint_sha256") != RELEASED_CHECKPOINT_SHA256[artifact.get("model")]:
            errors.append("artifact checkpoint is not the known released checkpoint")
        if artifact.get("model") in RELEASED_CONFIG_SHA256 and artifact.get("config_sha256") != RELEASED_CONFIG_SHA256[artifact.get("model")]:
            errors.append("artifact config is not the known released config")
        if artifact.get("dataset_split") != DATASET_SPLIT:
            errors.append("artifact dataset split is not val")
        try:
            if artifact.get("source_bundle_sha256") != study_source_bundle_sha256():
                errors.append("artifact source bundle hash does not match current sources")
        except ValueError as exc:
            errors.append(str(exc))
        dataset_root_value = artifact.get("dataset_root")
        if not dataset_root_value:
            errors.append("artifact is missing dataset_root for independent dataset verification")
        else:
            artifact_dataset_root = Path(str(dataset_root_value))
            actual_files = _dataset_files_fingerprint(artifact_dataset_root, DATASET_SPLIT)
            if artifact.get("dataset_files") != actual_files:
                errors.append("artifact extracted dataset file hashes do not match dataset")
            actual_lengths = _sha256_file(artifact_dataset_root / DATASET_SPLIT / "seq_lengths.pkl")
            actual_states = _sha256_file(artifact_dataset_root / DATASET_SPLIT / "states.pth")
            if artifact.get("seq_lengths_sha256") != actual_lengths:
                errors.append("artifact seq_lengths SHA256 does not match dataset")
            if artifact.get("dataset_states_sha256") != actual_states:
                errors.append("artifact states SHA256 does not match dataset")
            try:
                actual_segment = _segment_data_hashes(
                    artifact_dataset_root,
                    int(artifact.get("episode", -1)),
                    int(artifact.get("offset", -1)),
                    dataset_split=DATASET_SPLIT,
                    segment_length=int(artifact.get("frameskip", 5)) * int(artifact.get("horizon", 6)) + 1,
                )
                for field in ("states_segment_sha256", "actions_segment_sha256"):
                    if artifact.get(field) != actual_segment.get(field):
                        errors.append(f"artifact {field} does not match the extracted segment")
            except (TypeError, ValueError, OverflowError):
                errors.append("artifact episode/offset/segment dimensions cannot be independently verified")
        runtime = artifact.get("runtime")
        if not isinstance(runtime, Mapping) or artifact.get("runtime_sha256") != _canonical_sha256(runtime):
            errors.append("artifact runtime fingerprint hash mismatch")
        elif not isinstance(runtime.get("deterministic_flags_actual"), Mapping) or runtime["deterministic_flags_actual"].get("deterministic_algorithms") is not True:
            errors.append("artifact does not prove deterministic algorithms were enabled")
        hardware = artifact.get("hardware")
        if not isinstance(hardware, Mapping) or artifact.get("hardware_sha256") != _canonical_sha256(hardware):
            errors.append("artifact hardware fingerprint hash mismatch")
        if artifact.get("official_root"):
            official_root = Path(str(artifact["official_root"]))
            if not _git_tracked_clean(official_root):
                errors.append("artifact official checkout has tracked modifications")
            if dataset_root_value and not _archive_matches_extracted_split(
                official_root / "pusht_noise.zip", Path(str(dataset_root_value)), DATASET_SPLIT
            ):
                errors.append("artifact extracted validation files do not match the released archive")
        else:
            errors.append("artifact is missing official_root for checkout/archive verification")
        if expected_bindings is not None:
            for field in (
                "protocol_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256",
                "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256",
                "dataset_states_sha256", "dataset_split",
            ):
                expected = expected_bindings.get(field)
                if expected is not None and artifact.get(field) != expected:
                    errors.append(f"artifact {field} does not match the execution binding")
            for field in ("model", "checkpoint_sha256", "config_sha256"):
                expected = expected_bindings.get(field)
                if expected is not None and artifact.get(field) != expected:
                    errors.append(f"artifact {field} does not match the execution binding")
            expected_dinov2 = expected_bindings.get("dinov2")
            if expected_dinov2 is not None and artifact.get("dinov2") != expected_dinov2:
                errors.append("artifact DINOv2 dependency binding does not match the execution binding")
    if artifact.get("episode") != entry.get("episode") or artifact.get("offset") != entry.get("offset"):
        errors.append("artifact episode/offset does not match manifest")
    if artifact.get("trajectory_id") != entry.get("trajectory_id"):
        errors.append("artifact trajectory_id does not match manifest")
    if strict_schema:
        for field, expected in (
            ("frameskip", int(LOCKED_PROTOCOL["frameskip"])),
            ("horizon", int(LOCKED_PROTOCOL["horizon"])),
            ("cem_iterations", int(LOCKED_PROTOCOL["iterations"])),
            ("cem_num_samples", int(LOCKED_PROTOCOL["samples"])),
            ("cem_num_elites", int(LOCKED_PROTOCOL["elites"])),
            ("model_action_dim", int(LOCKED_PROTOCOL["action_dim"])),
            ("physical_replay_steps", int(LOCKED_PROTOCOL["frameskip"]) * int(LOCKED_PROTOCOL["horizon"])),
        ):
            if artifact.get(field) != expected:
                errors.append(f"artifact {field} does not match the locked protocol")
        for field in ("states_segment_sha256", "actions_segment_sha256"):
            if entry.get(field) is not None and artifact.get(field) != entry.get(field):
                errors.append(f"artifact {field} does not match manifest segment binding")
    initial_inventory = artifact.get("initial_inventory")
    if (
        not isinstance(initial_inventory, Mapping)
        or initial_inventory.get("outcome_blind") is not True
        or initial_inventory.get("count") != 300
        or not isinstance(initial_inventory.get("normalized_sha256"), str)
    ):
        errors.append("artifact must preserve the 300-plan outcome-blind initial inventory")
    if strict_schema and isinstance(initial_inventory, Mapping):
        for field in ("physical_sha256", "objective_scores_sha256", "score_capture"):
            if not isinstance(initial_inventory.get(field), str):
                errors.append(f"initial inventory is missing {field}")
    common_inventory = artifact.get("common_inventory")
    if (
        not isinstance(common_inventory, Mapping)
        or common_inventory.get("outcome_blind") is not True
        or common_inventory.get("count") != 300
        or "predicted_choice" not in common_inventory
        or "encoded_real_choice" not in common_inventory
        or "native_reward_choice" not in common_inventory
        or "goal_aligned_choice" not in common_inventory
    ):
        errors.append("artifact must contain the full common-inventory P/O/G/J audit")
    if strict_schema and isinstance(common_inventory, Mapping):
        for field in (
            "initial_physical_action_sha256", "initial_j_sha256", "initial_g_sha256",
            "initial_objective_scores_sha256", "initial_raw_visual_digest_array_sha256",
            "initial_raw_proprio_digest_array_sha256",
        ):
            if not isinstance(common_inventory.get(field), str):
                errors.append(f"common inventory is missing {field}")
        common_matched_abstention = (
            common_inventory.get("source_recomputed_abstained") is True
            and common_inventory.get("source_planner_choice") is None
            and common_inventory.get("predicted_choice") is None
        )
        if common_inventory.get("initial_captured_winner_agrees") is not True and not common_matched_abstention:
            errors.append("common inventory captured-vs-recomputed winner disagrees without matched abstention")
    expected_id = str(entry.get("decision_id"))
    if artifact.get("model") != entry.get("model"):
        errors.append("artifact model does not match manifest")
    if artifact.get("decision_id") not in (None, expected_id):
        errors.append("artifact top-level decision_id does not match manifest")
    decision = artifact.get("decision")
    if not isinstance(decision, Mapping) or decision.get("decision_id") != expected_id:
        errors.append("artifact decision_id does not match manifest")
    candidates = artifact.get("candidate_rows")
    if not isinstance(candidates, list) or len(candidates) != 300:
        errors.append("artifact must contain exactly 300 candidate_rows")
    else:
        ids = [row.get("candidate_id") for row in candidates if isinstance(row, Mapping)]
        if len(ids) != 300 or len(set(ids)) != 300:
            errors.append("candidate IDs must be unique across 300 rows")
        for index, row in enumerate(candidates):
            if not isinstance(row, Mapping):
                errors.append(f"candidate row {index} is not an object")
                continue
            if np.asarray(row.get("action_model", [])).shape != (6, 10):
                errors.append(f"candidate row {index} action_model must have shape (6, 10)")
            for field in ("native_return", "native_state_distance", "predicted_score", "encoded_real_score"):
                if field not in row:
                    errors.append(f"candidate row {index} missing {field}")
    if isinstance(decision, Mapping):
        if decision.get("candidate_count") != 300:
            errors.append("decision candidate_count must be 300")
        if "native_reward_choice" not in decision or "goal_aligned_choice" not in decision:
            errors.append("decision must report both reward and goal-distance winners")
        if "reward_comparison_abstained" not in decision or "goal_comparison_abstained" not in decision:
            errors.append("decision must report reward and goal-distance abstentions")
        decision_matched_abstention = (
            decision.get("source_recomputed_abstained") is True
            and decision.get("source_planner_choice") is None
            and decision.get("predicted_choice") is None
        )
        if strict_schema and decision.get("source_recomputed_winner_agrees") is not True and not decision_matched_abstention:
            errors.append("final captured-vs-recomputed winner disagrees without matched abstention")
    gate = artifact.get("reproducibility_gate")
    if certificate_sha256 is None:
        if require_reproducibility_gate and (not isinstance(gate, Mapping) or gate.get("passed") is not True or int(gate.get("repeat_count", 0)) < 3):
            errors.append("reproducibility gate must pass with at least three repeats")
    else:
        certificate_ref = artifact.get("reproducibility_certificate")
        if (
            not isinstance(certificate_ref, Mapping)
            or certificate_ref.get("status") != "passed"
            or certificate_ref.get("model") != entry.get("model")
            or certificate_ref.get("sha256") != certificate_sha256
        ):
            errors.append("artifact does not reference the validated engineering certificate")
    chunk = artifact.get("chunk_equivalence")
    if strict_schema:
        errors.extend(_validate_chunk_equivalence(chunk))
    elif not isinstance(chunk, Mapping) or chunk.get("elite_set_agrees", chunk.get("elite_order_agrees")) is not True:
        errors.append("real-model chunk equivalence elite-membership gate did not pass")
    else:
        if float(chunk.get("tolerance", math.inf)) != float(LOCKED_PROTOCOL["chunk_equivalence_tolerance"]):
            errors.append("chunk equivalence tolerance is not locked")
        if float(chunk.get("max_abs_score_difference", math.inf)) > float(LOCKED_PROTOCOL["chunk_equivalence_tolerance"]):
            errors.append("chunk equivalence score difference exceeds tolerance")
    tolerance = float(artifact.get("score_reproduction_tolerance", math.inf))
    if tolerance != float(LOCKED_PROTOCOL["score_reproduction_tolerance"]):
        errors.append("score reproduction tolerance is not locked")
    if float(artifact.get("score_reproduction_max_abs_error", math.inf)) > tolerance:
        errors.append("source planner score reproduction exceeds tolerance")
    if float(artifact.get("decomposition_max_abs_error", math.inf)) > 1e-8:
        errors.append("latent decomposition identity is not exact")
    if strict_schema:
        if float(artifact.get("initial_score_reproduction_max_abs_error", math.inf)) > tolerance:
            errors.append("initial captured CEM score reproduction exceeds tolerance")
        if float(artifact.get("initial_decomposition_max_abs_error", math.inf)) > 1e-8:
            errors.append("initial latent decomposition identity is not exact")
        if abs(float(artifact.get("returned_mean_score_reproduction_error", math.inf))) > tolerance:
            errors.append("returned-mean score reproduction exceeds tolerance")
        if float(artifact.get("returned_mean_decomposition_identity_error", math.inf)) > 1e-8:
            errors.append("returned-mean latent decomposition identity is not exact")
        if not isinstance(common_inventory, Mapping) or float(common_inventory.get("decomposition_max_abs_error", math.inf)) > 1e-8:
            errors.append("common initial latent decomposition identity is not exact")
    sidecar = arrays_path
    if sidecar is None and isinstance(artifact.get("arrays_sidecar"), str):
        sidecar = Path(artifact["arrays_sidecar"])
    if sidecar is None:
        errors.append("artifact has no arrays sidecar")
    else:
        reported_sidecar = artifact.get("arrays_sidecar")
        if arrays_path is not None and isinstance(reported_sidecar, str):
            if Path(reported_sidecar).resolve() != arrays_path.resolve():
                errors.append("artifact arrays_sidecar does not match expected output path")
        expected_hash = artifact.get("arrays_sidecar_sha256")
        actual_hash = _sha256_file(sidecar)
        if actual_hash is None or actual_hash != expected_hash:
            errors.append("arrays sidecar SHA256 mismatch")
        errors.extend(_validate_sidecar(sidecar, strict=strict_schema))
        try:
            with np.load(sidecar, allow_pickle=False) as archive:
                if strict_schema and isinstance(initial_inventory, Mapping):
                    actual_initial_hash = _array_sha256(archive["initial_actions_normalized"])
                    if initial_inventory.get("normalized_sha256") != actual_initial_hash:
                        errors.append("initial normalized action hash does not match NPZ sidecar")
                    actual_physical_hash = _array_sha256(archive["initial_physical_actions"])
                    if initial_inventory.get("physical_sha256") != actual_physical_hash:
                        errors.append("initial physical action hash does not match NPZ sidecar")
                    actual_objective_hash = _array_sha256(archive["initial_objective_scores"])
                    if initial_inventory.get("objective_scores_sha256") != actual_objective_hash:
                        errors.append("initial objective-score hash does not match NPZ sidecar")
                if strict_schema and isinstance(common_inventory, Mapping):
                    if common_inventory.get("initial_j_sha256") != _array_sha256(archive["initial_native_returns"]):
                        errors.append("initial J hash does not match NPZ sidecar")
                    initial_g = -np.asarray(archive["initial_native_state_distances"], dtype=np.float64)
                    if common_inventory.get("initial_g_sha256") != _array_sha256(initial_g):
                        errors.append("initial G hash does not match NPZ sidecar")
                    if common_inventory.get("initial_objective_scores_sha256") != _array_sha256(archive["initial_objective_scores"]):
                        errors.append("initial captured objective-score hash does not match NPZ sidecar")
                    expected_terminal_hash = _canonical_sha256({
                        "visual": _array_sha256(archive["initial_encoded_real_visual"]),
                        "proprio": _array_sha256(archive["initial_encoded_real_proprio"]),
                    })
                    if common_inventory.get("initial_terminal_observation_sha256") != expected_terminal_hash:
                        errors.append("initial terminal observation hash does not match NPZ sidecar")
                    for field, array_key in (
                        ("initial_raw_visual_digest_array_sha256", "initial_raw_visual_digests"),
                        ("initial_raw_proprio_digest_array_sha256", "initial_raw_proprio_digests"),
                    ):
                        if common_inventory.get(field) != _array_sha256(archive[array_key]):
                            errors.append(f"common inventory {field} does not match NPZ sidecar")
                if strict_schema:
                    reported_array_hashes = artifact.get("array_hashes")
                    actual_array_hashes = {key: _array_sha256(archive[key]) for key in sorted(archive.files)}
                    if not isinstance(reported_array_hashes, Mapping) or dict(reported_array_hashes) != actual_array_hashes:
                        errors.append("artifact array_hashes do not exactly match NPZ arrays")
                    reported_hashes = {
                        "initial_final_states_sha256": _array_sha256(archive["initial_final_states"]),
                        "initial_native_success_sha256": _array_sha256(archive["initial_native_success"]),
                        "initial_final_coverage_sha256": _array_sha256(archive["initial_final_coverage"]),
                        "initial_raw_visual_digest_array_sha256": _array_sha256(archive["initial_raw_visual_digests"]),
                        "initial_raw_proprio_digest_array_sha256": _array_sha256(archive["initial_raw_proprio_digests"]),
                        "final_raw_visual_digest_array_sha256": _array_sha256(archive["final_raw_visual_digests"]),
                        "final_raw_proprio_digest_array_sha256": _array_sha256(archive["final_raw_proprio_digests"]),
                        "returned_mean_raw_visual_digest_array_sha256": _array_sha256(archive["returned_mean_raw_visual_digests"]),
                        "returned_mean_raw_proprio_digest_array_sha256": _array_sha256(archive["returned_mean_raw_proprio_digests"]),
                        "returned_mean_final_state_sha256": _array_sha256(archive["returned_mean_final_state"]),
                        "returned_mean_native_state_distance_sha256": _array_sha256(archive["returned_mean_native_state_distance"]),
                        "returned_mean_native_success_sha256": _array_sha256(archive["returned_mean_native_success"]),
                        "returned_mean_final_coverage_sha256": _array_sha256(archive["returned_mean_final_coverage"]),
                        "final_states_sha256": _array_sha256(archive["final_states"]),
                        "native_success_sha256": _array_sha256(archive["native_success"]),
                        "final_coverage_sha256": _array_sha256(archive["final_coverage"]),
                    }
                    for field, expected in reported_hashes.items():
                        if artifact.get(field) != expected:
                            errors.append(f"artifact {field} does not match NPZ")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"sidecar hash verification failed: {exc}")
    if strict_schema and sidecar is not None and sidecar.is_file():
        # Recompute the decision from the immutable NPZ rather than trusting
        # the JSON summary.  This is intentionally redundant with the runner:
        # a hand-edited JSON must not become a valid scientific row.
        try:
            from latent_score_decomposition import decompose_latent_scores

            with np.load(sidecar, allow_pickle=False) as archive:
                final_actions = np.asarray(archive["final_actions_normalized"])
                initial_actions = np.asarray(archive["initial_actions_normalized"])
                final_source = np.asarray(archive["final_objective_scores"], dtype=np.float64)
                initial_source = np.asarray(archive["initial_objective_scores"], dtype=np.float64)
                final_native = np.asarray(archive["native_returns"], dtype=np.float64)
                final_dist = np.asarray(archive["native_state_distances"], dtype=np.float64)
                final_decomp = decompose_latent_scores(
                    archive["predicted_visual"], archive["encoded_real_visual"], archive["goal_visual"],
                    predicted_proprio=archive["predicted_proprio"],
                    realized_proprio=archive["encoded_real_proprio"],
                    goal_proprio=archive["goal_proprio"],
                    alpha=float(LOCKED_PROTOCOL["alpha"]),
                )
                pred = np.asarray(final_decomp["predicted_score"], dtype=np.float64)
                obs = np.asarray(final_decomp["observed_score"], dtype=np.float64)
                residual = np.asarray(final_decomp["reconstruction_residual"], dtype=np.float64)
                if not np.allclose(pred, final_source, atol=float(LOCKED_PROTOCOL["score_reproduction_tolerance"]), rtol=0):
                    errors.append("NPZ final objective scores do not reproduce predicted P")
                if (float(np.max(np.abs(residual))) if residual.size else math.inf) > float(LOCKED_PROTOCOL["decomposition_tolerance"]):
                    errors.append("NPZ final latent decomposition identity is not exact")
                rows = candidates if isinstance(candidates, list) else []
                if len(rows) == 300:
                    row_source = np.asarray([row.get("source_planner_score") for row in rows], dtype=np.float64)
                    if not np.array_equal(row_source, final_source):
                        errors.append("candidate source scores do not exactly match final_objective_scores")
                    row_pred = np.asarray([row.get("predicted_score") for row in rows], dtype=np.float64)
                    row_obs = np.asarray([row.get("encoded_real_score") for row in rows], dtype=np.float64)
                    if not np.allclose(row_pred, pred, atol=1e-12, rtol=0):
                        errors.append("candidate predicted scores do not match NPZ decomposition")
                    if not np.allclose(row_obs, obs, atol=1e-12, rtol=0):
                        errors.append("candidate encoded-real scores do not match NPZ decomposition")
                    for index, row in enumerate(rows):
                        if not np.array_equal(np.asarray(row.get("action_model"), dtype=np.float32), final_actions[index]):
                            errors.append(f"candidate {index} action does not match NPZ final action")
                        if row.get("normalized_action_sha256") != _array_sha256(final_actions[index]):
                            errors.append(f"candidate {index} normalized action hash mismatch")
                        if not math.isclose(float(row.get("native_return")), float(final_native[index]), rel_tol=0, abs_tol=1e-12):
                            errors.append(f"candidate {index} native return mismatch")
                        if not math.isclose(float(row.get("native_state_distance")), float(final_dist[index]), rel_tol=0, abs_tol=1e-12):
                            errors.append(f"candidate {index} native distance mismatch")
                        if not np.allclose(np.asarray(row.get("final_state"), dtype=np.float64), np.asarray(archive["final_states"][index], dtype=np.float64), atol=1e-12, rtol=0):
                            errors.append(f"candidate {index} final state mismatch")
                        if bool(row.get("native_success")) != bool(archive["native_success"][index]):
                            errors.append(f"candidate {index} native success mismatch")
                        if not math.isclose(float(row.get("final_coverage")), float(archive["final_coverage"][index]), rel_tol=0, abs_tol=1e-12):
                            errors.append(f"candidate {index} final coverage mismatch")
                        for field, expected in (
                            ("score_reproduction_error", float(pred[index] - final_source[index])),
                            ("directional_error", float(final_decomp["directional_correction"][index])),
                            ("magnitude_error", float(final_decomp["magnitude_correction"][index])),
                            ("decomposition_identity_error", float(residual[index])),
                        ):
                            if not math.isclose(float(row.get(field)), expected, rel_tol=0, abs_tol=1e-12):
                                errors.append(f"candidate {index} {field} mismatch")
                    hashes = [_array_sha256(final_actions[index]) for index in range(300)]
                    unique_indices = list(dict.fromkeys(hashes))
                    unique_indices = [hashes.index(value) for value in unique_indices]
                    if artifact.get("unique_plan_count") != len(unique_indices):
                        errors.append("artifact unique_plan_count does not match NPZ action hashes")
                    if isinstance(decision, Mapping) and decision.get("unique_plan_count") != len(unique_indices):
                        errors.append("decision unique_plan_count does not match NPZ action hashes")
                    for index, row in enumerate(rows):
                        if row.get("candidate_id") != f"sample_{index:04d}":
                            errors.append(f"candidate {index} has a non-canonical candidate_id")
                        if row.get("normalized_plan_multiplicity") != hashes.count(hashes[index]):
                            errors.append(f"candidate {index} normalized-plan multiplicity mismatch")
                    p_local = _strict_choice(pred[unique_indices], lower_better=True)
                    source_local = _strict_choice(final_source[unique_indices], lower_better=True)
                    o_local = _strict_choice(obs[unique_indices], lower_better=True)
                    j_local = _strict_choice(final_native[unique_indices], lower_better=False, allow_tie=True)
                    g_local = _strict_choice(final_dist[unique_indices], lower_better=True, allow_tie=True)
                    picks = {
                        "predicted_choice": p_local,
                        "source_planner_choice": source_local,
                        "encoded_real_choice": o_local,
                        "native_reward_choice": j_local,
                        "goal_aligned_choice": g_local,
                    }
                    ids = [row.get("candidate_id") for row in rows]
                    if isinstance(decision, Mapping):
                        for field, local in picks.items():
                            expected = ids[unique_indices[local]] if local is not None else None
                            if decision.get(field) != expected:
                                errors.append(f"decision {field} does not match NPZ ranking")
                        source_abstained = source_local is None and p_local is None
                        if decision.get("source_recomputed_abstained") is not source_abstained:
                            errors.append("decision source abstention is not matched None/None")
                        if decision.get("source_recomputed_winner_agrees") is not (source_local is not None and p_local is not None and source_local == p_local):
                            errors.append("decision source/recomputed winner agreement is false")
                        best_native = float(np.max(final_native[unique_indices]))
                        best_dist = float(np.min(final_dist[unique_indices]))
                        rp = best_native - float(final_native[unique_indices[p_local]]) if p_local is not None else None
                        ro = best_native - float(final_native[unique_indices[o_local]]) if o_local is not None else None
                        rgp = float(final_dist[unique_indices[p_local]] - best_dist) if p_local is not None else None
                        rgo = float(final_dist[unique_indices[o_local]] - best_dist) if o_local is not None else None
                        expected_values = {"R_P": rp, "R_O": ro, "goal_aligned_R_P": rgp, "goal_aligned_R_O": rgo,
                                           "Delta_reward": rp - ro if rp is not None and ro is not None else None,
                                           "Delta_goal": rgp - rgo if rgp is not None and rgo is not None else None}
                        for field, expected in expected_values.items():
                            actual = decision.get(field)
                            if expected is None:
                                if actual is not None:
                                    errors.append(f"decision {field} should be null under abstention")
                            elif not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
                                errors.append(f"decision {field} does not match NPZ regret")
                        if decision.get("reward_winner_pattern") != _winner_pattern(p_local, o_local, j_local):
                            errors.append("decision reward winner pattern does not match NPZ ranking")
                        if decision.get("goal_aligned_winner_pattern") != _winner_pattern(p_local, o_local, g_local):
                            errors.append("decision goal winner pattern does not match NPZ ranking")
                if strict_schema and isinstance(decision, Mapping):
                    returned_decomp = decompose_latent_scores(
                        archive["returned_predicted_visual"], archive["returned_encoded_real_visual"], archive["goal_visual"],
                        predicted_proprio=archive["returned_predicted_proprio"],
                        realized_proprio=archive["returned_encoded_real_proprio"],
                        goal_proprio=archive["goal_proprio"],
                        alpha=float(LOCKED_PROTOCOL["alpha"]),
                    )
                    returned_p = float(np.asarray(returned_decomp["predicted_score"]).reshape(-1)[0])
                    returned_o = float(np.asarray(returned_decomp["observed_score"]).reshape(-1)[0])
                    returned_residual = float(np.asarray(returned_decomp["reconstruction_residual"]).reshape(-1)[0])
                    returned_source = float(np.asarray(archive["returned_objective_score"]).reshape(-1)[0])
                    returned_action = np.asarray(archive["returned_mean_actions_normalized"], dtype=np.float32)
                    if not np.array_equal(np.asarray(decision.get("returned_mean_action_model"), dtype=np.float32), returned_action):
                        errors.append("returned mean action does not match NPZ")
                    returned_fields = {
                        "returned_mean_predicted_score": returned_p,
                        "returned_mean_observed_score": returned_o,
                        "returned_mean_score_reproduction_error": returned_p - returned_source,
                        "returned_mean_decomposition_identity_error": returned_residual,
                        "returned_mean_native_return": float(np.asarray(archive["returned_native_return"]).reshape(-1)[0]),
                        "returned_mean_native_state_distance": float(np.asarray(archive["returned_mean_native_state_distance"]).reshape(-1)[0]),
                        "returned_mean_native_success": bool(np.asarray(archive["returned_mean_native_success"]).reshape(-1)[0]),
                        "returned_mean_final_coverage": float(np.asarray(archive["returned_mean_final_coverage"]).reshape(-1)[0]),
                    }
                    for field, expected in returned_fields.items():
                        actual = decision.get(field)
                        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
                            errors.append(f"decision {field} does not match NPZ returned-mean data")
                    if not np.allclose(
                        np.asarray(artifact.get("returned_mean_replay_state"), dtype=np.float64),
                        np.asarray(archive["returned_mean_final_state"], dtype=np.float64).reshape(-1),
                        atol=1e-12, rtol=0,
                    ):
                        errors.append("decision returned mean replay state does not match NPZ")
                    if abs(returned_p - returned_source) > float(LOCKED_PROTOCOL["score_reproduction_tolerance"]):
                        errors.append("NPZ returned mean objective score exceeds reproduction tolerance")
                    if not math.isclose(
                        float(decision.get("returned_mean_decomposition_identity_error", math.inf)),
                        returned_residual, rel_tol=0, abs_tol=1e-12,
                    ):
                        errors.append("returned-mean decomposition identity does not match NPZ")
                initial_decomp = decompose_latent_scores(
                    archive["initial_predicted_visual"], archive["initial_encoded_real_visual"], archive["goal_visual"],
                    predicted_proprio=archive["initial_predicted_proprio"],
                    realized_proprio=archive["initial_encoded_real_proprio"],
                    goal_proprio=archive["goal_proprio"],
                    alpha=float(LOCKED_PROTOCOL["alpha"]),
                )
                initial_pred = np.asarray(initial_decomp["predicted_score"], dtype=np.float64)
                initial_obs = np.asarray(initial_decomp["observed_score"], dtype=np.float64)
                initial_residual = np.asarray(initial_decomp["reconstruction_residual"], dtype=np.float64)
                if not np.allclose(initial_pred, initial_source, atol=float(LOCKED_PROTOCOL["score_reproduction_tolerance"]), rtol=0):
                    errors.append("NPZ initial objective scores do not reproduce predicted P")
                if (float(np.max(np.abs(initial_residual))) if initial_residual.size else math.inf) > float(LOCKED_PROTOCOL["decomposition_tolerance"]):
                    errors.append("NPZ initial latent decomposition identity is not exact")
                if isinstance(common_inventory, Mapping):
                    common_hashes = {
                        "normalized_sha256": _array_sha256(archive["initial_actions_normalized"]),
                        "initial_physical_action_sha256": _array_sha256(archive["initial_physical_actions"]),
                        "initial_j_sha256": _array_sha256(archive["initial_native_returns"]),
                        "initial_g_sha256": _array_sha256(-np.asarray(archive["initial_native_state_distances"])),
                        "initial_objective_scores_sha256": _array_sha256(initial_source),
                    }
                    for field, expected in common_hashes.items():
                        if common_inventory.get(field) != expected:
                            errors.append(f"common inventory {field} does not match NPZ")
                    initial_hashes = [_array_sha256(initial_actions[index]) for index in range(300)]
                    initial_unique = list(dict.fromkeys(initial_hashes))
                    initial_unique = [initial_hashes.index(value) for value in initial_unique]
                    initial_p_local = _strict_choice(initial_pred[initial_unique], lower_better=True)
                    initial_source_local = _strict_choice(initial_source[initial_unique], lower_better=True)
                    initial_o_local = _strict_choice(initial_obs[initial_unique], lower_better=True)
                    initial_j_local = _strict_choice(np.asarray(archive["initial_native_returns"])[initial_unique], lower_better=False, allow_tie=True)
                    initial_g_local = _strict_choice(np.asarray(archive["initial_native_state_distances"])[initial_unique], lower_better=True, allow_tie=True)
                    initial_ids = [f"initial_{index:04d}" for index in range(300)]
                    common_picks = {
                        "predicted_choice": initial_p_local,
                        "source_planner_choice": initial_source_local,
                        "encoded_real_choice": initial_o_local,
                        "native_reward_choice": initial_j_local,
                        "goal_aligned_choice": initial_g_local,
                    }
                    for field, local in common_picks.items():
                        expected = initial_ids[initial_unique[local]] if local is not None else None
                        if common_inventory.get(field) != expected:
                            errors.append(f"common inventory {field} does not match NPZ ranking")
                    common_source_abstained = initial_source_local is None and initial_p_local is None
                    if common_inventory.get("source_recomputed_abstained") is not common_source_abstained:
                        errors.append("common source abstention is not matched None/None")
                    if common_inventory.get("source_recomputed_winner_agrees") is not (initial_source_local is not None and initial_p_local is not None and initial_source_local == initial_p_local):
                        errors.append("common source/recomputed winner agreement is false")
                    common_best_native = float(np.max(np.asarray(archive["initial_native_returns"])[initial_unique]))
                    common_best_dist = float(np.min(np.asarray(archive["initial_native_state_distances"])[initial_unique]))
                    common_rp = common_best_native - float(np.asarray(archive["initial_native_returns"])[initial_unique[initial_p_local]]) if initial_p_local is not None else None
                    common_ro = common_best_native - float(np.asarray(archive["initial_native_returns"])[initial_unique[initial_o_local]]) if initial_o_local is not None else None
                    common_rgp = float(np.asarray(archive["initial_native_state_distances"])[initial_unique[initial_p_local]] - common_best_dist) if initial_p_local is not None else None
                    common_rgo = float(np.asarray(archive["initial_native_state_distances"])[initial_unique[initial_o_local]] - common_best_dist) if initial_o_local is not None else None
                    common_values = {
                        "R_P_reward": common_rp, "R_O_reward": common_ro,
                        "R_P_goal": common_rgp, "R_O_goal": common_rgo,
                        "Delta_reward": common_rp - common_ro if common_rp is not None and common_ro is not None else None,
                        "Delta_goal": common_rgp - common_rgo if common_rgp is not None and common_rgo is not None else None,
                    }
                    for field, expected in common_values.items():
                        actual = common_inventory.get(field)
                        if expected is None:
                            if actual is not None:
                                errors.append(f"common inventory {field} should be null under abstention")
                        elif not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
                            errors.append(f"common inventory {field} does not match NPZ regret")
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"independent NPZ decision recomputation failed: {exc}")
    if artifact_path is not None and not artifact_path.is_file():
        errors.append("artifact JSON was not written")
    return errors


def study_estimate(*, decisions: int, iterations: int = 30, samples: int = 300, score_batch_size: int = 8) -> dict[str, Any]:
    """Return exact pinned operation counts and an uncompressed array estimate."""
    score_batches = math.ceil(samples / score_batch_size)
    prediction_calls = iterations * score_batches
    visual_single_bytes = 1 * 16 * 16 * 384 * 4
    proprio_single_bytes = 256 * 16 * 4
    # Each sidecar stores initial/final actions, the returned mean, two
    # candidate latent arrays for each inventory, the goal, and two
    # returned-mean latent arrays.
    latent_array_count = 4 * samples + 3
    visual_bytes = latent_array_count * visual_single_bytes
    proprio_bytes = latent_array_count * proprio_single_bytes
    action_bytes = 2 * samples * 6 * 10 * 4 + 6 * 10 * 4
    scalar_bytes = samples * 8 * 2 + 8
    arrays_bytes = visual_bytes + proprio_bytes + action_bytes + scalar_bytes
    return {
        "decisions": decisions,
        "per_decision": {
            "cem_population_plans": iterations * samples,
            "cem_score_batches": prediction_calls,
            "cem_predicted_model_steps": prediction_calls * 6,
            "common_initial_score_batches": score_batches,
            "common_initial_predicted_model_steps": score_batches * 6,
            "terminal_encode_batches": score_batches,
            "common_initial_terminal_encode_batches": score_batches,
            "initial_and_goal_encode_calls": 2,
            "real_model_chunk_equivalence_calls": 3,
            "returned_mean_encode_calls": 1,
            "replayed_candidate_physical_steps": samples * 30,
            "replayed_common_initial_physical_steps": samples * 30,
            "replayed_repeat_gate_physical_steps": 0,
            "replayed_engineering_certificate_extra_physical_steps": 2 * 30,
            "replayed_returned_mean_physical_steps": 30,
            "replayed_expert_goal_physical_steps": 30,
            "model_calls_estimate": (
                prediction_calls + score_batches + score_batches + score_batches
                + 2 + 3 + 1 + 1
            ),
            "arrays_sidecar_uncompressed_bytes_estimate": arrays_bytes,
            "arrays_sidecar_conservative_storage_gib": round(arrays_bytes / 1024**3, 3),
            "gpu_wall_time": "1261 model calls per confirmatory decision; measure one monitored row before extrapolating",
        },
        "study_sidecar_uncompressed_storage_gib_estimate": round(
            decisions * arrays_bytes / 1024**3, 3
        ),
    }


class ArtifactValidationError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class ScientificIntegrityError(RuntimeError):
    """A result cannot be safely continued, retried, or silently repaired."""


def _paired_inventory_errors(out_dir: Path, entries: list[Mapping[str, Any]]) -> list[str]:
    """Check the shared outcome-blind initial inventory once both model rows exist."""
    errors: list[str] = []
    by_segment: dict[tuple[Any, Any], dict[str, Mapping[str, Any]]] = {}
    for entry in entries:
        key = (entry.get("trajectory_id"), entry.get("offset"))
        artifact_path = out_dir / f"{entry['decision_id']}.json"
        if not artifact_path.is_file():
            continue
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        by_segment.setdefault(key, {})[str(entry.get("model"))] = artifact
    for key, models in by_segment.items():
        if not set(LOCKED_MODELS).issubset(models):
            continue
        if any(models[name].get("schema_version") != "official-pusht-decision-audit-v2" for name in LOCKED_MODELS):
            # Legacy engineering fixtures predate the paired initial-inventory
            # sidecar contract and are not eligible for a confirmatory gate.
            continue
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for model_name in LOCKED_MODELS:
            artifact = models[model_name]
            sidecar = artifact.get("arrays_sidecar")
            if not isinstance(sidecar, str):
                errors.append(f"{key}: {model_name} has no arrays sidecar for paired gate")
                continue
            try:
                with np.load(sidecar, allow_pickle=False) as archive:
                    arrays[model_name] = {
                        "normalized": np.asarray(archive["initial_actions_normalized"]),
                        "physical": np.asarray(archive["initial_physical_actions"]),
                        "J": np.asarray(archive["initial_native_returns"]),
                        "G": -np.asarray(archive["initial_native_state_distances"]),
                        "states": np.asarray(archive["initial_final_states"]),
                        "success": np.asarray(archive["initial_native_success"]),
                        "coverage": np.asarray(archive["initial_final_coverage"]),
                        "raw_terminal_visual": np.asarray(archive["initial_raw_visual_digests"]),
                        "raw_terminal_proprio": np.asarray(archive["initial_raw_proprio_digests"]),
                    }
            except (OSError, KeyError, ValueError) as exc:
                errors.append(f"{key}: paired sidecar unreadable ({exc})")
        if len(arrays) != 2:
            continue
        first, second = arrays[LOCKED_MODELS[0]], arrays[LOCKED_MODELS[1]]
        for label in (
            "normalized", "physical", "J", "G", "states", "success", "coverage",
            "raw_terminal_visual", "raw_terminal_proprio",
        ):
            if not np.array_equal(first[label], second[label]):
                errors.append(f"{key}: paired initial {label} arrays differ between JEPA and DINO")
        for model_name in LOCKED_MODELS:
            inventory = models[model_name].get("initial_inventory", {})
            if inventory.get("physical_sha256") != _array_sha256(arrays[model_name]["physical"]):
                errors.append(f"{key}: {model_name} initial physical hash does not match NPZ")
    return errors


def run_study(
    manifest: Mapping[str, Any],
    out_dir: Path,
    runner: Callable[..., Mapping[str, Any]],
    runner_kwargs: Mapping[str, Any],
    *,
    repeat_count: int = 1,
    certificate: Mapping[str, Any] | None = None,
    certificates: Mapping[str, Mapping[str, Any]] | None = None,
    certificate_sha256: Mapping[str, str] | str | None = None,
    certificate_bundle: Mapping[str, Any] | None = None,
    model_specs: Mapping[str, Mapping[str, Any]] | None = None,
    official_root: Path | None = None,
    manifest_sha256: str | None = None,
    resume_external: bool = False,
    interruption_reason: str | None = None,
    require_full: bool = True,
) -> dict[str, Any]:
    errors = validate_manifest(manifest, require_full=require_full)
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))
    if require_full and repeat_count != 1:
        raise ValueError("full confirmatory rows require repeat_count=1; reproducibility is certified separately")
    if require_full and manifest_sha256 is None:
        manifest_sha256 = _canonical_sha256(manifest)
    if resume_external and not interruption_reason:
        raise ValueError("resume_external requires a documented interruption_reason")
    if certificates is None and certificate is not None:
        certificates = {str(certificate.get("model", "")): certificate}
    if require_full:
        expected_models = {"official_jepa_wm", "official_dino_wm"}
        if certificates is None or set(certificates) != expected_models:
            raise ValueError("confirmatory study requires one validated certificate for each model")
        if model_specs is None or official_root is None:
            raise ValueError("full confirmatory study requires model_specs and official_root")
        if certificate_bundle is None:
            raise ValueError("full confirmatory study requires a v2 certificate bundle")
        binding_errors = validate_manifest_bindings(
            manifest, official_root=Path(official_root or runner_kwargs.get("official_root", "")),
            model_specs=model_specs,
        )
        if binding_errors:
            raise ValueError("invalid manifest bindings: " + "; ".join(binding_errors))
        from experiments.official_pusht_certificate_supervisor import validate_certificate_bundle
        bundle_errors = validate_certificate_bundle(certificate_bundle)
        if bundle_errors:
            raise ValueError("invalid certificate bundle: " + "; ".join(bundle_errors))
        bundle_certificates = certificate_bundle.get("certificates")
        bundle_hashes = certificate_bundle.get("certificate_sha256")
        if not isinstance(bundle_certificates, Mapping) or not isinstance(bundle_hashes, Mapping):
            raise ValueError("certificate bundle lacks exact certificates and hashes")
        if any(
            not isinstance(certificates.get(model_name), Mapping)
            or _canonical_sha256(certificates[model_name]) != _canonical_sha256(bundle_certificates[model_name])
            for model_name in expected_models
        ):
            raise ValueError("passed certificates do not exactly match the certificate bundle")
        if not isinstance(certificate_sha256, Mapping) or {
            str(model): str(value) for model, value in certificate_sha256.items()
        } != {str(model): str(value) for model, value in bundle_hashes.items()}:
            raise ValueError("passed certificate hashes do not exactly match the certificate bundle")
    certificate_hashes: dict[str, str] = {}
    if certificates is not None:
        if isinstance(certificate_sha256, Mapping):
            certificate_hashes.update({str(key): str(value) for key, value in certificate_sha256.items()})
        for model_name, cert in certificates.items():
            if require_full and (
                cert.get("fresh_process_supervisor") is not True
                or cert.get("schema_version") != "PWA-PushT-reproducibility-certificate-v2"
            ):
                raise ValueError("full confirmatory study requires the v2 fresh-process certificate bundle")
            spec = (model_specs or {}).get(model_name, {})
            certificate_errors = validate_reproducibility_certificate(
                cert,
                expected_model=model_name,
                expected_checkpoint_sha256=spec.get("checkpoint_sha256"),
                expected_config_sha256=spec.get("config_sha256"),
                expected_runtime_sha256=(manifest.get("bindings", {}) or {}).get("runtime_sha256"),
                expected_hardware_sha256=(manifest.get("bindings", {}) or {}).get("hardware_sha256"),
            )
            if certificate_errors:
                raise ValueError(f"invalid {model_name} reproducibility certificate: " + "; ".join(certificate_errors))
            if model_name not in certificate_hashes:
                certificate_hashes[model_name] = hashlib.sha256(
                    json.dumps(cert, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                ).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / LEDGER_NAME
    latest = _latest_events(ledger_path)
    counts = {"completed": 0, "skipped": 0, "failed": 0, "interrupted": 0}
    completed_seconds = 0.0
    for entry in manifest["entries"]:
        decision_id = str(entry["decision_id"])
        entry_certificate_hash = certificate_hashes.get(str(entry.get("model")))
        artifact_path = out_dir / f"{decision_id}.json"
        arrays_path = out_dir / f"{decision_id}.npz"
        prior = latest.get(decision_id)
        if prior:
            prior_status = prior.get("status")
            prior_is_terminal_failure = prior_status in {"failed", "interrupted"} or (
                prior_status == "skipped"
                and prior.get("reason") == "terminal_failure_no_implicit_retry"
            )
            prior_requires_existing_artifact = prior_status == "completed" or (
                prior_status == "skipped"
                and prior.get("reason") == "validated_existing_artifact"
            )
            if prior_is_terminal_failure and artifact_path.exists():
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "failed", "failure_kind": "scientific_integrity",
                    "reason": "terminal failure has a promoted artifact; refusing to reopen row",
                })
                counts["failed"] += 1
                raise ScientificIntegrityError(
                    f"terminal failure for {decision_id} has a promoted artifact"
                )
            if prior_requires_existing_artifact and not artifact_path.is_file():
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "failed", "failure_kind": "scientific_integrity",
                    "reason": "completed row artifact is missing; refusing regeneration",
                })
                counts["failed"] += 1
                raise ScientificIntegrityError(
                    f"completed row {decision_id} is missing its artifact; refusing regeneration"
                )
        existing = None
        if artifact_path.is_file():
            try:
                existing = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact_errors = validate_artifact(
                    existing, entry, artifact_path=artifact_path, arrays_path=arrays_path,
                    certificate_sha256=entry_certificate_hash,
                    manifest_sha256=manifest_sha256,
                    require_v2=require_full,
                    expected_bindings=(
                        {
                            **{field: (manifest.get("bindings", {}) or {}).get(field) for field in (
                                "protocol_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256",
                                "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256",
                                "dataset_states_sha256", "dataset_split", "dinov2",
                            )},
                            "model": entry.get("model"),
                            "checkpoint_sha256": ((manifest.get("bindings", {}) or {}).get("models", {}).get(str(entry.get("model")), {}) or {}).get("checkpoint_sha256"),
                            "config_sha256": ((manifest.get("bindings", {}) or {}).get("models", {}).get(str(entry.get("model")), {}) or {}).get("config_sha256"),
                        }
                        if require_full else None
                    ),
                )
            except (OSError, ValueError, TypeError) as exc:
                artifact_errors = [f"cannot load existing artifact: {exc}"]
            if not artifact_errors:
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "skipped", "reason": "validated_existing_artifact",
                })
                counts["skipped"] += 1
                continue
            _append_event(ledger_path, {
                "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                "status": "failed", "failure_kind": "scientific_integrity",
                "errors": artifact_errors, "reason": "existing artifact failed locked validation",
            })
            counts["failed"] += 1
            raise ScientificIntegrityError(
                f"existing artifact failed scientific validation for {decision_id}: {'; '.join(artifact_errors)}"
            )
        retrying_external = False
        prior_retry_count = 0
        if prior:
            try:
                prior_retry_count = int(prior.get("retry_count", 0))
            except (TypeError, ValueError, OverflowError):
                prior_retry_count = 1
            prior_status = prior.get("status")
            prior_failure_kind = prior.get("failure_kind")
            # A process that died after writing ``started`` has not proved why
            # it died.  It becomes retry-eligible only when the operator gives
            # an explicit documented resume reason.  A retry that itself dies
            # is never silently granted a second retry.
            resumable_external = (
                prior_status in {"started", "failed", "interrupted"}
                and prior_retry_count == 0
                and (
                    prior_failure_kind == "external_interruption"
                    or prior_status == "started"
                    or prior_failure_kind in {None, "unclassified_interruption"}
                )
            )
            if resume_external and resumable_external:
                if prior_status == "started" or prior_failure_kind in {None, "unclassified_interruption"}:
                    _append_event(ledger_path, {
                        "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                        "status": "interrupted", "failure_kind": "external_interruption",
                        "reason": interruption_reason,
                        "retry_count": 0,
                    })
                retrying_external = True
            elif prior_status == "started":
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "interrupted", "failure_kind": "unclassified_interruption",
                    "reason": "previous attempt left started; explicit resume reason required",
                    "retry_count": prior_retry_count,
                })
                counts["interrupted"] += 1
                continue
            elif prior_status in {"failed", "interrupted"}:
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "skipped", "reason": "terminal_failure_no_implicit_retry",
                })
                counts["skipped"] += 1
                continue
            elif prior_status == "skipped" and prior.get("reason") == "terminal_failure_no_implicit_retry":
                # A terminal failure remains terminal across every later
                # invocation; a bookkeeping skip must never reopen the row.
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "skipped", "reason": "terminal_failure_no_implicit_retry",
                })
                counts["skipped"] += 1
                continue
        if retrying_external:
            _append_event(ledger_path, {
                "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                "status": "retry_requested", "failure_kind": "external_interruption",
                "reason": interruption_reason,
                "retry_count": 1,
            })
        _append_event(ledger_path, {
            "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
            "status": "started", "episode": entry["episode"], "offset": entry["offset"],
            "retry_count": 1 if retrying_external else 0,
        })
        attempt_started = time.perf_counter()
        try:
            artifact = dict(runner(
                **dict(runner_kwargs), episode=int(entry["episode"]), offset=int(entry["offset"]),
                seed=int(entry["seed"]), arrays_out=arrays_path, repeat_count=repeat_count,
                model_name=entry.get("model"), decision_id=decision_id, planning_repeats=0,
                dataset_split=DATASET_SPLIT, manifest_sha256=manifest_sha256,
            ))
            artifact.setdefault("trajectory_id", entry["trajectory_id"])
            artifact.setdefault("model", entry.get("model"))
            artifact.setdefault("decision_id", decision_id)
            artifact.setdefault("seed", entry.get("seed"))
            artifact.setdefault("dataset_split", DATASET_SPLIT)
            if manifest_sha256 is not None:
                artifact["manifest_sha256"] = manifest_sha256
            if entry_certificate_hash is not None:
                artifact["reproducibility_certificate"] = {
                    "status": "passed", "model": entry["model"], "sha256": entry_certificate_hash,
                }
            # Validate the in-memory result before writing it.  Requiring the
            # JSON path here would make every fresh artifact fail its own
            # pre-write validation.
            artifact_errors = validate_artifact(
                artifact, entry, arrays_path=arrays_path, certificate_sha256=entry_certificate_hash,
                manifest_sha256=manifest_sha256, require_v2=require_full,
                expected_bindings=(
                    {
                        **{field: (manifest.get("bindings", {}) or {}).get(field) for field in (
                            "protocol_sha256", "official_git_commit", "runtime_sha256", "hardware_sha256",
                            "source_bundle_sha256", "dataset_archive_sha256", "seq_lengths_sha256",
                            "dataset_states_sha256", "dataset_split", "dinov2",
                        )},
                        "model": entry.get("model"),
                        "checkpoint_sha256": ((manifest.get("bindings", {}) or {}).get("models", {}).get(str(entry.get("model")), {}) or {}).get("checkpoint_sha256"),
                        "config_sha256": ((manifest.get("bindings", {}) or {}).get("models", {}).get(str(entry.get("model")), {}) or {}).get("config_sha256"),
                    }
                    if require_full else None
                ),
            )
            if artifact_errors:
                raise ArtifactValidationError(artifact_errors)
            _atomic_json_write(artifact_path, artifact)
            pair_errors = _paired_inventory_errors(out_dir, list(manifest["entries"]))
            if pair_errors:
                _append_event(ledger_path, {
                    "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                    "status": "failed", "failure_kind": "scientific_integrity",
                    "errors": pair_errors,
                })
                counts["failed"] += 1
                raise ScientificIntegrityError("paired initial-inventory gate failed: " + "; ".join(pair_errors))
            duration_seconds = time.perf_counter() - attempt_started
            _append_event(ledger_path, {
                "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                "status": "completed", "artifact": str(artifact_path), "arrays": str(arrays_path),
                "duration_seconds": duration_seconds,
            })
            counts["completed"] += 1
            completed_seconds += duration_seconds
        except KeyboardInterrupt as exc:
            duration_seconds = time.perf_counter() - attempt_started
            _append_event(ledger_path, {
                "decision_id": decision_id, "trajectory_id": entry["trajectory_id"],
                "status": "failed", "failure_kind": "external_interruption",
                "reason": interruption_reason or "keyboard interruption; resume requires an explicit reason",
                "retry_count": 1 if retrying_external else 0,
                "error": [str(exc)], "duration_seconds": duration_seconds,
            })
            counts["failed"] += 1
            break
        except Exception as exc:
            message = str(exc).lower()
            integrity_terms = (
                "non-finite", "nonfinite", "out of memory", "score reproduction", "score mismatch",
                "chunk equivalence", "replay", "determin", "hash", "sidecar", "decomposition",
                "paired initial", "scientific integrity",
            )
            if "non-finite" in message or "nonfinite" in message:
                failure_kind = "nonfinite"
            elif "out of memory" in message:
                failure_kind = "oom"
            else:
                failure_kind = "scientific_integrity" if isinstance(exc, (ArtifactValidationError, ScientificIntegrityError)) or any(term in message for term in integrity_terms) else "exception"
            details = getattr(exc, "errors", [str(exc)])
            duration_seconds = time.perf_counter() - attempt_started
            _append_event(ledger_path, {
                "decision_id": decision_id, "trajectory_id": entry["trajectory_id"], "status": "failed", "failure_kind": failure_kind,
                "error": details, "traceback": traceback.format_exc(),
                "retry_count": 1 if retrying_external else 0,
                "duration_seconds": duration_seconds,
            })
            counts["failed"] += 1
            if failure_kind in {"nonfinite", "oom", "scientific_integrity"} and require_full:
                raise ScientificIntegrityError(f"terminal scientific failure for {decision_id}: {exc}") from exc
    return {
        "manifest_entries": len(manifest["entries"]),
        "counts": counts,
        "ledger": str(ledger_path.resolve()),
        "measured_completed_wall_seconds": completed_seconds,
        "estimate": study_estimate(decisions=len(manifest["entries"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--make-manifest", action="store_true")
    parser.add_argument("--make-train-calibration-manifest", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume-external", action="store_true")
    parser.add_argument("--interruption-reason")
    parser.add_argument("--jepa-certificate", type=Path)
    parser.add_argument("--dino-certificate", type=Path)
    parser.add_argument(
        "--certificate-bundle", type=Path,
        help="model-bound certificate_bundle.json emitted by the fresh-process supervisor",
    )
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--jepa-config", type=Path)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--dino-config", type=Path)
    args = parser.parse_args()
    if args.make_manifest or args.make_train_calibration_manifest:
        if args.manifest.exists():
            raise SystemExit("refusing to overwrite an existing immutable manifest")
        if args.make_train_calibration_manifest:
            manifest = make_train_calibration_manifest(args.dataset_root, seed=args.seed)
        else:
            if args.official_root is None or any(path is None for path in (args.jepa_checkpoint, args.jepa_config, args.dino_checkpoint, args.dino_config)):
                raise SystemExit("making the locked manifest requires --official-root and both model checkpoint/config paths")
            model_specs = None
            if all(path is not None for path in (args.jepa_checkpoint, args.jepa_config, args.dino_checkpoint, args.dino_config)):
                model_specs = {
                    "official_jepa_wm": {
                        "checkpoint_sha256": _sha256_file(args.jepa_checkpoint),
                        "config_sha256": _sha256_file(args.jepa_config),
                    },
                    "official_dino_wm": {
                        "checkpoint_sha256": _sha256_file(args.dino_checkpoint),
                        "config_sha256": _sha256_file(args.dino_config),
                    },
                }
            manifest = make_manifest(
                args.dataset_root, seed=args.seed,
                bindings=_manifest_bindings(
                    args.dataset_root, dataset_split=DATASET_SPLIT, seed=args.seed,
                    official_root=args.official_root, model_specs=model_specs,
                ),
            )
        _atomic_json_write(args.manifest, manifest)
        print(json.dumps({"manifest": str(args.manifest.resolve()), "estimate": study_estimate(decisions=len(manifest.get("entries", [])))}, indent=2))
        return
    if not args.manifest.is_file():
        raise SystemExit("manifest does not exist; run with --make-manifest first")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_errors = validate_manifest(manifest, require_full=True)
    if validation_errors:
        raise SystemExit("invalid manifest: " + "; ".join(validation_errors))
    if args.estimate_only:
        print(json.dumps(study_estimate(decisions=len(manifest["entries"])), indent=2))
        return
    if args.out_dir is None or args.official_root is None:
        raise SystemExit("running requires --out-dir and --official-root")
    if args.certificate_bundle is None:
        raise SystemExit("full confirmatory runs require the v2 --certificate-bundle")
    if args.jepa_certificate is not None or args.dino_certificate is not None:
        raise SystemExit("per-model certificate files are legacy; use the v2 --certificate-bundle")
    required_paths = (args.jepa_checkpoint, args.jepa_config, args.dino_checkpoint, args.dino_config)
    if any(path is None for path in required_paths):
        raise SystemExit("running requires both --jepa-checkpoint/--jepa-config and --dino-checkpoint/--dino-config")
    certificate_hashes: dict[str, str]
    if args.certificate_bundle is not None:
        try:
            bundle = json.loads(args.certificate_bundle.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read certificate bundle: {exc}") from exc
        if bundle.get("schema_version") != "PWA-PushT-certificate-bundle-v1" or bundle.get("status") != "passed":
            raise SystemExit("certificate bundle must be a passed v1 supervisor bundle")
        advertised_bundle_hash = bundle.get("bundle_sha256")
        unsigned_bundle = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
        if advertised_bundle_hash != _canonical_sha256(unsigned_bundle):
            raise SystemExit("certificate bundle SHA256 does not match its contents")
        certificates = bundle.get("certificates")
        advertised_hashes = bundle.get("certificate_sha256")
        if not isinstance(certificates, Mapping) or set(certificates) != set(LOCKED_MODELS):
            raise SystemExit("certificate bundle must contain exactly one certificate for each released model")
        if not isinstance(advertised_hashes, Mapping):
            raise SystemExit("certificate bundle lacks model certificate hashes")
        certificate_hashes = {}
        for model_name in LOCKED_MODELS:
            certificate = certificates[model_name]
            actual_hash = _canonical_sha256(certificate)
            if advertised_hashes.get(model_name) != actual_hash:
                raise SystemExit(f"certificate bundle hash mismatch for {model_name}")
            certificate_hashes[model_name] = actual_hash
        from experiments.official_pusht_certificate_supervisor import validate_certificate_bundle
        bundle_errors = validate_certificate_bundle(bundle, bundle_path=args.certificate_bundle)
        if bundle_errors:
            raise SystemExit("invalid certificate bundle: " + "; ".join(bundle_errors))
    from experiments.official_pusht_decision_audit import run_one_decision

    model_paths = {
        "official_jepa_wm": {"checkpoint": args.jepa_checkpoint, "config": args.jepa_config},
        "official_dino_wm": {"checkpoint": args.dino_checkpoint, "config": args.dino_config},
    }

    def run_model_row(**kwargs):
        model_name = kwargs.pop("model_name")
        return run_one_decision(
            official_root=args.official_root, dataset_root=args.dataset_root,
            model_name=model_name, **model_paths[model_name], **kwargs,
        )

    result = run_study(
        manifest, args.out_dir, run_model_row, {},
        repeat_count=1,
        certificates=certificates,
        certificate_sha256=certificate_hashes,
        certificate_bundle=bundle,
        model_specs={
            model_name: {
                "checkpoint_sha256": _sha256_file(paths["checkpoint"]),
                "config_sha256": _sha256_file(paths["config"]),
            }
            for model_name, paths in model_paths.items()
        },
        official_root=args.official_root,
        manifest_sha256=_sha256_file(args.manifest),
        resume_external=args.resume_external,
        interruption_reason=args.interruption_reason,
        require_full=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
