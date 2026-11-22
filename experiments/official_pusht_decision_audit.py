"""Run one auditable official JEPA-WM Push-T planning decision.

The runner intentionally records the final CEM inventory separately from the
planner's returned elite mean.  The latter is an unscored optimizer output in
the upstream implementation, so treating it as one of the 300 samples would
silently mix two different experiments.

This module is importable without the official repository or its optional
environment dependencies.  Those are loaded only by ``run_one_decision``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

try:
    from latent_score_decomposition import decompose_latent_scores
except ModuleNotFoundError:  # Running this file directly puts only experiments/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from latent_score_decomposition import decompose_latent_scores


LOCKED_PROTOCOL = {
    "planner_name": "cem", "objective_type": "L2", "sum_all_diffs": False,
    "alpha": 0.1, "iterations": 30, "samples": 300, "elites": 10,
    "var_scale": 1.0, "horizon": 6, "num_act_stepped": 6, "frameskip": 5,
    "action_skip": 1, "num_frames": 1, "num_proprios": 1, "action_dim": 10,
    "distribute_planner": False,
    "score_reproduction_tolerance": 2e-5,
    "chunk_equivalence_tolerance": 2e-5,
    "tie_atol": 1e-10,
    "tie_rtol": 1e-8,
    "decomposition_tolerance": 1e-8,
}


def _protocol_sha256() -> str:
    return hashlib.sha256(
        json.dumps(LOCKED_PROTOCOL, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _runtime_fingerprint() -> dict[str, Any]:
    packages = sorted(
        f"{dist.metadata.get('Name', '').lower()}=={dist.version}"
        for dist in importlib_metadata.distributions()
        if dist.metadata.get("Name")
    )
    python_executable = Path(sys.executable).resolve()
    return {
        "python": sys.version.split()[0],
        "python_executable": str(python_executable),
        "python_executable_sha256": hashlib.sha256(python_executable.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "installed_distributions_sha256": hashlib.sha256("\n".join(packages).encode("utf-8")).hexdigest(),
        "installed_distribution_count": len(packages),
        "deterministic_flags_requested": {
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "cuda_matmul_tf32": False, "cudnn_tf32": False,
            "deterministic_algorithms": True,
        },
        "deterministic_flags_actual": {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
    }


def _hardware_fingerprint() -> dict[str, Any]:
    result = {"system": platform.system(), "release": platform.release(), "machine": platform.machine()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update({"gpu_name": props.name, "gpu_total_memory_bytes": int(props.total_memory), "gpu_count": int(torch.cuda.device_count())})
    else:
        result.update({"gpu_name": None, "gpu_total_memory_bytes": None, "gpu_count": 0})
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _strict_choice(
    values: np.ndarray, *, lower_better: bool, atol: float, rtol: float,
    allow_tie: bool = False,
) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    best = float(np.min(values) if lower_better else np.max(values))
    ties = np.isclose(values, best, atol=atol, rtol=rtol)
    if allow_tie:
        return int(np.flatnonzero(ties)[0])
    return int(np.flatnonzero(ties)[0]) if int(ties.sum()) == 1 else None


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


def _score_actions_in_chunks(
    unroll: Callable[[Any, Any], Any],
    objective: Callable[[Any, Any], Any],
    z_init: Any,
    actions: Any,
    *,
    batch_size: int,
    keep_terminal: bool = False,
) -> tuple[Any, dict[str, Any] | None]:
    """Score one CEM population in chunks without changing its sampled order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    score_parts, terminal_parts = [], {}
    for start in range(0, actions.shape[1], batch_size):
        batch = actions[:, start : start + batch_size]
        predicted = unroll(z_init, batch)
        score_parts.append(objective(predicted, batch).reshape(-1))
        if keep_terminal:
            for key, value in predicted.items():
                terminal_parts.setdefault(key, []).append(value[-1].detach().clone())
    scores = torch.cat(score_parts, dim=0)
    terminal = {key: torch.cat(parts, dim=0) for key, parts in terminal_parts.items()} if keep_terminal else None
    return scores, terminal


@torch.no_grad()
def capture_cem_inventory(
    unroll: Callable[[Any, Any], Any],
    objective: Callable[[Any, Any], Any],
    z_init: Any,
    *,
    iterations: int = 30,
    num_samples: int = 300,
    num_elites: int = 10,
    horizon: int = 6,
    action_dim: int = 10,
    var_scale: float = 1.0,
    seed: int = 1,
    batch_size: int = 8,
    check_batch_size: int | None = 4,
    equivalence_tolerance: float = LOCKED_PROTOCOL["chunk_equivalence_tolerance"],
) -> dict[str, Any]:
    """Replicate official CEM while retaining its final sampled inventory.

    ``unroll`` and ``objective`` are deliberately injected so the capture seam
    can be unit-tested without importing CUDA, Gym, or the official checkout.
    The return values are detached tensors on the same device as ``z_init``.
    """
    import torch

    if iterations < 1 or num_samples < 2 or not 1 < num_elites <= num_samples:
        raise ValueError("invalid CEM iteration/sample/elite counts")
    device = z_init.device
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.full_like(mean, float(var_scale))
    initial_samples = initial_scores = initial_predicted = None
    final_samples = final_scores = final_predicted = None
    chunk_trace = []

    for iteration in range(iterations):
        actions = mean[:, None, :] + std[:, None, :] * torch.randn(
            horizon, num_samples, action_dim, device=device, generator=generator
        )
        # This is the official planner's mean-sample inclusion trick.
        actions[:, 0, :] = mean
        if iteration == 0:
            # Keep the common, outcome-blind starting population.  This is
            # useful when two model variants are compared with the same seed;
            # it is not part of the final P/O/N decision inventory.
            initial_samples = actions.detach().clone()
        scores, terminal = _score_actions_in_chunks(
            unroll, objective, z_init, actions, batch_size=batch_size,
            keep_terminal=iteration in (0, iterations - 1),
        )
        if scores.numel() != num_samples:
            raise ValueError("objective must return one score per sampled plan")
        if iteration == 0:
            initial_scores = scores.detach().clone()
            initial_predicted = terminal
        if not torch.isfinite(scores).all():
            finite = torch.isfinite(scores)
            finite_scores = scores[finite]
            score_range = (
                (float(finite_scores.min().item()), float(finite_scores.max().item()))
                if finite_scores.numel()
                else None
            )
            raise ValueError(
                "official CEM produced non-finite scores: "
                f"iteration={iteration}, finite={int(finite.sum().item())}/{num_samples}, "
                f"finite_score_range={score_range}, max_abs_action={float(actions.abs().max().item())}, "
                f"max_std={float(std.max().item())}"
            )
        if check_batch_size is not None and check_batch_size != batch_size:
            reference_scores, _ = _score_actions_in_chunks(
                unroll, objective, z_init, actions, batch_size=check_batch_size,
            )
            if not torch.isfinite(reference_scores).all():
                raise ValueError(f"CEM reference chunk score was non-finite at iteration {iteration}")
            score_difference = float(torch.max(torch.abs(scores - reference_scores)).item())
            reference_elites = torch.topk(-reference_scores, num_elites, dim=0).indices
            primary_elites = torch.topk(-scores, num_elites, dim=0).indices
            elite_order_agrees = torch.equal(primary_elites, reference_elites)
            elite_set_agrees = torch.equal(torch.sort(primary_elites).values, torch.sort(reference_elites).values)
            chunk_trace.append({
                "iteration": iteration,
                "population_size": num_samples,
                "primary_batch_size": batch_size,
                "reference_batch_size": check_batch_size,
                "max_abs_score_difference": score_difference,
                "elite_order_agrees": elite_order_agrees,
                "elite_set_agrees": elite_set_agrees,
                "tolerance": equivalence_tolerance,
            })
            if score_difference > equivalence_tolerance or not elite_set_agrees:
                raise RuntimeError(
                    "full-population CEM chunk equivalence failed: "
                    f"iteration={iteration}, max_score_difference={score_difference}, "
                    f"elite_set_agrees={elite_set_agrees}"
                )

        if iteration == iterations - 1:
            final_samples = actions.detach().clone()
            final_scores = scores.detach().clone()
            final_predicted = terminal

        elite_indices = torch.topk(-scores, num_elites, dim=0).indices
        elite_actions = actions[:, elite_indices, :]
        mean = elite_actions.mean(dim=1)
        std = elite_actions.std(dim=1)

    assert initial_samples is not None and initial_scores is not None and initial_predicted is not None
    assert final_samples is not None and final_scores is not None and final_predicted is not None
    returned_actions = mean.detach().clone()
    returned_prediction = unroll(z_init, returned_actions[:, None, :])
    returned_predicted = {
        key: value[-1].detach().clone() for key, value in returned_prediction.items()
    }
    returned_score = float(objective(returned_prediction, returned_actions[:, None, :]).reshape(-1)[0].item())
    return {
        "final_samples": final_samples,
        "initial_samples": initial_samples,
        "initial_scores": initial_scores,
        "initial_predicted": initial_predicted,
        "final_scores": final_scores,
        "final_predicted": final_predicted,
        "returned_actions": returned_actions,
        "returned_predicted": returned_predicted,
        "returned_score": returned_score,
        "iterations": iterations,
        "num_samples": num_samples,
        "num_elites": num_elites,
        "horizon": horizon,
        "action_dim": action_dim,
        "seed": seed,
        "score_batch_size": batch_size,
        "chunk_trace": chunk_trace,
    }


def expand_model_actions(actions: Any, *, frameskip: int, physical_action_dim: int = 2) -> Any:
    """Convert [candidates, model steps, frameskip * action_dim] to physical actions."""
    import torch

    tensor = torch.as_tensor(actions)
    if tensor.ndim != 3 or tensor.shape[-1] != frameskip * physical_action_dim:
        raise ValueError("expected [candidates, steps, frameskip * physical_action_dim] actions")
    return tensor.reshape(tensor.shape[0], tensor.shape[1] * frameskip, physical_action_dim)


def _sha256_bytes(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _digest_array(values: list[Any]) -> np.ndarray:
    """Store per-candidate SHA256 digests as a compact, pickle-free NPZ array."""
    return np.asarray(
        [np.frombuffer(bytes.fromhex(_sha256_bytes(value)), dtype=np.uint8) for value in values],
        dtype=np.uint8,
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bundle_sha256() -> str:
    """Hash the exact study source bundle without importing it at module load."""
    try:
        from experiments.official_pusht_batch_study import study_source_bundle_sha256
    except ModuleNotFoundError:
        from official_pusht_batch_study import study_source_bundle_sha256
    return study_source_bundle_sha256()


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _raw_observation(obs: Any, info: Mapping[str, Any]) -> dict[str, Any]:
    """Turn official PixelWrapper output into model.encode's [B,T,...] input."""
    import torch

    visual = torch.as_tensor(obs)
    proprio = torch.as_tensor(info["proprio"])
    if visual.ndim == 3:
        visual = visual.unsqueeze(0)
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    if visual.ndim != 4 or proprio.ndim != 2:
        raise ValueError(f"unexpected observation shapes: visual={tuple(visual.shape)}, proprio={tuple(proprio.shape)}")
    return {"visual": visual.unsqueeze(0), "proprio": proprio.unsqueeze(0)}


def _encode_replays_in_chunks(model: Any, replays: list[Mapping[str, Any]], chunk_size: int = 8) -> dict[str, Any]:
    """Encode terminal observations in small batches to stay below 8 GB VRAM."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    visual_parts, proprio_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(replays), chunk_size):
            chunk = replays[start : start + chunk_size]
            encoded = model.encode({
                "visual": torch.stack([torch.as_tensor(item["obs"]) for item in chunk]),
                "proprio": torch.stack([torch.as_tensor(item["info"]["proprio"]) for item in chunk]),
            })
            visual_parts.append(encoded["visual"][:, -1].float().cpu())
            proprio_parts.append(encoded["proprio"][:, -1].float().cpu())
    return {"visual": torch.cat(visual_parts), "proprio": torch.cat(proprio_parts)}


def _replay_one(env: Any, seed: int, init_state: np.ndarray, env_info: Mapping[str, Any], actions: Any) -> dict[str, Any]:
    obs, _ = env.prepare(seed, init_state, env_info=env_info)
    obs_list, rewards, _, infos = env.step_multiple(actions)
    if len(obs_list) != len(actions) or not infos:
        raise RuntimeError(f"replay terminated early: expected {len(actions)} steps, got {len(obs_list)}")
    final_info = infos[-1]
    reward_trace = np.asarray(
        [float(torch.as_tensor(reward, dtype=torch.float32).item()) for reward in rewards],
        dtype=np.float32,
    )
    episode_return = torch.zeros((), dtype=torch.float32)
    for reward in reward_trace:
        episode_return += torch.as_tensor(reward, dtype=torch.float32)
    return {
        "obs": obs_list[-1],
        "info": final_info,
        "state": np.asarray(final_info["state"], dtype=np.float64),
        "reward_trace": reward_trace,
        "episode_return": float(episode_return.item()),
    }


def _native_return(env: Any, goal_state: np.ndarray, state: np.ndarray, replay: Mapping[str, Any]) -> dict[str, Any]:
    info = replay["info"]
    evaluation = env.eval_state(goal_state, state)
    return {
        # This matches PlanEvaluator's Push-T episode return: sum the
        # coverage rewards produced by each executed simulator action.
        "native_return": float(replay["episode_return"]),
        "native_state_distance": float(evaluation["state_dist"]),
        "native_success": bool(evaluation["success"]),
        "final_coverage": float(info.get("final_coverage", math.nan)),
    }


def _planning_signature(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the complete seeded CEM outputs for the engineering gate."""
    final = capture["final_samples"].permute(1, 0, 2).detach().cpu().numpy()
    initial = capture["initial_samples"].permute(1, 0, 2).detach().cpu().numpy()
    returned = capture["returned_actions"].detach().cpu().numpy()
    source_scores = capture["final_scores"].detach().cpu().numpy()
    source_local = _strict_choice(source_scores, lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    return {
        "initial_inventory_sha256": _sha256_bytes(initial),
        "final_inventory_sha256": _sha256_bytes(final),
        "returned_mean_sha256": _sha256_bytes(returned),
        "source_scores_sha256": _sha256_bytes(source_scores),
        "source_planner_winner_index": source_local,
    }


def _offline_dinov2_loader(torch_module: Any, source_root: Path, weights: Path) -> Callable[..., Any]:
    """Intercept only the official DINOv2 hub request with local pinned inputs."""
    from experiments.official_pusht_batch_study import (
        DINOV2_LOADED_STATE_DICT_SHA256, DINOV2_MODULE_RELATIVE,
        dinov2_loaded_state_dict_sha256,
    )

    source_root = Path(source_root).resolve()
    weights = Path(weights).resolve()
    original_load = torch_module.hub.load

    def load(repo_or_dir: Any, model_name: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir != "facebookresearch/dinov2":
            return original_load(repo_or_dir, model_name, *args, **kwargs)
        if model_name != "dinov2_vits14":
            raise RuntimeError(f"unexpected DINOv2 hub model: {model_name!r}")
        local_kwargs = dict(kwargs)
        local_kwargs.pop("source", None)
        local_kwargs.pop("pretrained", None)
        # source=local prevents torch.hub from consulting the network or hub
        # cache.  The official request's pretrained default is replaced by the
        # exact released local state dictionary below.
        local_model = original_load(
            str(source_root), model_name, *args, source="local", pretrained=False, **local_kwargs,
        )
        state_dict = torch_module.load(str(weights), map_location="cpu", weights_only=True)
        incompatible = local_model.load_state_dict(state_dict, strict=True)
        module_file = Path(inspect.getfile(type(local_model))).resolve()
        expected_module = (source_root / DINOV2_MODULE_RELATIVE).resolve()
        if module_file != expected_module:
            raise RuntimeError(f"DINOv2 encoder module escaped pinned source tree: {module_file}")
        state_hash = dinov2_loaded_state_dict_sha256(local_model.state_dict())
        if state_hash != DINOV2_LOADED_STATE_DICT_SHA256:
            raise RuntimeError("loaded DINOv2 encoder state dictionary is not the pinned state")
        load.last_binding = {
            "module_file": str(module_file),
            "loaded_state_dict_sha256": state_hash,
            "loaded_model_name": model_name,
            "patch_size": int(local_model.patch_size),
            "feature_dim": int(local_model.num_features),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        return local_model

    load.last_binding = None
    return load


def _load_model(
    official_root: Path, checkpoint: Path, config: Path, *,
    dinov2_source_root: Path | None = None, dinov2_source_archive: Path | None = None,
    dinov2_weights: Path | None = None, torch_home: Path | None = None,
):
    """Load official JEPA-WM with the released model and offline DINOv2 input."""
    import torch
    from experiments.official_pusht_batch_study import (
        DINOV2_SOURCE_ARCHIVE, DINOV2_SOURCE_ROOT, DINOV2_TORCH_HOME, DINOV2_WEIGHTS,
        dinov2_binding, validate_dinov2_bindings,
    )

    official_root = official_root.resolve()
    checkpoint = checkpoint.resolve()
    config = config.resolve()
    if not official_root.is_dir() or not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("official root, checkpoint, and config must exist")
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    dino_binding = dinov2_binding(
        source_root=dinov2_source_root or DINOV2_SOURCE_ROOT,
        source_archive=dinov2_source_archive or DINOV2_SOURCE_ARCHIVE,
        weights=dinov2_weights or DINOV2_WEIGHTS,
        torch_home=torch_home or DINOV2_TORCH_HOME,
    )
    dino_errors = validate_dinov2_bindings(dino_binding, require_complete=True, verify_files=True)
    if dino_errors:
        raise RuntimeError("invalid pinned offline DINOv2 dependency: " + "; ".join(dino_errors))
    torch_home = Path(dino_binding["torch_home"])
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    import yaml
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    model_cfg = raw["model_kwargs"]
    pretrain = copy.deepcopy(model_cfg["pretrain_kwargs"])
    pretrain["heads_cfg"] = {}
    data_cfg = model_cfg["data"]
    aug_cfg = model_cfg["data_aug"]
    stats = get_data_stats("pusht")
    transform = make_transforms(
        img_size=data_cfg["img_size"], normalize=aug_cfg["normalize"],
        random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False, motion_shift=False,
    )
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]), action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]), state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]), proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
    )
    device = torch.device("cuda:0")
    # dino.py calls torch.hub.load("facebookresearch/dinov2", ...).  Guard
    # just that request during official construction; every other hub call
    # retains its original behavior.
    original_hub_load = torch.hub.load
    offline_loader = _offline_dinov2_loader(torch, Path(dino_binding["source_root"]), Path(dino_binding["weights"]))
    torch.hub.load = offline_loader
    try:
        model = init_module(
            folder=checkpoint.parent, checkpoint=checkpoint.name, model_kwargs=pretrain,
            device=device, action_dim=stats["action_dim"], proprio_dim=stats["proprio_dim"],
            preprocessor=preprocessor, cfgs_data=data_cfg, wrapper_kwargs=model_cfg["wrapper_kwargs"],
        )
    finally:
        torch.hub.load = original_hub_load
    if not isinstance(offline_loader.last_binding, Mapping):
        raise RuntimeError("official model did not load the pinned DINOv2 encoder")
    # These two fields are derived from the actual loaded class/state dict,
    # rather than merely copied from a manifest claim.
    dino_binding.update(offline_loader.last_binding)
    dino_errors = validate_dinov2_bindings(dino_binding, require_complete=True, verify_files=True)
    if dino_errors:
        raise RuntimeError("loaded DINOv2 dependency failed binding validation: " + "; ".join(dino_errors))
    model.eval()
    return raw, stats, model, preprocessor, dino_binding


def run_one_decision(
    *,
    official_root: Path,
    checkpoint: Path,
    config: Path,
    dataset_root: Path,
    dataset_split: str = "val",
    episode: int = 0,
    offset: int = 0,
    seed: int = 1,
    arrays_out: Path | None = None,
    repeat_count: int = 3,
    planning_repeats: int = 0,
    inventory_mode: str = "natural_final",
    initial_only: bool = False,
    model_name: str | None = None,
    decision_id: str | None = None,
    trajectory_id: str | None = None,
    manifest_sha256: str | None = None,
    dinov2_source_root: Path | None = None,
    dinov2_source_archive: Path | None = None,
    dinov2_weights: Path | None = None,
    torch_home: Path | None = None,
) -> dict[str, Any]:
    """Run CEM, replay its 300 final samples, and emit one decision audit."""
    # This validation intentionally precedes *all* model imports and loading.
    # The v1.2 invocation-error stop was caused by discovering an invalid
    # planning_repeats value only after an expensive planning capture.
    if isinstance(planning_repeats, bool) or not isinstance(planning_repeats, (int, np.integer)):
        raise ValueError("planning_repeats must be an integer")
    if int(planning_repeats) < 0 or (0 < int(planning_repeats) < 3):
        raise ValueError("planning_repeats must be zero or at least three")
    if inventory_mode not in {"natural_final", "common_initial_only"}:
        raise ValueError("inventory_mode must be natural_final or common_initial_only")
    if initial_only:
        inventory_mode = "common_initial_only"
    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    import torch
    from omegaconf import OmegaConf

    # Freeze the common CUDA sources of run-to-run drift before loading the
    # checkpoint.  If a released operator cannot satisfy deterministic mode,
    # the run fails rather than silently changing the sampled inventory.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms = True
    except (RuntimeError, AttributeError):
        deterministic_algorithms = False
    if not deterministic_algorithms:
        raise RuntimeError("PyTorch deterministic algorithms could not be enabled")

    raw_cfg, stats, model, preprocessor, dinov2_dependency = _load_model(
        official_root, checkpoint, config,
        dinov2_source_root=dinov2_source_root, dinov2_source_archive=dinov2_source_archive,
        dinov2_weights=dinov2_weights, torch_home=torch_home,
    )
    if inventory_mode == "common_initial_only":
        from experiments.official_pusht_v12_common_initial import V12_LOCKED_SEEDS
        if seed not in V12_LOCKED_SEEDS:
            raise ValueError(f"v1.2 menu seed must be one of {V12_LOCKED_SEEDS}")
    elif seed != 1:
        raise ValueError("locked Push-T audit protocol requires seed=1")
    if dataset_split != "val":
        raise ValueError("confirmatory Push-T audit requires dataset_split=val")
    sys.path.insert(0, str(official_root))
    from app.plan_common.datasets.pusht_dset import PushTDataset
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

    dataset = PushTDataset(data_path=str(dataset_root / dataset_split), normalize_action=True, with_velocity=True)
    frameskip = int(raw_cfg["model_kwargs"]["data"]["custom"]["frameskip"])
    horizon = int(raw_cfg["planner"]["horizon"])
    planner_cfg = raw_cfg["planner"]
    objective_cfg = planner_cfg["planning_objective"]
    task_cfg = raw_cfg["task_specification"]
    custom_cfg = raw_cfg["model_kwargs"]["data"]["custom"]
    pinned = {
        "planner_name": planner_cfg.get("planner_name") == "cem",
        "objective_type": objective_cfg.get("objective_type") == "L2",
        "sum_all_diffs": objective_cfg.get("sum_all_diffs") is False,
        "alpha": math.isclose(float(objective_cfg.get("alpha")), 0.1, abs_tol=1e-12),
        "horizon": horizon == 6,
        "num_act_stepped": planner_cfg.get("num_act_stepped") == 6,
        "repeat_actskip": planner_cfg.get("repeat_actskip") is False,
        "frameskip": frameskip == 5,
        "action_skip": custom_cfg.get("action_skip") == 1,
        "num_frames": task_cfg.get("num_frames") == 1,
        "num_proprios": task_cfg.get("num_proprios") == 1,
        "distribute_planner": planner_cfg.get("distribute_planner") is False,
        "iterations": planner_cfg.get("iterations") == 30,
        "num_samples": planner_cfg.get("num_samples") == 300,
        "num_elites": planner_cfg.get("num_elites") == 10,
        "var_scale": math.isclose(float(planner_cfg.get("var_scale")), 1.0, abs_tol=1e-12),
    }
    if not all(pinned.values()):
        raise ValueError(f"config is not the pinned official CEM audit protocol: {pinned}")
    for key in ("max_norms", "max_norm_dims", "momentum_mean", "momentum_std"):
        value = planner_cfg.get(key)
        if value not in (None, 0, 0.0, [], [[]]):
            raise ValueError(f"unsupported non-default CEM option in official config: {key}={value!r}")
    needed = frameskip * horizon + 1
    if not 0 <= episode < len(dataset) or offset + needed > dataset.get_seq_length(episode):
        raise ValueError("selected validation segment is outside the dataset")
    states = dataset.states[episode].numpy()
    actions = dataset.actions[episode]
    init_state = np.asarray(states[offset], dtype=np.float64)
    expert_normalized = actions[offset : offset + frameskip * horizon]
    expert_actions = preprocessor.denormalize_actions(expert_normalized)
    shape = dataset.shapes[episode]
    env_cfg = OmegaConf.create(raw_cfg)
    env_cfg.frameskip = frameskip
    env_cfg.tasks = ["pusht-base"]
    env_cfg.task_specification.multitask = False
    env = make_env(env_cfg)
    env.update_env({"shape": shape})
    env_info = {"shape": shape}
    # v1.2 varies only the Gaussian proposal menu.  The simulator, initial
    # observation, and expert-derived goal always use the released seed 1.
    environment_seed = 1

    # Construct the dset goal exactly as the official evaluator does: replay
    # the selected ground-truth segment, then restore the same initial state.
    expert_rollout = _replay_one(env, environment_seed, init_state, env_info, expert_actions)
    goal_state = expert_rollout["state"]
    goal_raw = _raw_observation(expert_rollout["obs"], expert_rollout["info"])
    init_obs, init_info = env.prepare(environment_seed, init_state, env_info=env_info)
    init_raw = _raw_observation(init_obs, init_info)

    with torch.inference_mode():
        z_init = model.encode(init_raw)
        z_goal = model.encode(goal_raw)
    objective = ReprTargetDistMPCObjective(
        raw_cfg, target_enc=z_goal, sum_all_diffs=False, alpha=float(raw_cfg["planner"]["planning_objective"]["alpha"])
    )
    score_tolerance = float(LOCKED_PROTOCOL["score_reproduction_tolerance"])
    if inventory_mode == "common_initial_only":
        # v1.2 deliberately has no later CEM iterations and no optimizer
        # returned-mean candidate.  It samples/scorers only the shared
        # outcome-blind iteration-0 Gaussian population.
        from experiments.official_pusht_v12_common_initial import capture_initial_only_cem

        captured = capture_initial_only_cem(
            model.unroll, objective, z_init,
            num_samples=int(raw_cfg["planner"]["num_samples"]), horizon=horizon,
            action_dim=int(model.action_dim), var_scale=float(raw_cfg["planner"]["var_scale"]), seed=seed,
            batch_size=8, check_batch_size=4, equivalence_tolerance=score_tolerance,
        )
    else:
        captured = capture_cem_inventory(
            model.unroll, objective, z_init, iterations=int(raw_cfg["planner"]["iterations"]),
            num_samples=int(raw_cfg["planner"]["num_samples"]), num_elites=int(raw_cfg["planner"]["num_elites"]),
            horizon=horizon, action_dim=int(model.action_dim), var_scale=float(raw_cfg["planner"]["var_scale"]), seed=seed,
            batch_size=8, check_batch_size=4, equivalence_tolerance=score_tolerance,
        )
    chunk_trace = captured["chunk_trace"]
    chunk_score_diff = max(item["max_abs_score_difference"] for item in chunk_trace)
    if inventory_mode == "common_initial_only":
        # There are no elites in v1.2: only the iteration-0 winner and bounded
        # score drift are meaningful across scoring chunk sizes.
        chunk_elite_order_agrees = all(item["score_selector_agrees"] for item in chunk_trace)
        chunk_elite_set_agrees = chunk_elite_order_agrees
    else:
        chunk_elite_order_agrees = all(item["elite_order_agrees"] for item in chunk_trace)
        chunk_elite_set_agrees = all(item.get("elite_set_agrees", item["elite_order_agrees"]) for item in chunk_trace)
    if planning_repeats and planning_repeats < 3:
        raise ValueError("planning_repeats must be zero or at least three")
    planning_gate: dict[str, Any]
    if planning_repeats:
        planning_captures = [captured]
        for _ in range(1, planning_repeats):
            if inventory_mode == "common_initial_only":
                from experiments.official_pusht_v12_common_initial import capture_initial_only_cem

                planning_captures.append(
                    capture_initial_only_cem(
                        model.unroll, objective, z_init,
                        num_samples=int(raw_cfg["planner"]["num_samples"]), horizon=horizon,
                        action_dim=int(model.action_dim), var_scale=float(raw_cfg["planner"]["var_scale"]), seed=seed,
                        batch_size=8, check_batch_size=4, equivalence_tolerance=score_tolerance,
                    )
                )
            else:
                planning_captures.append(
                    capture_cem_inventory(
                        model.unroll, objective, z_init,
                        iterations=int(raw_cfg["planner"]["iterations"]),
                        num_samples=int(raw_cfg["planner"]["num_samples"]),
                        num_elites=int(raw_cfg["planner"]["num_elites"]),
                        horizon=horizon, action_dim=int(model.action_dim),
                        var_scale=float(raw_cfg["planner"]["var_scale"]), seed=seed,
                        batch_size=8, check_batch_size=4, equivalence_tolerance=score_tolerance,
                    )
                )
        if inventory_mode == "common_initial_only":
            planning_signatures = [
                {
                    "initial_inventory_sha256": _sha256_bytes(item["initial_samples"].permute(1, 0, 2).detach().cpu().numpy()),
                    "source_scores_sha256": _sha256_bytes(item["initial_scores"].detach().cpu().numpy()),
                }
                for item in planning_captures
            ]
        else:
            planning_signatures = [_planning_signature(item) for item in planning_captures]
        repeated_identically = len({json.dumps(item, sort_keys=True) for item in planning_signatures}) == 1
        planning_gate = {
            "status": "passed" if repeated_identically else "failed",
            "passed": repeated_identically,
            "repeat_count": planning_repeats,
            "initial_inventory_sha256": [item["initial_inventory_sha256"] for item in planning_signatures],
            "source_scores_sha256": [item["source_scores_sha256"] for item in planning_signatures],
        }
        if inventory_mode != "common_initial_only":
            planning_gate.update({
                "final_inventory_sha256": [item["final_inventory_sha256"] for item in planning_signatures],
                "returned_mean_sha256": [item["returned_mean_sha256"] for item in planning_signatures],
                "source_planner_winner_index": [item["source_planner_winner_index"] for item in planning_signatures],
            })
        if not planning_gate["passed"]:
            raise RuntimeError("repeated seeded CEM planning outputs were not identical")
    else:
        planning_gate = {
            "status": "external_certificate_required",
            "passed": False,
            "repeat_count": 0,
        }
    if inventory_mode == "common_initial_only":
        planning_gate["chunk_equivalence"] = {
            "passed": bool(chunk_elite_order_agrees and chunk_score_diff <= score_tolerance),
            "max_abs_score_difference": float(chunk_score_diff),
            "selector_agrees": bool(chunk_elite_order_agrees),
            "trace": chunk_trace,
        }
    if int(model.action_dim) != 10:
        raise ValueError(f"pinned Push-T protocol expects model action width 10, got {model.action_dim}")

    initial_model = captured["initial_samples"].permute(1, 0, 2).cpu()
    final_model = (
        initial_model
        if inventory_mode == "common_initial_only"
        else captured["final_samples"].permute(1, 0, 2).cpu()
    )
    initial_physical = expand_model_actions(initial_model, frameskip=frameskip)
    initial_physical = preprocessor.denormalize_actions(initial_physical)
    initial_scores_tensor = captured["initial_scores"]
    initial_predicted = captured["initial_predicted"]
    initial_replays = []
    for candidate_actions in initial_physical:
        replay = _replay_one(env, environment_seed, init_state, env_info, candidate_actions)
        replay.update(_native_return(env, goal_state, replay["state"], replay))
        initial_replays.append(replay)
    initial_raw_visual_digests = _digest_array([item["obs"] for item in initial_replays])
    initial_raw_proprio_digests = _digest_array([item["info"]["proprio"] for item in initial_replays])
    initial_encoded_real = _encode_replays_in_chunks(model, initial_replays, chunk_size=8)
    initial_pred_v = initial_predicted["visual"].float().cpu().numpy()
    initial_pred_p = initial_predicted["proprio"].float().cpu().numpy()
    initial_real_v = initial_encoded_real["visual"].numpy()
    initial_real_p = initial_encoded_real["proprio"].numpy()
    goal_v = z_goal["visual"][:, -1].float().cpu().numpy()
    goal_p = z_goal["proprio"][:, -1].float().cpu().numpy()
    raw_terminal_visual = np.asarray([item["obs"] for item in initial_replays], dtype=np.uint8)
    raw_terminal_proprio = np.asarray(
        [item["info"]["proprio"] for item in initial_replays], dtype=np.float32
    )
    raw_initial_visual = np.asarray(init_obs, dtype=np.uint8)
    raw_initial_proprio = np.asarray(init_info["proprio"], dtype=np.float32)
    raw_goal_visual = np.asarray(expert_rollout["obs"], dtype=np.uint8)
    raw_goal_proprio = np.asarray(expert_rollout["info"]["proprio"], dtype=np.float32)
    # Re-encode the exact lossless arrays that will be serialized through the
    # frozen production path, then separately test decision stability under a
    # different chunk size.
    retained_replays = [
        {"obs": raw_terminal_visual[index], "info": {"proprio": raw_terminal_proprio[index]}}
        for index in range(len(raw_terminal_visual))
    ]
    retained_encoded = _encode_replays_in_chunks(model, retained_replays, chunk_size=8)
    cross_chunk_encoded = _encode_replays_in_chunks(model, retained_replays, chunk_size=4)
    retained_goal = _encode_replays_in_chunks(
        model, [{"obs": raw_goal_visual, "info": {"proprio": raw_goal_proprio}}], chunk_size=1
    )
    retained_initial = _encode_replays_in_chunks(
        model, [{"obs": raw_initial_visual, "info": {"proprio": raw_initial_proprio}}], chunk_size=1
    )
    reencode_differences = {
        "terminal_visual": float(np.max(np.abs(retained_encoded["visual"].numpy() - initial_real_v))),
        "terminal_proprio": float(np.max(np.abs(retained_encoded["proprio"].numpy() - initial_real_p))),
        "goal_visual": float(np.max(np.abs(retained_goal["visual"].numpy() - goal_v))),
        "goal_proprio": float(np.max(np.abs(retained_goal["proprio"].numpy() - goal_p))),
        "initial_visual": float(np.max(np.abs(retained_initial["visual"].numpy() - z_init["visual"][:, -1].float().cpu().numpy()))),
        "initial_proprio": float(np.max(np.abs(retained_initial["proprio"].numpy() - z_init["proprio"][:, -1].float().cpu().numpy()))),
    }
    production_latents_exact = all((
        np.array_equal(retained_encoded["visual"].numpy(), initial_real_v),
        np.array_equal(retained_encoded["proprio"].numpy(), initial_real_p),
        np.array_equal(retained_goal["visual"].numpy(), goal_v),
        np.array_equal(retained_goal["proprio"].numpy(), goal_p),
        np.array_equal(retained_initial["visual"].numpy(), z_init["visual"][:, -1].float().cpu().numpy()),
        np.array_equal(retained_initial["proprio"].numpy(), z_init["proprio"][:, -1].float().cpu().numpy()),
    ))
    if not production_latents_exact:
        raise RuntimeError(f"retained raw evidence did not reproduce encoded latents: {reencode_differences}")
    cross_chunk_latent_differences = {
        "terminal_visual": float(np.max(np.abs(cross_chunk_encoded["visual"].numpy() - initial_real_v))),
        "terminal_proprio": float(np.max(np.abs(cross_chunk_encoded["proprio"].numpy() - initial_real_p))),
    }
    raw_reencoding_gate = {
        "passed": False, "source": "lossless serialized initial/endpoint/goal arrays",
        "production_chunk_size": 8, "cross_check_chunk_size": 4,
        "production_latents_exact": production_latents_exact,
        "max_abs_differences": reencode_differences,
        "cross_chunk_latent_max_abs_differences_descriptive": cross_chunk_latent_differences,
        "terminal_visual_sha256": _sha256_bytes(raw_terminal_visual),
        "terminal_proprio_sha256": _sha256_bytes(raw_terminal_proprio),
        "goal_visual_sha256": _sha256_bytes(raw_goal_visual),
        "goal_proprio_sha256": _sha256_bytes(raw_goal_proprio),
        "initial_visual_sha256": _sha256_bytes(raw_initial_visual),
        "initial_proprio_sha256": _sha256_bytes(raw_initial_proprio),
        "realized_visual_sha256": _sha256_bytes(initial_real_v),
        "realized_proprio_sha256": _sha256_bytes(initial_real_p),
        "encoded_goal_visual_sha256": _sha256_bytes(goal_v),
        "encoded_goal_proprio_sha256": _sha256_bytes(goal_p),
    }
    initial_decomp = decompose_latent_scores(
        initial_pred_v, initial_real_v, goal_v,
        predicted_proprio=initial_pred_p, realized_proprio=initial_real_p, goal_proprio=goal_p,
        alpha=float(raw_cfg["planner"]["planning_objective"]["alpha"]),
    )
    initial_predicted_scores = np.asarray(initial_decomp["predicted_score"], dtype=np.float64)
    initial_observed_scores = np.asarray(initial_decomp["observed_score"], dtype=np.float64)
    retained_decomp = decompose_latent_scores(
        initial_pred_v, retained_encoded["visual"].numpy(), retained_goal["visual"].numpy(),
        predicted_proprio=initial_pred_p,
        realized_proprio=retained_encoded["proprio"].numpy(),
        goal_proprio=retained_goal["proprio"].numpy(),
        alpha=float(raw_cfg["planner"]["planning_objective"]["alpha"]),
    )
    retained_observed_scores = np.asarray(retained_decomp["observed_score"], dtype=np.float64)
    cross_chunk_decomp = decompose_latent_scores(
        initial_pred_v, cross_chunk_encoded["visual"].numpy(), retained_goal["visual"].numpy(),
        predicted_proprio=initial_pred_p,
        realized_proprio=cross_chunk_encoded["proprio"].numpy(),
        goal_proprio=retained_goal["proprio"].numpy(),
        alpha=float(raw_cfg["planner"]["planning_objective"]["alpha"]),
    )
    cross_chunk_observed_scores = np.asarray(cross_chunk_decomp["observed_score"], dtype=np.float64)
    retained_o_drift = float(np.max(np.abs(retained_observed_scores - initial_observed_scores)))
    retained_o_exact = bool(np.array_equal(retained_observed_scores, initial_observed_scores))
    initial_o_full = _strict_choice(
        initial_observed_scores, lower_better=True,
        atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"],
    )
    retained_o_full = _strict_choice(
        retained_observed_scores, lower_better=True,
        atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"],
    )
    cross_chunk_o_full = _strict_choice(
        cross_chunk_observed_scores, lower_better=True,
        atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"],
    )
    cross_chunk_o_drift = float(np.max(np.abs(cross_chunk_observed_scores - initial_observed_scores)))
    raw_reencoding_gate.update({
        "O_score_max_abs_difference": retained_o_drift,
        "O_scores_exact": retained_o_exact,
        "O_selector_agrees": retained_o_full == initial_o_full,
        "O_choice_index": initial_o_full,
        "cross_chunk_O_score_max_abs_difference": cross_chunk_o_drift,
        "cross_chunk_O_selector_agrees": cross_chunk_o_full == initial_o_full,
        "passed": (
            retained_o_exact and retained_o_full == initial_o_full
            and cross_chunk_o_drift <= score_tolerance and cross_chunk_o_full == initial_o_full
        ),
    })
    if raw_reencoding_gate["passed"] is not True:
        raise RuntimeError(f"retained raw evidence did not reproduce O scores/selector: {raw_reencoding_gate}")
    initial_source_scores = initial_scores_tensor.cpu().numpy().astype(np.float64)
    initial_score_errors = initial_predicted_scores - initial_source_scores
    if float(np.max(np.abs(initial_score_errors))) > score_tolerance:
        raise RuntimeError(
            f"initial CEM score reproduction exceeded tolerance: {float(np.max(np.abs(initial_score_errors)))} > {score_tolerance}"
        )
    initial_decomp_error = float(np.max(np.abs(np.asarray(initial_decomp["reconstruction_residual"]))))
    if initial_decomp_error > LOCKED_PROTOCOL["decomposition_tolerance"]:
        raise RuntimeError(f"initial latent decomposition identity exceeded tolerance: {initial_decomp_error}")
    initial_native = np.asarray([item["native_return"] for item in initial_replays], dtype=np.float64)
    initial_goal_distances = np.asarray([item["native_state_distance"] for item in initial_replays], dtype=np.float64)
    initial_hashes = [_sha256_bytes(initial_model[index].numpy()) for index in range(len(initial_model))]
    initial_unique_indices = list(dict.fromkeys(initial_hashes))
    initial_unique_indices = [initial_hashes.index(item) for item in initial_unique_indices]
    initial_p_local = _strict_choice(initial_predicted_scores[initial_unique_indices], lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    initial_source_local = _strict_choice(initial_source_scores[initial_unique_indices], lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    initial_o_local = _strict_choice(initial_observed_scores[initial_unique_indices], lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    initial_j_local = _strict_choice(initial_native[initial_unique_indices], lower_better=False, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"], allow_tie=True)
    initial_g_local = _strict_choice(initial_goal_distances[initial_unique_indices], lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"], allow_tie=True)
    initial_p_idx = initial_unique_indices[initial_p_local] if initial_p_local is not None else None
    initial_source_idx = initial_unique_indices[initial_source_local] if initial_source_local is not None else None
    initial_o_idx = initial_unique_indices[initial_o_local] if initial_o_local is not None else None
    initial_j_idx = initial_unique_indices[initial_j_local] if initial_j_local is not None else None
    initial_g_idx = initial_unique_indices[initial_g_local] if initial_g_local is not None else None
    initial_best_native = float(np.max(initial_native[initial_unique_indices]))
    initial_best_goal_distance = float(np.min(initial_goal_distances[initial_unique_indices]))
    initial_ids = [f"initial_{index:04d}" for index in range(len(initial_model))]
    common_inventory = {
        "outcome_blind": True,
        "count": int(len(initial_model)),
        "unique_plan_count": int(len(initial_unique_indices)),
        "normalized_sha256": _sha256_bytes(initial_model.numpy()),
        "physical_sha256": _sha256_bytes(initial_physical.numpy()),
        "objective_scores_sha256": _sha256_bytes(initial_source_scores),
        "J_sha256": _sha256_bytes(initial_native),
        "G_sha256": _sha256_bytes(-initial_goal_distances),
        "predicted_choice": initial_ids[initial_p_idx] if initial_p_idx is not None else None,
        "source_planner_choice": initial_ids[initial_source_idx] if initial_source_idx is not None else None,
        "source_recomputed_winner_agrees": initial_source_idx is not None and initial_p_idx is not None and initial_source_idx == initial_p_idx,
        "source_recomputed_abstained": initial_source_idx is None and initial_p_idx is None,
        "captured_score_source": "cem_iteration_0_objective_scores",
        "initial_captured_winner_agrees": initial_source_idx is not None and initial_p_idx is not None and initial_source_idx == initial_p_idx,
        "encoded_real_choice": initial_ids[initial_o_idx] if initial_o_idx is not None else None,
        "native_reward_choice": initial_ids[initial_j_idx] if initial_j_idx is not None else None,
        "goal_aligned_choice": initial_ids[initial_g_idx] if initial_g_idx is not None else None,
        "reward_comparison_abstained": initial_p_idx is None or initial_o_idx is None,
        "goal_comparison_abstained": initial_p_idx is None or initial_o_idx is None,
        "R_P_reward": initial_best_native - float(initial_native[initial_p_idx]) if initial_p_idx is not None else None,
        "R_O_reward": initial_best_native - float(initial_native[initial_o_idx]) if initial_o_idx is not None else None,
        "R_P_goal": float(initial_goal_distances[initial_p_idx] - initial_best_goal_distance) if initial_p_idx is not None else None,
        "R_O_goal": float(initial_goal_distances[initial_o_idx] - initial_best_goal_distance) if initial_o_idx is not None else None,
        "Delta_reward": (
            initial_native[initial_o_idx] - initial_native[initial_p_idx]
            if initial_p_idx is not None and initial_o_idx is not None else None
        ),
        "Delta_goal": (
            initial_goal_distances[initial_p_idx] - initial_goal_distances[initial_o_idx]
            if initial_p_idx is not None and initial_o_idx is not None else None
        ),
        "delta_definition": "Delta = R_P - R_O",
        "G_definition": "negative official final state distance",
        "J_definition": "float32 sum of 30 official coverage rewards",
        "native_best_return": initial_best_native,
        "goal_aligned_best_distance": initial_best_goal_distance,
        "predicted_score_max_abs_error": float(np.max(np.abs(initial_predicted_scores - initial_source_scores))),
        "initial_score_reproduction_max_abs_error": float(np.max(np.abs(initial_score_errors))),
        "decomposition_max_abs_error": initial_decomp_error,
        "initial_j_sha256": _sha256_bytes(initial_native),
        "initial_g_sha256": _sha256_bytes(-initial_goal_distances),
        "initial_physical_action_sha256": _sha256_bytes(initial_physical.numpy()),
        "initial_objective_scores_sha256": _sha256_bytes(initial_source_scores),
        "initial_raw_visual_digest_array_sha256": _sha256_bytes(initial_raw_visual_digests),
        "initial_raw_proprio_digest_array_sha256": _sha256_bytes(initial_raw_proprio_digests),
        "initial_terminal_observation_sha256": _canonical_sha256({
            "visual": _sha256_bytes(initial_real_v),
            "proprio": _sha256_bytes(initial_real_p),
        }),
    }
    if inventory_mode == "common_initial_only":
        # v1.2 stops here: these are the only 300 candidate replays.  The
        # optimizer's returned mean and every post-iteration population are
        # intentionally absent from the artifact.
        from experiments.official_pusht_v12_common_initial import (
            build_v12_artifact, proposal_support_diagnostics,
        )

        underlying_env = getattr(env, "unwrapped", env)
        action_scale = float(getattr(underlying_env, "action_scale", 100.0))
        relative_actions = bool(getattr(underlying_env, "relative", True))
        noop_target = np.zeros(2, dtype=np.float32) if relative_actions else init_state[:2] / action_scale
        noop_normalized = (
            torch.as_tensor(noop_target, dtype=torch.float32) - preprocessor.action_mean
        ) / preprocessor.action_std
        noop_physical = preprocessor.denormalize_actions(noop_normalized.repeat(frameskip * horizon, 1))
        noop_replay = _replay_one(env, environment_seed, init_state, env_info, noop_physical)
        noop_replay.update(_native_return(env, goal_state, noop_replay["state"], noop_replay))
        expert_outcome = _native_return(env, goal_state, goal_state, expert_rollout)
        # Use only frozen action support; no native outcome enters this
        # diagnostic.  The released segment is one physical 30-step sequence.
        released_val_reference = np.asarray(expert_normalized.numpy(), dtype=np.float32).reshape(1, -1)
        support = proposal_support_diagnostics(
            initial_model.numpy(), initial_physical.numpy(),
            released_val_actions=released_val_reference,
        )
        repeats = [initial_replays[0]]
        for _ in range(1, repeat_count):
            repeat = _replay_one(env, environment_seed, init_state, env_info, initial_physical[0])
            repeat.update(_native_return(env, goal_state, repeat["state"], repeat))
            repeats.append(repeat)
        repeat_state_hashes = [_sha256_bytes(item["state"]) for item in repeats]
        repeat_obs_hashes = [_sha256_bytes(item["obs"]) for item in repeats]
        repeat_returns = [item["episode_return"] for item in repeats]
        if len(set(repeat_state_hashes)) != 1 or len(set(repeat_obs_hashes)) != 1 or len(set(repeat_returns)) != 1:
            raise RuntimeError("v1.2 replay repeatability check failed")
        baselines = {
            "expert_goal_defining_segment": {
                "J": expert_outcome["native_return"], "G": -expert_outcome["native_state_distance"],
                "native_return": expert_outcome["native_return"],
                "native_state_distance": expert_outcome["native_state_distance"],
            },
            "hold_position_noop": {
                "J": noop_replay["native_return"], "G": -noop_replay["native_state_distance"],
                "native_return": noop_replay["native_return"],
                "native_state_distance": noop_replay["native_state_distance"],
                "physical_action_sha256": _sha256_bytes(noop_physical.numpy()),
            },
            "repeatability": {
                "passed": repeat_count >= 3, "repeat_count": repeat_count,
                "state_sha256": repeat_state_hashes, "terminal_observation_sha256": repeat_obs_hashes,
                "cumulative_reward": repeat_returns,
            },
        }
        artifact = build_v12_artifact(
            model=model_name or ("official_dino_wm" if "dino" in (checkpoint.name + config.name).lower() else "official_jepa_wm"),
            decision_id=decision_id or f"v12_pusht_val_episode_{episode:03d}_offset_{offset:03d}",
            episode=episode, offset=offset, seed=seed, trajectory_id=trajectory_id,
            normalized_actions=initial_model.numpy(), physical_actions=initial_physical.numpy(),
            predicted_visual=initial_pred_v, realized_visual=initial_real_v, goal_visual=goal_v,
            predicted_proprio=initial_pred_p, realized_proprio=initial_real_p, goal_proprio=goal_p,
            objective_scores=initial_source_scores, native_returns=initial_native,
            native_state_distances=initial_goal_distances,
            final_states=np.asarray([item["state"] for item in initial_replays], dtype=np.float64),
            goal_state=np.asarray(goal_state, dtype=np.float64),
            reward_traces=np.asarray([item["reward_trace"] for item in initial_replays], dtype=np.float32),
            expert_reward_trace=expert_rollout["reward_trace"],
            noop_physical_actions=noop_physical.numpy(),
            noop_final_state=noop_replay["state"],
            noop_reward_trace=noop_replay["reward_trace"],
            raw_visual_observations=raw_terminal_visual,
            raw_proprio_observations=raw_terminal_proprio,
            raw_initial_visual=raw_initial_visual, raw_initial_proprio=raw_initial_proprio,
            raw_goal_visual=raw_goal_visual, raw_goal_proprio=raw_goal_proprio,
            raw_reencoding_gate=raw_reencoding_gate,
            raw_visual_digests=initial_raw_visual_digests, raw_proprio_digests=initial_raw_proprio_digests,
            arrays_out=arrays_out, proposal_diagnostics=support, baselines=baselines,
            planning_gate=planning_gate,
            bindings={
                "runtime": _runtime_fingerprint(), "hardware": _hardware_fingerprint(),
                "runtime_sha256": _canonical_sha256(_runtime_fingerprint()),
                "hardware_sha256": _canonical_sha256(_hardware_fingerprint()),
                "official_git_commit": _git_commit(official_root),
                "official_root": str(official_root.resolve()),
                "checkpoint_sha256": _sha256_file(checkpoint), "config_sha256": _sha256_file(config),
                "dataset_root": str(dataset_root.resolve()), "dataset_split": dataset_split,
                "manifest_sha256": manifest_sha256,
                "dataset_archive_sha256": _sha256_file(official_root / "pusht_noise.zip"),
                "seq_lengths_sha256": _sha256_file(dataset_root / dataset_split / "seq_lengths.pkl"),
                "dataset_states_sha256": _sha256_file(dataset_root / dataset_split / "states.pth"),
                "environment_seed": environment_seed,
                "initial_state_sha256": _sha256_bytes(init_state),
                "goal_state_sha256": _sha256_bytes(goal_state),
                "initial_observation_sha256": _canonical_sha256({"visual": _sha256_bytes(init_obs), "proprio": _sha256_bytes(init_info["proprio"])}),
                "goal_observation_sha256": _canonical_sha256({"visual": _sha256_bytes(expert_rollout["obs"]), "proprio": _sha256_bytes(expert_rollout["info"]["proprio"])}),
                "states_segment_sha256": _sha256_bytes(np.asarray(states[offset : offset + needed])),
                "actions_segment_sha256": _sha256_bytes(np.asarray(expert_normalized)),
                "dinov2": dinov2_dependency,
            },
        )
        artifact["audit_status"] = "v1.2_common_initial_only"
        artifact["terminal_observation_sha256"] = common_inventory["initial_terminal_observation_sha256"]
        return _json_value(artifact)
    returned_model = captured["returned_actions"].unsqueeze(0).cpu()
    final_physical = expand_model_actions(final_model, frameskip=frameskip)
    returned_physical = expand_model_actions(returned_model, frameskip=frameskip)[0]
    final_physical = preprocessor.denormalize_actions(final_physical)
    returned_physical = preprocessor.denormalize_actions(returned_physical)

    replays = []
    for candidate_actions in final_physical:
        replay = _replay_one(env, environment_seed, init_state, env_info, candidate_actions)
        replay.update(_native_return(env, goal_state, replay["state"], replay))
        replays.append(replay)
    repeats = [replays[0]]
    for _ in range(1, repeat_count):
        repeat = _replay_one(env, environment_seed, init_state, env_info, final_physical[0])
        repeat.update(_native_return(env, goal_state, repeat["state"], repeat))
        repeats.append(repeat)
    repeat_state_hashes = [_sha256_bytes(item["state"]) for item in repeats]
    repeat_obs_hashes = [_sha256_bytes(item["obs"]) for item in repeats]
    repeat_returns = [item["episode_return"] for item in repeats]
    if len(set(repeat_state_hashes)) != 1 or len(set(repeat_obs_hashes)) != 1 or len(set(repeat_returns)) != 1:
        raise RuntimeError("replaying candidate zero from the same state was not deterministic")
    returned_replay = _replay_one(env, environment_seed, init_state, env_info, returned_physical)
    returned_replay.update(_native_return(env, goal_state, returned_replay["state"], returned_replay))
    # Preserve raw simulator terminal evidence independently of model-specific
    # encodings.  These compact digests make every final candidate and the
    # optimizer's returned mean checkable in the NPZ sidecar.
    final_raw_visual_digests = _digest_array([item["obs"] for item in replays])
    final_raw_proprio_digests = _digest_array([item["info"]["proprio"] for item in replays])
    returned_mean_raw_visual_digests = _digest_array([returned_replay["obs"]])
    returned_mean_raw_proprio_digests = _digest_array([returned_replay["info"]["proprio"]])

    # Descriptive competence controls.  They are replayed from the same frozen
    # initial state and never enter the P/O/J/G winner or denominator.
    expert_outcome = _native_return(env, goal_state, goal_state, expert_rollout)
    underlying_env = getattr(env, "unwrapped", env)
    action_scale = float(getattr(underlying_env, "action_scale", 100.0))
    relative_actions = bool(getattr(underlying_env, "relative", True))
    # Push-T's released dataset uses relative targets.  If an alternate
    # official wrapper exposes absolute PD targets, the same no-op is the
    # initial agent position expressed in the wrapper's action scale.
    noop_target = np.zeros(2, dtype=np.float32) if relative_actions else init_state[:2] / action_scale
    noop_normalized = (
        torch.as_tensor(noop_target, dtype=torch.float32) - preprocessor.action_mean
    ) / preprocessor.action_std
    noop_physical = preprocessor.denormalize_actions(
        noop_normalized.repeat(frameskip * horizon, 1)
    )
    noop_replay = _replay_one(env, environment_seed, init_state, env_info, noop_physical)
    noop_replay.update(_native_return(env, goal_state, noop_replay["state"], noop_replay))

    returned_visual = torch.as_tensor(returned_replay["obs"]).unsqueeze(0)
    returned_proprio = torch.as_tensor(returned_replay["info"]["proprio"]).unsqueeze(0)
    encoded_real = _encode_replays_in_chunks(model, replays, chunk_size=8)
    with torch.inference_mode():
        encoded_returned = model.encode({"visual": returned_visual, "proprio": returned_proprio})

    pred_v = captured["final_predicted"]["visual"].float().cpu().numpy()
    pred_p = captured["final_predicted"]["proprio"].float().cpu().numpy()
    real_v = encoded_real["visual"].numpy()
    real_p = encoded_real["proprio"].numpy()
    goal_v = z_goal["visual"][:, -1].float().cpu().numpy()
    goal_p = z_goal["proprio"][:, -1].float().cpu().numpy()
    decomp = decompose_latent_scores(
        pred_v, real_v, goal_v, predicted_proprio=pred_p, realized_proprio=real_p, goal_proprio=goal_p,
        alpha=float(raw_cfg["planner"]["planning_objective"]["alpha"]),
    )
    predicted_scores = np.asarray(decomp["predicted_score"], dtype=np.float64)
    observed_scores = np.asarray(decomp["observed_score"], dtype=np.float64)
    source_scores = captured["final_scores"].cpu().numpy().astype(np.float64)
    score_errors = predicted_scores - source_scores
    if float(np.max(np.abs(score_errors))) > score_tolerance:
        raise RuntimeError(
            f"CEM score reproduction exceeded tolerance: {float(np.max(np.abs(score_errors)))} > {score_tolerance}"
        )
    native = np.asarray([item["native_return"] for item in replays], dtype=np.float64)
    goal_distances = np.asarray(
        [item["native_state_distance"] for item in replays], dtype=np.float64
    )
    normalized_hashes = [_sha256_bytes(final_model[index].numpy()) for index in range(len(final_model))]
    hash_counts = {action_hash: normalized_hashes.count(action_hash) for action_hash in set(normalized_hashes)}
    unique_indices = list(dict.fromkeys(normalized_hashes))
    unique_indices = [normalized_hashes.index(action_hash) for action_hash in unique_indices]
    unique_predicted = predicted_scores[unique_indices]
    unique_observed = observed_scores[unique_indices]
    unique_native = native[unique_indices]
    unique_source = source_scores[unique_indices]
    p_local = _strict_choice(unique_predicted, lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    source_local = _strict_choice(unique_source, lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    o_local = _strict_choice(unique_observed, lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"])
    n_local = _strict_choice(unique_native, lower_better=False, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"], allow_tie=True)
    goal_local = _strict_choice(
        goal_distances[unique_indices], lower_better=True, atol=LOCKED_PROTOCOL["tie_atol"], rtol=LOCKED_PROTOCOL["tie_rtol"], allow_tie=True
    )
    p_idx = unique_indices[p_local] if p_local is not None else None
    source_idx = unique_indices[source_local] if source_local is not None else None
    o_idx = unique_indices[o_local] if o_local is not None else None
    n_idx = unique_indices[n_local] if n_local is not None else None
    goal_idx = unique_indices[goal_local] if goal_local is not None else None
    best_native = float(np.max(unique_native))
    best_goal_distance = float(np.min(goal_distances[unique_indices]))

    returned_decomp = decompose_latent_scores(
        captured["returned_predicted"]["visual"].float().cpu().numpy(),
        encoded_returned["visual"][:, -1].float().cpu().numpy(), goal_v,
        predicted_proprio=captured["returned_predicted"]["proprio"].float().cpu().numpy(),
        realized_proprio=encoded_returned["proprio"][:, -1].float().cpu().numpy(), goal_proprio=goal_p,
        alpha=float(raw_cfg["planner"]["planning_objective"]["alpha"]),
    )
    returned_p = float(np.asarray(returned_decomp["predicted_score"]).reshape(-1)[0])
    returned_o = float(np.asarray(returned_decomp["observed_score"]).reshape(-1)[0])
    returned_score_error = returned_p - float(captured["returned_score"])
    if abs(returned_score_error) > score_tolerance:
        raise RuntimeError(
            f"returned-mean score reproduction exceeded tolerance: {returned_score_error} > {score_tolerance}"
        )
    returned_native = float(returned_replay["native_return"])
    candidate_rows = []
    for index in range(len(replays)):
        candidate_rows.append({
            "candidate_id": f"sample_{index:04d}",
            "predicted_score": float(predicted_scores[index]),
            "source_planner_score": float(source_scores[index]),
            "score_reproduction_error": float(score_errors[index]),
            "encoded_real_score": float(observed_scores[index]),
            "native_return": float(native[index]),
            "native_state_distance": replays[index]["native_state_distance"],
            "native_success": replays[index]["native_success"],
            "final_coverage": replays[index]["final_coverage"],
            "directional_error": float(decomp["directional_correction"][index]),
            "magnitude_error": float(decomp["magnitude_correction"][index]),
            "decomposition_identity_error": float(decomp["reconstruction_residual"][index]),
            "action_model": final_model[index].numpy(),
            "normalized_action_sha256": normalized_hashes[index],
            "normalized_plan_multiplicity": hash_counts[normalized_hashes[index]],
            "physical_action_sha256": _sha256_bytes(final_physical[index].numpy()),
            "final_state": replays[index]["state"],
        })
    ids = [row["candidate_id"] for row in candidate_rows]
    random_model_label = model_name or ("official_dino_wm" if "dino" in (checkpoint.name + config.name).lower() else "official_jepa_wm")
    random_seed_material = f"{random_model_label}:{episode}:{offset}".encode("utf-8")
    random_index = int(hashlib.sha256(random_seed_material).hexdigest(), 16) % len(candidate_rows)
    if p_idx is not None and random_index == p_idx:
        random_index = (random_index + 1) % len(candidate_rows)
    random_outcome = replays[random_index]
    returned_hash = _sha256_bytes(returned_model[0].numpy())
    returned_inventory_index = normalized_hashes.index(returned_hash) if returned_hash in normalized_hashes else None
    effective_decision_id = decision_id or f"pusht_val_episode_{episode:03d}_offset_{offset:03d}"
    decision = {
        "decision_id": effective_decision_id,
        "candidate_count": len(candidate_rows),
        "unique_plan_count": len(unique_indices),
        "initial_inventory_count": int(initial_model.shape[0]),
        "initial_inventory_normalized_sha256": _sha256_bytes(initial_model.numpy()),
        "predicted_choice": ids[p_idx] if p_idx is not None else None,
        "source_planner_choice": ids[source_idx] if source_idx is not None else None,
        "source_recomputed_winner_agrees": (
            source_idx is not None and p_idx is not None and source_idx == p_idx
        ),
        "source_recomputed_abstained": source_idx is None and p_idx is None,
        "final_captured_winner_agrees": source_idx is not None and p_idx is not None and source_idx == p_idx,
        "captured_score_source": "cem_final_iteration_objective_scores",
        "encoded_real_choice": ids[o_idx] if o_idx is not None else None,
        "native_reward_choice": ids[n_idx] if n_idx is not None else None,
        "goal_aligned_choice": ids[goal_idx] if goal_idx is not None else None,
        "predicted_tie": p_idx is None,
        "encoded_real_tie": o_idx is None,
        "native_tie": n_idx is None,
        "goal_aligned_tie": goal_idx is None,
        "reward_comparison_abstained": p_idx is None or o_idx is None,
        "goal_comparison_abstained": p_idx is None or o_idx is None,
        "reward_winner_pattern": _winner_pattern(p_idx, o_idx, n_idx),
        "goal_aligned_winner_pattern": _winner_pattern(p_idx, o_idx, goal_idx),
        "native_best_return": best_native,
        "goal_aligned_best_distance": best_goal_distance,
        "R_P": best_native - float(native[p_idx]) if p_idx is not None else None,
        "R_O": best_native - float(native[o_idx]) if o_idx is not None else None,
        "Delta_reward": (
            float(native[o_idx] - native[p_idx])
            if p_idx is not None and o_idx is not None else None
        ),
        "goal_aligned_R_P": (
            float(goal_distances[p_idx] - best_goal_distance) if p_idx is not None else None
        ),
        "goal_aligned_R_O": (
            float(goal_distances[o_idx] - best_goal_distance) if o_idx is not None else None
        ),
        "Delta_goal": (
            float(goal_distances[p_idx] - goal_distances[o_idx])
            if p_idx is not None and o_idx is not None else None
        ),
        "delta_definition": "Delta = R_P - R_O",
        "returned_mean_regret": max(0.0, best_native - returned_native),
        "returned_mean_native_gap_signed": best_native - returned_native,
        "returned_mean_inventory_match": returned_inventory_index is not None,
        "returned_mean_inventory_candidate_id": (
            ids[returned_inventory_index] if returned_inventory_index is not None else None
        ),
        "optimizer_gap": returned_p - float(np.min(predicted_scores)),
        "returned_mean_predicted_score": returned_p,
        "returned_mean_score_reproduction_error": returned_score_error,
        "returned_mean_observed_score": returned_o,
        "returned_mean_native_return": returned_native,
        "returned_mean_native_state_distance": returned_replay["native_state_distance"],
        "returned_mean_goal_aligned_regret": max(0.0, float(
            returned_replay["native_state_distance"] - best_goal_distance
        )),
        "returned_mean_goal_aligned_gap_signed": float(
            returned_replay["native_state_distance"] - best_goal_distance
        ),
        "returned_mean_native_success": returned_replay["native_success"],
        "returned_mean_final_coverage": returned_replay["final_coverage"],
        "returned_mean_action_model": returned_model[0].numpy(),
        "returned_mean_physical_action_sha256": _sha256_bytes(returned_physical.numpy()),
        "returned_mean_directional_error": float(np.asarray(returned_decomp["directional_correction"]).reshape(-1)[0]),
        "returned_mean_magnitude_error": float(np.asarray(returned_decomp["magnitude_correction"]).reshape(-1)[0]),
        "returned_mean_decomposition_identity_error": float(np.asarray(returned_decomp["reconstruction_residual"]).reshape(-1)[0]),
    }
    baselines = {
        "descriptive_only": True,
        "expert_goal_defining_segment": {
            "J": expert_outcome["native_return"],
            "G": -expert_outcome["native_state_distance"],
            "native_return": expert_outcome["native_return"],
            "native_state_distance": expert_outcome["native_state_distance"],
            "native_success": expert_outcome["native_success"],
            "final_coverage": expert_outcome["final_coverage"],
            "J_definition": "float32 sum of 30 official coverage rewards",
            "G_definition": "negative official final state distance",
        },
        "hold_position_noop": {
            "J": noop_replay["native_return"],
            "G": -noop_replay["native_state_distance"],
            "native_return": noop_replay["native_return"],
            "native_state_distance": noop_replay["native_state_distance"],
            "native_success": noop_replay["native_success"],
            "final_coverage": noop_replay["final_coverage"],
            "physical_action_sha256": _sha256_bytes(noop_physical.numpy()),
            "relative_action_semantics": relative_actions,
            "action_scale": action_scale,
            "definition": "repeat the initial agent position for absolute PD targets; zero relative displacement for the released relative-action wrapper",
            "J_definition": "float32 sum of 30 official coverage rewards",
            "G_definition": "negative official final state distance",
        },
        "frozen_hash_random_distinct_candidate": {
            "candidate_id": ids[random_index],
            "candidate_index": random_index,
            "normalized_action_sha256": normalized_hashes[random_index],
            "J": random_outcome["native_return"],
            "G": -random_outcome["native_state_distance"],
            "native_return": random_outcome["native_return"],
            "native_state_distance": random_outcome["native_state_distance"],
            "native_success": random_outcome["native_success"],
            "final_coverage": random_outcome["final_coverage"],
            "selection": "sha256(model, episode, offset) modulo final inventory, shifted if equal to predicted winner",
            "J_definition": "float32 sum of 30 official coverage rewards",
            "G_definition": "negative official final state distance",
        },
    }
    video_path = dataset_root / dataset_split / "obses" / f"episode_{episode:03d}.mp4"
    archive_path = official_root / "pusht_noise.zip"
    model_label = model_name or ("official_dino_wm" if "dino" in (checkpoint.name + config.name).lower() else "official_jepa_wm")
    output = {
        "schema_version": "official-pusht-decision-audit-v2",
        "scientific_result": False,
        "audit_status": (
            "hardware_adapted_same_process_engineering_repeat_only"
            if planning_repeats >= 3 and repeat_count >= 3
            else "hardware_adapted_confirmatory_row_external_certificate"
        ),
        "model": model_label,
        "decision_id": effective_decision_id,
        "runtime_determinism": {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
            "deterministic_algorithms": deterministic_algorithms,
        },
        "runtime": _runtime_fingerprint(),
        "hardware": _hardware_fingerprint(),
        "runtime_sha256": _canonical_sha256(_runtime_fingerprint()),
        "hardware_sha256": _canonical_sha256(_hardware_fingerprint()),
        "official_git_commit": _git_commit(official_root),
        "protocol": LOCKED_PROTOCOL,
        "protocol_sha256": _protocol_sha256(),
        "official_root": str(official_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "config": str(config.resolve()),
        "config_sha256": _sha256_file(config),
        "dinov2": dinov2_dependency,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_split": dataset_split,
        "seq_lengths_sha256": _sha256_file(dataset_root / dataset_split / "seq_lengths.pkl"),
        "dataset_states_sha256": _sha256_file(dataset_root / dataset_split / "states.pth"),
        "dataset_files": {
            relative: _sha256_file(path)
            for path in sorted(
                (item for item in (dataset_root / dataset_split).rglob("*") if item.is_file()),
                key=lambda item: item.as_posix(),
            )
            for relative in [path.relative_to(dataset_root / dataset_split).as_posix()]
        },
        "source_bundle_sha256": _source_bundle_sha256(),
        "dataset_archive_sha256": _sha256_file(archive_path),
        "states_segment_sha256": _sha256_bytes(np.asarray(states[offset : offset + needed])),
        "actions_segment_sha256": _sha256_bytes(np.asarray(expert_normalized)),
        "episode_video_sha256": _sha256_file(video_path),
        "episode": episode,
        "offset": offset,
        "trajectory_id": trajectory_id or f"val_episode_{episode:03d}",
        "seed": seed,
        "manifest_sha256": manifest_sha256,
        "frameskip": frameskip,
        "horizon": horizon,
        "cem_iterations": int(raw_cfg["planner"]["iterations"]),
        "cem_num_samples": int(raw_cfg["planner"]["num_samples"]),
        "cem_num_elites": int(raw_cfg["planner"]["num_elites"]),
        "model_action_dim": int(model.action_dim),
        "physical_action_dim": 2,
        "physical_replay_steps": int(frameskip * horizon),
        "initial_state": init_state,
        "goal_state": goal_state,
        "candidate_rows": candidate_rows,
        "decision": decision,
        "baselines": baselines,
        "common_inventory": common_inventory,
        "unique_plan_count": len(unique_indices),
        "initial_inventory": {
            "outcome_blind": True,
            "count": int(initial_model.shape[0]),
            "normalized_sha256": _sha256_bytes(initial_model.numpy()),
            "physical_sha256": _sha256_bytes(initial_physical.numpy()),
            "objective_scores_sha256": _sha256_bytes(initial_source_scores),
            "score_capture": "cem_iteration_0",
            "sidecar_key": "initial_actions_normalized",
        },
        "initial_score_reproduction_max_abs_error": float(np.max(np.abs(initial_score_errors))),
        "initial_decomposition_max_abs_error": initial_decomp_error,
        "score_reproduction_max_abs_error": float(np.max(np.abs(score_errors))),
        "score_reproduction_tolerance": score_tolerance,
        "score_reproduction_tolerance_status": "locked_protocol_tolerance",
        "returned_mean_score_reproduction_error": returned_score_error,
        "decomposition_max_abs_error": float(np.max(np.abs(np.asarray(decomp["reconstruction_residual"]))),),
        "returned_mean_replay_state": returned_replay["state"],
        "planning_reproducibility_gate": planning_gate,
        "reproducibility_gate": {
            "repeat_count": repeat_count,
            "passed": repeat_count >= 3,
            "status": "passed" if repeat_count >= 3 else "row_not_repeated_external_certificate_required",
            "candidate_id": "sample_0000",
            "state_sha256": repeat_state_hashes,
            "terminal_observation_sha256": repeat_obs_hashes,
            "cumulative_reward": repeat_returns,
            "native_state_distance": [item["native_state_distance"] for item in repeats],
            "native_success": [item["native_success"] for item in repeats],
            "winner_identities": {
                "predicted_choice": decision["predicted_choice"],
                "encoded_real_choice": decision["encoded_real_choice"],
                "native_reward_choice": decision["native_reward_choice"],
                "goal_aligned_choice": decision["goal_aligned_choice"],
            },
        },
        "execution_deviation": {
            "cem_population_samples": int(raw_cfg["planner"]["num_samples"]),
            "cem_score_batch_size": int(captured["score_batch_size"]),
            "upstream_cem_score_batch_size": int(raw_cfg["planner"]["num_samples"]),
            "terminal_encode_batch_size": 8,
            "note": "The same 300 sampled plans are scored with batches of 8 and 4 at every CEM iteration; batch-dependent floating-point differences are recorded and must remain within tolerance.",
        },
        "chunk_equivalence": {
            "population_size": int(raw_cfg["planner"]["num_samples"]),
            "iterations_checked": len(chunk_trace),
            "full_population_checked": True,
            "full_batch_size": 8,
            "chunk_batch_size": 4,
            "max_abs_score_difference": chunk_score_diff,
            "elite_set_agrees": chunk_elite_set_agrees,
            "elite_order_agrees": chunk_elite_order_agrees,
            "tolerance": score_tolerance,
            "trace": chunk_trace,
        },
        "initial_final_states_sha256": _sha256_bytes(np.asarray([item["state"] for item in initial_replays], dtype=np.float64)),
        "initial_native_success_sha256": _sha256_bytes(np.asarray([item["native_success"] for item in initial_replays], dtype=np.bool_)),
        "initial_final_coverage_sha256": _sha256_bytes(np.asarray([item["final_coverage"] for item in initial_replays], dtype=np.float64)),
        "initial_raw_visual_digest_array_sha256": _sha256_bytes(initial_raw_visual_digests),
        "initial_raw_proprio_digest_array_sha256": _sha256_bytes(initial_raw_proprio_digests),
        "final_states_sha256": _sha256_bytes(np.asarray([item["state"] for item in replays], dtype=np.float64)),
        "native_success_sha256": _sha256_bytes(np.asarray([item["native_success"] for item in replays], dtype=np.bool_)),
        "final_coverage_sha256": _sha256_bytes(np.asarray([item["final_coverage"] for item in replays], dtype=np.float64)),
        "final_raw_visual_digest_array_sha256": _sha256_bytes(final_raw_visual_digests),
        "final_raw_proprio_digest_array_sha256": _sha256_bytes(final_raw_proprio_digests),
        "returned_mean_raw_visual_digest_array_sha256": _sha256_bytes(returned_mean_raw_visual_digests),
        "returned_mean_raw_proprio_digest_array_sha256": _sha256_bytes(returned_mean_raw_proprio_digests),
        "returned_mean_final_state_sha256": _sha256_bytes(np.asarray([returned_replay["state"]], dtype=np.float64)),
        "returned_mean_native_state_distance_sha256": _sha256_bytes(np.asarray([returned_replay["native_state_distance"]], dtype=np.float64)),
        "returned_mean_native_success_sha256": _sha256_bytes(np.asarray([returned_replay["native_success"]], dtype=np.bool_)),
        "returned_mean_final_coverage_sha256": _sha256_bytes(np.asarray([returned_replay["final_coverage"]], dtype=np.float64)),
    }
    if arrays_out is not None:
        arrays_out = arrays_out.resolve()
        arrays_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_out,
            final_actions_normalized=final_model.numpy().astype(np.float32),
            initial_actions_normalized=initial_model.numpy().astype(np.float32),
            initial_physical_actions=initial_physical.numpy().astype(np.float32),
            initial_objective_scores=initial_source_scores.astype(np.float64),
            final_objective_scores=source_scores.astype(np.float64),
            initial_predicted_visual=initial_pred_v.astype(np.float32),
            initial_encoded_real_visual=initial_real_v.astype(np.float32),
            initial_predicted_proprio=initial_pred_p.astype(np.float32),
            initial_encoded_real_proprio=initial_real_p.astype(np.float32),
            initial_native_returns=initial_native.astype(np.float64),
            initial_native_state_distances=initial_goal_distances.astype(np.float64),
            initial_final_states=np.asarray([item["state"] for item in initial_replays], dtype=np.float64),
            initial_native_success=np.asarray([item["native_success"] for item in initial_replays], dtype=np.bool_),
            initial_final_coverage=np.asarray([item["final_coverage"] for item in initial_replays], dtype=np.float64),
            initial_raw_visual_digests=initial_raw_visual_digests,
            initial_raw_proprio_digests=initial_raw_proprio_digests,
            final_raw_visual_digests=final_raw_visual_digests,
            final_raw_proprio_digests=final_raw_proprio_digests,
            returned_mean_actions_normalized=returned_model[0].numpy().astype(np.float32),
            predicted_visual=pred_v.astype(np.float32),
            encoded_real_visual=real_v.astype(np.float32),
            goal_visual=goal_v.astype(np.float32),
            predicted_proprio=pred_p.astype(np.float32),
            encoded_real_proprio=real_p.astype(np.float32),
            goal_proprio=goal_p.astype(np.float32),
            native_returns=native.astype(np.float64),
            native_state_distances=goal_distances.astype(np.float64),
            final_states=np.asarray([item["state"] for item in replays], dtype=np.float64),
            native_success=np.asarray([item["native_success"] for item in replays], dtype=np.bool_),
            final_coverage=np.asarray([item["final_coverage"] for item in replays], dtype=np.float64),
            returned_predicted_visual=captured["returned_predicted"]["visual"].float().cpu().numpy().astype(np.float32),
            returned_encoded_real_visual=encoded_returned["visual"][:, -1].float().cpu().numpy().astype(np.float32),
            returned_predicted_proprio=captured["returned_predicted"]["proprio"].float().cpu().numpy().astype(np.float32),
            returned_encoded_real_proprio=encoded_returned["proprio"][:, -1].float().cpu().numpy().astype(np.float32),
            returned_objective_score=np.asarray([captured["returned_score"]], dtype=np.float64),
            returned_native_return=np.asarray([returned_native], dtype=np.float64),
            returned_mean_raw_visual_digests=returned_mean_raw_visual_digests,
            returned_mean_raw_proprio_digests=returned_mean_raw_proprio_digests,
            returned_mean_final_state=np.asarray([returned_replay["state"]], dtype=np.float64),
            returned_mean_native_state_distance=np.asarray([returned_replay["native_state_distance"]], dtype=np.float64),
            returned_mean_native_success=np.asarray([returned_replay["native_success"]], dtype=np.bool_),
            returned_mean_final_coverage=np.asarray([returned_replay["final_coverage"]], dtype=np.float64),
        )
        with np.load(arrays_out, allow_pickle=False) as archive:
            output["array_hashes"] = {
                key: _sha256_bytes(archive[key]) for key in sorted(archive.files)
            }
        output["arrays_sidecar"] = str(arrays_out)
        output["arrays_sidecar_sha256"] = _sha256_file(arrays_out)
    return _json_value(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    from experiments.official_pusht_batch_study import (
        DINOV2_SOURCE_ARCHIVE, DINOV2_SOURCE_ROOT, DINOV2_TORCH_HOME, DINOV2_WEIGHTS,
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays-out", type=Path)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--planning-repeats", type=int, default=0)
    parser.add_argument(
        "--inventory-mode", choices=("natural_final", "common_initial_only"),
        default="natural_final",
        help="v1.2 common-initial-only mode samples and scores iteration 0 only",
    )
    parser.add_argument("--model-name")
    parser.add_argument("--decision-id")
    parser.add_argument("--trajectory-id")
    parser.add_argument("--dinov2-source-root", type=Path, default=DINOV2_SOURCE_ROOT)
    parser.add_argument("--dinov2-source-archive", type=Path, default=DINOV2_SOURCE_ARCHIVE)
    parser.add_argument("--dinov2-weights", type=Path, default=DINOV2_WEIGHTS)
    parser.add_argument("--torch-home", type=Path, default=DINOV2_TORCH_HOME)
    args = parser.parse_args()
    result = run_one_decision(
        official_root=args.official_root, checkpoint=args.checkpoint, config=args.config,
        dataset_root=args.dataset_root, episode=args.episode, offset=args.offset, seed=args.seed,
        arrays_out=args.arrays_out or args.out.with_suffix(".npz"),
        repeat_count=args.repeat_count,
        planning_repeats=args.planning_repeats,
        inventory_mode=args.inventory_mode,
        model_name=args.model_name,
        decision_id=args.decision_id,
        trajectory_id=args.trajectory_id,
        dinov2_source_root=args.dinov2_source_root,
        dinov2_source_archive=args.dinov2_source_archive,
        dinov2_weights=args.dinov2_weights,
        torch_home=args.torch_home,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
