"""Audit one official Wall initial action menu for P/O agreement.

P is the official world-model terminal latent distance.  O is the same latent
distance after executing each normalized menu candidate in the official Wall
simulator and re-encoding its terminal observation.  Native state distance and
success are reported as descriptive simulator outcomes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

PROTOCOL = {
    "task": "wall-base",
    "horizon": 6,
    "frameskip": 5,
    "action_dim": 2,
    "model_action_dim": 10,
    "menu_candidates": 300,
    "menu_distribution": "N(0,1) normalized action coordinates",
    "objective": "terminal latent L2, visual + 0.1 * proprio",
    "native_success_threshold": 4.5,
    "cem_iterations": 0,
}

# These are the immutable DINOv2 inputs used by the released Wall configs.
# Keep the paths portable (resolved relative to the checkout), while failing
# closed if the bytes differ from the pinned release.
DINO_PROVENANCE = {
    "source_archive_sha256": "d679cb1bd644cde72cc4adf496b2a8c1644cdc7d8ee6d0defdcda40dc4795f6f",
    "source_tree_sha256": "f6331d091125a706392f6d7637e1b701b35cc3a4f465d358c7392c12333a383f",
    "weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_file_hashes(root: Path) -> dict[str, str]:
    """Hash source files while ignoring interpreter-generated caches."""
    ignored_suffixes = {".pyc", ".pyo", ".pyd", ".so", ".dll"}
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in ignored_suffixes:
            continue
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    return files


def sample_initial_menu(menu_seed: int, candidates: int = 300, *, device: str = "cpu") -> np.ndarray:
    """Return the common [N, H, 10] normalized menu in deterministic order."""
    if candidates < 2:
        raise ValueError("candidates must be at least 2")
    import torch

    generator = torch.Generator(device=device).manual_seed(int(menu_seed))
    # Match the official CEM initial population: zero mean, unit std, no clipping.
    menu = torch.randn((PROTOCOL["horizon"], candidates, PROTOCOL["model_action_dim"]), generator=generator, device=device).permute(1, 0, 2)
    # Match the released CEM's zero-mean proposal at population index 0.
    menu[0, :, :] = 0.0
    return menu.cpu().numpy().astype(np.float32)


def expand_and_denormalize(menu: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    """Unpack each model action into five physical 2-D actions."""
    values = np.asarray(menu, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (6, 10):
        raise ValueError(f"expected [N,6,10], got {values.shape}")
    mean = np.asarray(action_mean, dtype=np.float32).reshape(1, 1, 1, 2)
    std = np.asarray(action_std, dtype=np.float32).reshape(1, 1, 1, 2)
    return (values.reshape(values.shape[0], 6, 5, 2) * std + mean).reshape(values.shape[0], 30, 2)


def select_winner(costs: np.ndarray, *, atol: float = 1e-10, rtol: float = 1e-8) -> tuple[int | None, float | None]:
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1 or costs.size < 2 or not np.isfinite(costs).all():
        raise ValueError("costs must be a finite vector with at least two entries")
    order = np.argsort(costs, kind="stable")
    best, second = float(costs[order[0]]), float(costs[order[1]])
    ties = np.isclose(costs, best, atol=atol, rtol=rtol)
    # Deterministic first-index tie breaking is part of the P/O audit contract.
    return int(order[0]), second - best


def tie_count(costs: np.ndarray, *, atol: float = 1e-10, rtol: float = 1e-8) -> int:
    costs = np.asarray(costs, dtype=np.float64)
    return int(np.isclose(costs, np.min(costs), atol=atol, rtol=rtol).sum())


def p_only_distinct_margin(costs: np.ndarray, physical_actions: np.ndarray) -> dict[str, Any]:
    """Frozen P-only top-two distinct-action margin diagnostic."""
    costs = np.asarray(costs, dtype=np.float64)
    hashes = [hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest() for action in np.asarray(physical_actions)]
    first_by_hash: dict[str, int] = {}
    for index, digest in enumerate(hashes):
        previous = first_by_hash.get(digest)
        if previous is None or (costs[index], index) < (costs[previous], previous):
            first_by_hash[digest] = index
    distinct = sorted(first_by_hash.values(), key=lambda index: (costs[index], index))
    if len(distinct) < 2:
        return {"distinct_action_count": len(distinct), "top_two_distinct_indices": distinct, "normalized_margin": None, "score": None}
    values = costs[distinct]
    gap = float(values[1] - values[0])
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    normalized = gap / max(iqr, 1e-12)
    return {"distinct_action_count": len(distinct), "top_two_distinct_indices": distinct[:2], "top_two_gap": gap, "p_iqr": iqr, "normalized_margin": normalized, "score": -float(np.log10(max(normalized, 1e-12)))}


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(k): _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def _to_raw_obs(obs: Any, info: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    visual = torch.as_tensor(obs["visual"] if isinstance(obs, Mapping) else obs)
    if visual.ndim == 3 and visual.shape[-1] == 3:
        visual = visual.permute(2, 0, 1)
    if visual.ndim == 3:
        visual = visual.unsqueeze(0)
    proprio = torch.as_tensor(info["proprio"])
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    return {"visual": visual.unsqueeze(0), "proprio": proprio.unsqueeze(0)}


def _latent_cost(predicted: Any, target: Any, alpha: float = 0.1) -> Any:
    """Official ReprTargetDistMPCObjective terminal branch, without imports."""
    visual = (target["visual"] - predicted["visual"]).pow(2).mean(dim=tuple(range(2, predicted["visual"].ndim)))
    proprio = (target["proprio"] - predicted["proprio"]).pow(2).mean(dim=tuple(range(2, predicted["proprio"].ndim)))
    return visual[-1] + float(alpha) * proprio[-1]


def _stack_raw(replays: list[tuple[Any, Mapping[str, Any]]]) -> dict[str, Any]:
    import torch

    return {
        "visual": torch.stack([torch.as_tensor(obs["visual"] if isinstance(obs, Mapping) else obs).permute(2, 0, 1) for obs, _ in replays]).unsqueeze(1),
        "proprio": torch.stack([torch.as_tensor(info["proprio"]) for _, info in replays]).unsqueeze(1),
    }


def _encode_chunks(model: Any, raw: dict[str, Any], chunk_size: int = 8) -> Any:
    import torch

    result = []
    for start in range(0, raw["visual"].shape[0], chunk_size):
        chunk = {key: value[start : start + chunk_size] for key, value in raw.items()}
        with torch.inference_mode():
            result.append(model.encode(chunk))
    if len(result) == 1:
        return result[0]
    from tensordict import TensorDict

    return TensorDict({key: torch.cat([item[key] for item in result], dim=0) for key in result[0].keys()})


def _plan_costs(model: Any, z_init: Any, z_goal: Any, menu: np.ndarray, device: Any, chunk_size: int = 8) -> np.ndarray:
    """Score a menu in bounded chunks; used by the audit and parity checks."""
    import torch

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    parts = []
    with torch.inference_mode():
        for start in range(0, len(menu), chunk_size):
            plans = torch.as_tensor(menu[start : start + chunk_size], device=device).permute(1, 0, 2)
            predicted = model.unroll(z_init, act_suffix=plans)
            parts.append(_latent_cost(predicted, z_goal).detach().float().cpu())
    return torch.cat(parts).numpy()


def _load_model(official_root: Path, checkpoint: Path, model_name: str, config_path: Path, device: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch
    import yaml
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stats = get_data_stats("wall")
    data_cfg = cfg["model_kwargs"]["data"]
    aug_cfg = cfg["model_kwargs"]["data_aug"]
    transform = make_transforms(img_size=int(data_cfg["img_size"]), normalize=aug_cfg["normalize"], random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0), random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False, motion_shift=False)
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]), action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]), state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]), proprio_std=torch.tensor(stats["proprio_std"]), transform=transform,
    )
    model_kwargs = dict(cfg["model_kwargs"])
    pretrain_kwargs = dict(model_kwargs["pretrain_kwargs"])
    pretrain_kwargs["heads_cfg"] = {}  # the audit only needs frozen encoder/predictor outputs
    dependency = {}
    if model_name in {"jepa", "dino"}:
        from experiments.official_pusht_decision_audit import _offline_dinov2_loader

        source = official_root.parent / "dinov2-release-pinned"
        archive = official_root.parent / "dinov2-b194f00db6136677fc8a4cc2ef2168f7699dfba2.tar.gz"
        torch_home = official_root.parent / "torch-cache"
        weights = official_root.parent / "dinov2-pinned-weights" / "dinov2_vits14_pretrain.pth"
        if not source.exists() or not archive.exists() or not weights.exists():
            raise FileNotFoundError("controlled offline DINOv2 source/archive/weights are required")
        torch_home.mkdir(parents=True, exist_ok=True)
        original_hub_load = torch.hub.load
        loader = _offline_dinov2_loader(torch, source, weights)
        torch.hub.load = loader
    try:
        model = init_module(
            folder=checkpoint.parent, checkpoint=checkpoint.name, model_kwargs=pretrain_kwargs,
            device=device, action_dim=stats["action_dim"], proprio_dim=stats["proprio_dim"], preprocessor=preprocessor,
            cfgs_data=data_cfg, wrapper_kwargs=cfg["model_kwargs"].get("wrapper_kwargs", {}),
        )
    finally:
        if model_name in {"jepa", "dino"}:
            torch.hub.load = original_hub_load
    if model_name in {"jepa", "dino"}:
        from experiments.official_pusht_batch_study import dinov2_binding, dinov2_source_tree_sha256

        pinned = dinov2_binding(source_root=source, source_archive=archive, weights=weights, torch_home=torch_home)
        pinned.update(loader.last_binding or {})
        # The live source tree can acquire interpreter __pycache__ files.  Use
        # a clean extraction of the already-hashed archive for the source-tree
        # fingerprint, while the loader above still imports the explicit tree.
        with tempfile.TemporaryDirectory(prefix="wall_dinov2_validate_") as temp_dir:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(temp_dir)
            roots = [path for path in Path(temp_dir).iterdir() if path.is_dir()]
            if len(roots) != 1:
                raise RuntimeError("DINOv2 archive must contain one source root")
            pinned["source_tree_sha256"] = dinov2_source_tree_sha256(roots[0])
            clean_files = _source_file_hashes(roots[0])
        live_files = _source_file_hashes(source)
        mismatched_files = sorted(
            relative for relative, digest in clean_files.items()
            if live_files.get(relative) != digest
        )
        extra_source_files = sorted(set(live_files) - set(clean_files))
        if mismatched_files or extra_source_files:
            details = []
            if mismatched_files:
                details.append("changed/missing: " + ", ".join(mismatched_files[:5]))
            if extra_source_files:
                details.append("extra: " + ", ".join(extra_source_files[:5]))
            raise RuntimeError("live DINOv2 source tree differs from pinned archive (" + "; ".join(details) + ")")
        pinned["source_tree_validation"] = "clean_source_archive_extract"
        mismatches = [
            f"{field}={pinned.get(field)!r} (expected {expected})"
            for field, expected in DINO_PROVENANCE.items()
            if pinned.get(field) != expected
        ]
        if mismatches:
            raise RuntimeError("controlled offline DINOv2 bytes failed validation: " + "; ".join(mismatches))
        dependency = {"dinov2": pinned}
    return model, preprocessor, cfg, dependency


def run_case(official_root: Path, model_name: str, checkpoint: Path, case_seed: int, menu_seed: int, output: Path, candidates: int = 300, device: str = "cuda", case_id: str | None = None, decision_id: str | None = None, pilot: bool = False, _planning_chunk_size: int = 8) -> dict[str, Any]:
    """Run one case; caller owns manifest/resume and final output promotion."""
    started = time.perf_counter()
    official_root, checkpoint, output = Path(official_root).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if model_name not in {"jepa", "dino"}:
        raise ValueError("model must be jepa or dino")
    if not official_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("official root or checkpoint is missing")
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    config_path = official_root / "configs" / "evals" / "simu_env_planning" / "wall" / ("jepa-wm" if model_name == "jepa" else "dino-wm")
    configs = sorted(config_path.glob("*.yaml"))
    if len(configs) != 1:
        raise RuntimeError(f"expected one released Wall config under {config_path}, found {len(configs)}")
    config_path = configs[0]
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch_device = torch.device(device)
    random.seed(int(case_seed))
    np.random.seed(int(case_seed))
    torch.manual_seed(int(case_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(case_seed))
    model, preprocessor, cfg_dict, dependency = _load_model(official_root, checkpoint, model_name, config_path, torch_device)
    cfg = _namespace(cfg_dict)
    cfg.task_specification.max_episode_steps = 30
    from evals.simu_env_planning.envs.wall_gym_wrap import make_env
    env = make_env(cfg, device="cpu")
    init_state, goal_state = env.sample_random_init_goal_states(int(case_seed))
    init_obs, init_info = env.prepare(int(case_seed), init_state)
    goal_obs, goal_info = env.prepare(int(case_seed), goal_state)
    init_raw, goal_raw = _to_raw_obs(init_obs, init_info), _to_raw_obs(goal_obs, goal_info)
    with torch.inference_mode():
        z_init, z_goal = model.encode(init_raw), model.encode(goal_raw)
    menu = sample_initial_menu(menu_seed, candidates, device="cpu")
    physical = expand_and_denormalize(menu, preprocessor.action_mean.numpy(), preprocessor.action_std.numpy())
    with torch.inference_mode():
        planning_started = time.perf_counter()
        p_cost = _plan_costs(model, z_init, z_goal, menu, torch_device, chunk_size=_planning_chunk_size)
    planning_elapsed = time.perf_counter() - planning_started
    replays: list[tuple[Any, Mapping[str, Any]]] = []
    terminal_states = []
    native_distances, native_success = [], []
    replay_started = time.perf_counter()
    for actions in physical:
        obs, info = env.prepare(int(case_seed), init_state)
        env.set_elapsed_steps(0)
        del obs
        final_obs = None
        final_info = info
        for action in actions:
            final_obs, _, _, _, final_info = env.step(torch.as_tensor(action, dtype=torch.float32))
        assert final_obs is not None
        replays.append((final_obs, final_info))
        state_value = final_info["state"]
        if torch.is_tensor(state_value):
            state_value = state_value.detach().cpu()
        state = np.asarray(state_value, dtype=np.float64)
        evaluation = env.eval_state(goal_state, state)
        terminal_states.append(state)
        native_distances.append(float(evaluation["state_dist"]))
        native_success.append(bool(evaluation["success"]))
    endpoint_started = time.perf_counter()
    endpoint_latent = _encode_chunks(model, _stack_raw(replays))
    with torch.inference_mode():
        o_cost = _latent_cost({"visual": endpoint_latent["visual"].transpose(0, 1), "proprio": endpoint_latent["proprio"].transpose(0, 1)}, z_goal).detach().float().cpu().numpy()
    endpoint_elapsed = time.perf_counter() - endpoint_started
    if not np.isfinite(p_cost).all() or not np.isfinite(o_cost).all():
        raise RuntimeError("non-finite P/O endpoint cost; scientific row failed")
    if not np.isfinite(np.asarray(terminal_states, dtype=np.float64)).all() or not np.isfinite(np.asarray(native_distances, dtype=np.float64)).all():
        raise RuntimeError("non-finite native terminal state/distance; scientific row failed")
    p_winner, p_margin = select_winner(p_cost)
    o_winner, o_margin = select_winner(o_cost)
    p_margin_diagnostic = p_only_distinct_margin(p_cost, physical)
    npz_path = output.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    np.savez_compressed(npz_path, actions_normalized=menu, actions_physical=physical, predicted_cost=p_cost, encoded_real_cost=o_cost, terminal_states=np.asarray(terminal_states), native_state_distance=np.asarray(native_distances), native_success=np.asarray(native_success, dtype=np.bool_), initial_state=np.asarray(init_state), goal_state=np.asarray(goal_state))
    save_elapsed = time.perf_counter() - save_started
    records = []
    for index in range(candidates):
        records.append({"candidate": index, "predicted_cost_P": float(p_cost[index]), "encoded_real_cost_O": float(o_cost[index]), "native_terminal_distance": float(native_distances[index]), "native_terminal_success": bool(native_success[index]), "actions_normalized": menu[index].tolist(), "actions_physical": physical[index].tolist(), "actions_physical_sha256": hashlib.sha256(np.ascontiguousarray(physical[index]).tobytes()).hexdigest(), "terminal_state": np.asarray(terminal_states[index]).tolist()})
    result = {
        "status": "PASS",
        "scientific_result": candidates == 300 and not pilot,
        "pilot": bool(pilot or candidates != 300),
        "model": model_name,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "official_root": str(official_root), "official_git_commit": git_commit(official_root), "config": str(config_path), "config_sha256": sha256_file(config_path),
        "dependencies": dependency,
        "protocol": PROTOCOL, "protocol_sha256": canonical_sha256(PROTOCOL),
        "case_id": case_id, "decision_id": decision_id, "case_seed": int(case_seed), "menu_seed": int(menu_seed), "candidates": int(candidates), "engineering_only": bool(pilot or candidates != 300),
        "predicted_choice_P": p_winner, "encoded_real_choice_O": o_winner,
        "P_margin_second_minus_best": p_margin, "O_margin_second_minus_best": o_margin,
        "P_tie_count": tie_count(p_cost), "O_tie_count": tie_count(o_cost),
        "P_only_distinct_margin": p_margin_diagnostic,
        "native_choice_by_distance": int(np.argmin(native_distances)),
        "initial_state": np.asarray(init_state).tolist(), "goal_state": np.asarray(goal_state).tolist(),
        "candidates_detail": records, "arrays_sidecar": str(npz_path), "arrays_sidecar_sha256": sha256_file(npz_path),
        "elapsed_seconds": {"total": time.perf_counter() - started, "model_and_setup": replay_started - started, "planning_scoring": planning_elapsed, "realized_replays": endpoint_started - replay_started, "endpoint_encoding": endpoint_elapsed, "save": save_elapsed},
        "peak_cuda_mib": float(torch.cuda.max_memory_allocated(torch_device) / 1024**2) if torch_device.type == "cuda" else 0.0,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "device": str(torch_device)},
        "notes": ["P and O are latent endpoint distances; native distance/success are simulator diagnostics.", "This audit makes no policy improvement claim.", "Each candidate is reset to the same initial state before realization."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    result["adapter_sha256"] = sha256_file(Path(__file__).resolve())
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--model", choices=("jepa", "dino"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-seed", type=int, required=True)
    parser.add_argument("--menu-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--decision-id")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=300)
    args = parser.parse_args()
    values = vars(args).copy()
    values["model_name"] = values.pop("model")
    result = run_case(**values)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()), "model": result["model"], "case_id": result["case_id"], "decision_id": result["decision_id"], "P": result["predicted_choice_P"], "O": result["encoded_real_choice_O"]}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
