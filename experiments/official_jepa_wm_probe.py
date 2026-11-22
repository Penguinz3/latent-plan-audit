"""Load an official Push-T world model and run one encode/unroll interface check.

This is an engineering gate, not a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(official_root: Path, checkpoint: Path, config: Path) -> dict[str, object]:
    official_root = official_root.resolve()
    checkpoint = checkpoint.resolve()
    config = config.resolve()
    for path in (official_root, checkpoint, config):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(official_root))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import torch
    import yaml

    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    if not torch.cuda.is_available():
        raise RuntimeError("the official-model probe requires CUDA")

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    model_cfg = cfg["model_kwargs"]
    pretrain = dict(model_cfg["pretrain_kwargs"])
    pretrain["heads_cfg"] = {}  # Latent planning does not require image/state decoders.
    data_cfg = model_cfg["data"]
    aug_cfg = model_cfg["data_aug"]
    stats = get_data_stats("pusht")
    transform = make_transforms(
        img_size=data_cfg["img_size"],
        normalize=aug_cfg["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]),
        action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]),
        state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]),
        proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
    )
    device = torch.device("cuda:0")
    model = init_module(
        folder=checkpoint.parent,
        checkpoint=checkpoint.name,
        model_kwargs=pretrain,
        device=device,
        action_dim=stats["action_dim"],
        proprio_dim=stats["proprio_dim"],
        preprocessor=preprocessor,
        cfgs_data=data_cfg,
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
    )

    size = data_cfg["img_size"]
    context_frames = int(cfg["task_specification"]["num_frames"])
    context = {
        "visual": torch.zeros((1, context_frames, 3, size, size), dtype=torch.uint8),
        "proprio": torch.tensor(stats["proprio_mean"], dtype=torch.float32)
        .reshape(1, 1, -1)
        .repeat(1, context_frames, 1),
    }
    with torch.inference_mode():
        encoded = model.encode(context)
        action_width = int(model.action_dim)
        predicted = model.unroll(
            encoded,
            torch.zeros((1, 1, action_width), device=device, dtype=torch.float32),
        )

    return {
        "status": "PASS_INTERFACE_ONLY",
        "scientific_result": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(config),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "action_width": action_width,
        "encoded_visual_shape": list(encoded["visual"].shape),
        "encoded_proprio_shape": list(encoded["proprio"].shape),
        "predicted_visual_shape": list(predicted["visual"].shape),
        "predicted_proprio_shape": list(predicted["proprio"].shape),
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_probe(args.official_root, args.checkpoint, args.config)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
