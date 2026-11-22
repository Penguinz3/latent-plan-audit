from __future__ import annotations

import numpy as np

from experiments.analyze_pusht_terminal_reward import aggregate, terminal_reward_from_trace


def _row(trajectory: str, offset: int, seed: int, model: str, effect: float, uniform: float) -> dict[str, float]:
    return {
        "trajectory_id": trajectory,
        "offset": offset,
        "seed": seed,
        "model": model,
        "o_minus_p_terminal_reward": effect,
        "p_minus_uniform_terminal_reward": uniform,
    }


def test_terminal_reward_uses_last_step_and_fixed_mask_denominator():
    trace = np.zeros((300, 30), dtype=np.float32)
    trace[:, -1] = 7.5
    assert np.all(terminal_reward_from_trace(trace) == 7.5)

    cells = {"t0": [(0, seed) for seed in range(1, 16)], "t1": [(0, seed) for seed in range(1, 16)]}
    records = {}
    for trajectory in cells:
        for offset, seed in cells[trajectory]:
            for model in ("official_jepa_wm", "official_dino_wm"):
                records[(trajectory, offset, seed, model)] = _row(trajectory, offset, seed, model, 2.0, 1.0)
    # One missing model row makes one cell unpaired; its fixed-denominator
    # contribution is zero for both models rather than a fabricated reward.
    del records[("t1", 0, 15, "official_dino_wm")]
    result = aggregate(records, cells)
    assert sum(result["matched_cell_counts"].values()) == 29
    assert result["pooled"]["o_minus_p_terminal_reward"]["mean"] == 29 * 2.0 / 30
    assert result["pooled"]["p_minus_uniform_terminal_reward"]["mean"] == 29 * 1.0 / 30
