"""Small CPU checks for scientific invariants; no original artifacts needed."""
import numpy as np
from pathlib import Path

from analyze_task_native_selector import (DISTANCE, SECONDARY, CONTRASTS, aggregate,
    assert_identical, derive_selector, first_minimum, set_analysis_mask, summarize, validate_output_directory)


def example(environment="pusht", cluster="t0", seed=1, model="jepa", valid=True):
    distances = np.arange(300, dtype=float) + 1
    distances[0] = distances[1] = 0
    reward = np.zeros(300); reward[2] = 10; reward[0] = 1; reward[1] = 2
    predicted = np.arange(300, dtype=float); predicted[2] = -1
    realized = np.arange(300, dtype=float); realized[1] = -1
    return {"environment": environment, "decision_id": f"{cluster}-{seed}-{model}", "cluster_id": cluster,
            "physical_set_id": f"{cluster}-{seed}", "menu_seed": seed, "model": model, "valid": valid,
            "analysis_eligible": valid, "a_pred": 2 if valid else None, "a_real": 1 if valid else None, "a_state": None,
            **({"distances": distances, "secondary": reward, "physical_hashes": [str(i) for i in range(300)],
               "predicted_cost": predicted, "realized_cost": realized} if valid else {})}


def main():
    root = Path(__file__).resolve().parent
    for bundle in (root / "output/pusht_mechanism_20260929/bundle", root / "output/wall_recompute_20260910_v3"):
        for destination in (bundle, bundle / "nested/new-results"):
            try:
                validate_output_directory(root, destination)
            except ValueError:
                pass
            else:
                raise AssertionError("Output inside immutable evidence must be rejected")
    validate_output_directory(root, root / "separate_results")
    row = example()
    assert first_minimum(row["distances"]) == 0
    derive_selector(row)
    assert row["a_state"] == 0 and row["a_state_exact_minimum_tie_count"] == 2
    assert row[f"a_state__{SECONDARY['pusht']}"] == 1  # Do not replace with reward argmax.
    assert row[f"{CONTRASTS[1]}__{SECONDARY['pusht']}"] == -1
    for outcome in (DISTANCE, SECONDARY["pusht"]):
        assert row[f"{CONTRASTS[0]}__{outcome}"] + row[f"{CONTRASTS[1]}__{outcome}"] == row[f"{CONTRASTS[2]}__{outcome}"]
    duplicate = example(); duplicate["physical_hashes"][0] = duplicate["physical_hashes"][1]
    derive_selector(duplicate)
    assert duplicate["a_real_vs_a_state__candidate_index"]
    assert not duplicate["a_real_vs_a_state__physical_action"]
    assert duplicate["a_state_distinct_physical_minimum_tie_count"] == 1
    try:
        assert_identical({"actions": np.array([1, 2])}, {"actions": np.array([1, 3])}, "changed candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("Changed physical candidate must block analysis")
    rows = [example(model="jepa"), example(model="dino"), example(seed=2), example(seed=2, model="dino", valid=False)]
    clusters = set_analysis_mask(rows, "pusht")
    assert [row["analysis_eligible"] for row in rows] == [True, True, False, False]
    for item in rows:
        derive_selector(item)
    assert rows[-1]["a_state"] is None and f"a_state__{DISTANCE}" not in rows[-1]
    field = f"{CONTRASTS[1]}__{SECONDARY['pusht']}"
    assert aggregate(rows, field, clusters, "pusht") == [-1 / 15]
    # Frozen p-values are sensitive to the last bits: preserve NumPy's 15-cell mean.
    scheduled = [1e16, 0, 1, 1, 1, 1, 1, -1e16, 1, 1, 1, 1, 1, 1, 1]
    floating_rows = [{"cluster_id": "float", "model": model, "offset": index // 3, "menu_seed": index % 3 + 1,
                      "analysis_eligible": index != 1, "effect": value}
                     for model in ("jepa", "dino") for index, value in enumerate(scheduled)]
    expected = float(np.mean([np.mean(scheduled), np.mean(scheduled)]))
    assert expected != sum(scheduled) / 15
    assert aggregate(list(reversed(floating_rows)), "effect", ["float"], "pusht") == [expected]
    summary, values, draw_hash = summarize(rows, clusters, "pusht", 20261002)
    assert len(summary) == 18 and len(values) == 18 and len(draw_hash) == 64
    wall = [example("wall", "c0", seed, model) for seed in (1, 2, 3) for model in ("jepa", "dino")]
    wall += [example("wall", "c1", seed, model, valid=not(seed == 1 and model == "jepa")) for seed in (1, 2, 3) for model in ("jepa", "dino")]
    assert set_analysis_mask(wall, "wall") == ["c0"]
    assert sum(item["analysis_eligible"] for item in wall) == 6
    for item in wall:
        derive_selector(item)
    assert aggregate(wall, f"{CONTRASTS[0]}__{DISTANCE}", ["c0"], "wall") == [3]
    print("PASS: immutable output guard, exact ties, same distance-selected reward, index/action disagreement, identity failure, missingness, frozen floating-order aggregation and contrast closure")


if __name__ == "__main__":
    main()
