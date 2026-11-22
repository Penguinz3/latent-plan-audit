from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.analyze_wall_audit import FROZEN_PATH, ProductionNotReady, _attempt_dir, _result_path, analyze_records, canonical, load_verified_records, record_from_result, validate_manifest


def _frozen(threshold: float = 1.23):
    model = {"near_tie_score_greater_equal": threshold, "training_quantile": 0.75}
    probability = {"feature_transform": "identity", "center": 0.0, "scale": 1.0, "coefficients": [0.0, 0.0, 1.0], "column_order": ["intercept", "is_dino", "standardized_feature"]}
    raw_probability = copy.deepcopy(probability)
    raw_probability["feature_transform"] = "-log10(max(raw_margin,1e-12))"
    return {"frozen_before_wall_outcomes": True, "thresholds": {"jepa": model, "dino": model}, "probability_models": {"near_tie_score": probability, "raw_margin": raw_probability}}


def _result(p_distance: float = 8.0, o_distance: float = 2.0):
    details = []
    for index in range(300):
        details.append({
            "candidate": index,
            "predicted_cost_P": float(index),
            "encoded_real_cost_O": float(index),
            "native_terminal_distance": float(o_distance if index == 0 else p_distance if index == 1 else 20 + index),
            "native_terminal_success": bool(index == 0),
            "actions_physical_sha256": "duplicate" if index in (0, 1) else f"a{index}",
        })
    details[1]["predicted_cost_P"] = 0.5
    return {"status": "PASS", "scientific_result": True, "candidates": 300, "predicted_choice_P": 0, "encoded_real_choice_O": 1, "candidates_detail": details}


def test_selector_mask_and_frozen_threshold_transfer_without_retuning():
    item = {"case_id": "wall_000", "case_seed": 2026091100, "menu_seed": 1, "model": "jepa"}
    frozen = _frozen()
    row = record_from_result(item, _result(), frozen)
    assert row["p_index"] == 0 and row["o_index"] == 1
    assert row["unique_physical_action_count"] == 299
    # The second-lowest distinct physical action is candidate 2 (candidate 1
    # is a duplicate of P), so the deduplicated raw margin is 2.0.
    assert row["raw_margin"] == pytest.approx(2.0)
    assert row["near_tie_threshold"] == pytest.approx(1.23)
    # Outcome edits do not change the frozen score threshold or flag.
    changed = _result(p_distance=100.0, o_distance=0.1)
    assert record_from_result(item, changed, frozen)["near_tie_flagged"] == row["near_tie_flagged"]


def test_manifest_requires_the_full_frozen_case_menu_model_grid():
    rows = [{"case_id": f"wall_{case:03d}", "case_seed": 2026091100 + case, "menu_seed": menu, "model": model} for case in range(30) for menu in (1, 2, 3) for model in ("jepa", "dino")]
    assert len(validate_manifest({"rows": rows})) == 180
    with pytest.raises(ProductionNotReady):
        validate_manifest({"rows": rows[:-1]})


def test_complete_case_mask_and_pooled_two_outcome_families():
    records = []
    for case in range(30):
        for menu in (1, 2, 3):
            for model in ("jepa", "dino"):
                records.append({
                    "case_id": f"wall_{case:03d}", "model": model, "menu_seed": menu,
                    "o_minus_p_native_utility": 2.0, "o_minus_p_binary_success": 1.0,
                    "p_minus_uniform_native_utility": 0.5, "p_minus_uniform_binary_success": 0.25,
                    "p_signed_native_menu_gap": -1.0, "o_signed_native_menu_gap": -0.5,
                    "near_tie_flagged": case % 2 == 0, "p_choice_o_choice_switch": True,
                    "near_tie_score": 2.0, "raw_margin": 0.1,
                    "frozen_probability_near_tie": 0.5, "frozen_probability_raw_margin": 0.5,
                })
    result = analyze_records(records, _frozen())
    assert result["coverage"]["complete_case_count"] == 30
    assert result["coverage"]["primary_support_quality_passed"] is True
    assert result["primary"]["pooled"]["o_minus_p_native_utility"]["mean"] == pytest.approx(2.0)
    assert result["primary"]["pooled"]["p_minus_uniform_binary_success"]["mean"] == pytest.approx(0.25)
    assert set(result["primary"]["pooled"]["holm_adjusted_p"]) == {"O_minus_P_family", "P_minus_uniform_family"}


def test_missing_complete_case_and_single_row_are_reported_without_imputation():
    records = []
    for case in range(30):
        for menu in (1, 2, 3):
            for model in ("jepa", "dino"):
                records.append({
                    "case_id": f"wall_{case:03d}", "model": model, "menu_seed": menu,
                    "o_minus_p_native_utility": 2.0, "o_minus_p_binary_success": 1.0,
                    "p_minus_uniform_native_utility": 0.5, "p_minus_uniform_binary_success": 0.25,
                    "p_signed_native_menu_gap": -1.0, "o_signed_native_menu_gap": -0.5,
                    "near_tie_flagged": case % 2 == 0, "p_choice_o_choice_switch": True,
                    "near_tie_score": 2.0, "raw_margin": 0.1,
                    "frozen_probability_near_tie": 0.5, "frozen_probability_raw_margin": 0.5,
                })
    result = analyze_records(records[:-6], _frozen(), source={"terminal_failure_rows": 6})
    assert result["coverage"]["complete_case_count"] == 29
    assert result["coverage"]["primary_support_quality_passed"] is True
    incomplete = analyze_records(records[6:-1], _frozen(), source={"terminal_failure_rows": 7})
    assert incomplete["coverage"]["complete_case_count"] == 28
    assert incomplete["coverage"]["valid_rows"] == 173


def _write_finalized_run(root: Path, *, corrupt_receipt: bool = False) -> None:
    run = root / "run"
    (run / "results").mkdir(parents=True)
    (run / "attempts").mkdir(parents=True)
    rows = [{"case_id": f"wall_{case:03d}", "case_seed": 2026091100 + case, "menu_seed": menu, "model": model, "decision_id": f"wall_{case:03d}_seed{menu:02d}_{model}"} for case in range(30) for menu in (1, 2, 3) for model in ("jepa", "dino")]
    (run / "manifest.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    protocol = json.loads(Path("experiments/wall_audit_20260910/protocol.json").read_text(encoding="utf-8"))
    (run / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"status": "complete_with_failures", "total": 180, "complete": 179, "failed": 1}), encoding="utf-8")
    for index, item in enumerate(rows):
        attempt = _attempt_dir(run, index, item) / "0001"
        attempt.mkdir(parents=True)
        receipt = {"attempt": 1, "item": item, "status": "failed", "finished_at": "2026-09-11T00:00:00Z"}
        if index != 0:
            details = [{"candidate": i, "predicted_cost_P": float(i), "encoded_real_cost_O": float(i), "native_terminal_distance": float(i + 1), "native_terminal_success": i == 0} for i in range(300)]
            result = {**item, "status": "PASS", "scientific_result": True, "candidates": 300, "predicted_choice_P": 0, "encoded_real_choice_O": 1, "candidates_detail": details}
            path = _result_path(run, item)
            path.write_text(json.dumps(result), encoding="utf-8")
            receipt = {"attempt": 1, "item": item, "status": "complete", "finished_at": "2026-09-11T00:00:00Z", "result": str(path), "result_sha256": "bad" if corrupt_receipt and index == 1 else __import__("hashlib").sha256(canonical(result)).hexdigest()}
        (attempt / "attempt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_finalized_failure_is_allowed_but_unattempted_or_corrupt_receipt_is_blocked(tmp_path):
    run = tmp_path / "run"
    _write_finalized_run(tmp_path)
    records, source = load_verified_records(run, _frozen())
    assert len(records) == 179 and source["terminal_failure_rows"] == 1
    _write_finalized_run(tmp_path / "unfinished")
    unfinished_receipt = next((tmp_path / "unfinished" / "run" / "attempts").glob("00000_*/0001/attempt.json"))
    unfinished_receipt.write_text(json.dumps({"attempt": 1, "status": "running"}), encoding="utf-8")
    with pytest.raises(ProductionNotReady, match="unattempted, running, or interrupted"):
        load_verified_records(tmp_path / "unfinished" / "run", _frozen())
    _write_finalized_run(tmp_path / "corrupt", corrupt_receipt=True)
    with pytest.raises(ProductionNotReady, match="matching complete attempt receipt/hash"):
        load_verified_records(tmp_path / "corrupt" / "run", _frozen())
