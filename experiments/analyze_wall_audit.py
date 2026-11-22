"""Analyze a finalized, receipt-verified Wall initial-menu audit without running Wall."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

MODELS = ("jepa", "dino")
MENU_SEEDS = (1, 2, 3)
CASE_COUNT = 30
EXPECTED_ROWS = CASE_COUNT * len(MODELS) * len(MENU_SEEDS)
MIN_PRIMARY_CASES = 27
BOOTSTRAP_DRAWS = 10_000
MONTE_CARLO_SIGN_FLIPS = 100_000
BOOTSTRAP_SEED = 20260911
SIGN_FLIP_SEED = 20260912
FROZEN_PATH = Path("experiments/wall_audit_20260910/frozen_near_tie.json")
PLAN_PATH = Path("experiments/wall_audit_20260910/analysis_plan.json")


class ProductionNotReady(RuntimeError):
    """The run directory is not finalized or lacks receipt-verified evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def _holm(p_values: Mapping[str, Any]) -> dict[str, float | None]:
    values = sorted((float(value), key) for key, value in p_values.items() if _finite(value))
    result: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    for rank, (value, key) in enumerate(values):
        running = max(running, min(1.0, (len(values) - rank) * value))
        result[key] = running
    return result


def _bootstrap(values: list[float], label: str, draws: int = BOOTSTRAP_DRAWS) -> list[float] | None:
    if not values:
        return None
    # The protocol freezes one reproducible bootstrap seed.  Do not derive
    # endpoint-specific seeds from observed labels or outcomes.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(values, dtype=np.float64)
    sampled = data[rng.integers(0, len(data), size=(draws, len(data)))].mean(axis=1)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def _sign_flip(values: list[float]) -> tuple[float | None, str]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return None, "unavailable"
    observed = abs(float(np.mean(data)))
    if len(data) <= 20:
        total = 1 << len(data)
        bits = np.arange(len(data), dtype=np.uint64)
        exceed = 0
        for start in range(0, total, 65_536):
            masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)[:, None]
            signs = 1.0 - 2.0 * ((masks >> bits) & 1).astype(np.float64)
            exceed += int(np.count_nonzero(np.abs((signs @ data) / len(data)) >= observed - 1e-15))
        return float(exceed / total), "exact"
    rng = np.random.default_rng(SIGN_FLIP_SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(MONTE_CARLO_SIGN_FLIPS, len(data)))
    means = (signs @ data) / len(data)
    return float((np.count_nonzero(np.abs(means) >= observed - 1e-15) + 1) / (MONTE_CARLO_SIGN_FLIPS + 1)), "monte_carlo_100000_plus1"


def _stats(values: Iterable[float], label: str) -> dict[str, Any]:
    data = [float(x) for x in values if _finite(x)]
    p, method = _sign_flip(data)
    return {
        "mean": float(np.mean(data)) if data else None,
        "median": float(np.median(data)) if data else None,
        "sd": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0 if data else None,
        "case_count": len(data),
        "percentile_bootstrap_95_ci_unadjusted": _bootstrap(data, label),
        "two_sided_sign_flip_p": p,
        "sign_flip_method": method,
        "case_values": data,
    }


def _auroc(scores: Iterable[float], labels: Iterable[bool]) -> float | None:
    pairs = sorted((float(score), bool(label)) for score, label in zip(scores, labels) if _finite(score))
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[index:end])
        index = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _safe_action_key(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("actions_physical_sha256"), str):
        return row["actions_physical_sha256"]
    return hashlib.sha256(canonical(row.get("actions_physical"))).hexdigest()


def _frozen_probability(frozen: Mapping[str, Any], name: str, model: str, feature: float) -> float:
    spec = frozen["probability_models"][name]
    standardized = (feature - float(spec["center"])) / float(spec["scale"])
    is_dino = 1.0 if model == "dino" else 0.0
    intercept, dino_coef, feature_coef = (float(x) for x in spec["coefficients"])
    return _sigmoid(intercept + dino_coef * is_dino + feature_coef * standardized)


def record_from_result(item: Mapping[str, Any], result: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one verified runner JSON result into analysis primitives."""
    details = result.get("candidates_detail")
    if result.get("status") != "PASS" or result.get("scientific_result") is not True or result.get("candidates") != 300 or not isinstance(details, list) or len(details) != 300:
        raise ValueError("result is not a complete 300-candidate PASS artifact")
    p_index, o_index = result.get("predicted_choice_P"), result.get("encoded_real_choice_O")
    if not isinstance(p_index, int) or not isinstance(o_index, int) or not 0 <= p_index < 300 or not 0 <= o_index < 300:
        raise ValueError("result lacks integer P/O first-winner indices")
    distances = np.asarray([float(row["native_terminal_distance"]) for row in details], dtype=np.float64)
    successes = np.asarray([row["native_terminal_success"] for row in details], dtype=np.float64)
    if not np.isfinite(distances).all() or not all(isinstance(row["native_terminal_success"], bool) for row in details):
        raise ValueError("native distance/success vectors are invalid")
    unique: dict[str, float] = {}
    for row in details:
        key = _safe_action_key(row)
        score = float(row["predicted_cost_P"])
        if not _finite(score):
            raise ValueError("predicted P score is not finite")
        unique[key] = min(unique.get(key, float("inf")), score)
    unique_scores = np.sort(np.asarray(list(unique.values()), dtype=np.float64))
    raw_margin = float(unique_scores[1] - unique_scores[0]) if len(unique_scores) >= 2 else None
    iqr = float(np.subtract(*np.quantile(unique_scores, [0.75, 0.25]))) if len(unique_scores) else None
    normalized_margin = raw_margin / max(iqr, 1e-12) if raw_margin is not None and iqr is not None else None
    near_tie_score = -math.log10(max(normalized_margin, 1e-12)) if normalized_margin is not None else None
    model = str(item["model"])
    threshold = float(frozen["thresholds"][model]["near_tie_score_greater_equal"])
    p_dist, o_dist = float(distances[p_index]), float(distances[o_index])
    p_success, o_success = float(successes[p_index]), float(successes[o_index])
    uniform_utility = float(np.mean(-distances))
    uniform_success = float(np.mean(successes))
    min_distance, max_distance = float(np.min(distances)), float(np.max(distances))
    distance_range = max_distance - min_distance
    p_gap, o_gap = min_distance - p_dist, min_distance - o_dist
    p_norm = p_gap / distance_range if distance_range > 0 else None
    o_norm = o_gap / distance_range if distance_range > 0 else None
    return {
        "case_id": str(item["case_id"]), "case_seed": int(item["case_seed"]), "menu_seed": int(item["menu_seed"]), "model": model,
        "decision_id": result.get("decision_id", item.get("decision_id")),
        "p_index": p_index, "o_index": o_index, "p_distance": p_dist, "o_distance": o_dist,
        "p_success": p_success, "o_success": o_success,
        "o_minus_p_native_utility": p_dist - o_dist,
        "o_minus_p_binary_success": o_success - p_success,
        "p_minus_uniform_native_utility": -p_dist - uniform_utility,
        "p_minus_uniform_binary_success": p_success - uniform_success,
        "p_signed_native_menu_gap": p_gap, "o_signed_native_menu_gap": o_gap,
        "p_normalized_native_menu_gap": p_norm, "o_normalized_native_menu_gap": o_norm,
        "menu_min_distance": min_distance, "menu_max_distance": max_distance, "menu_distance_range": distance_range,
        "menu_constant_distance": distance_range == 0.0,
        "p_choice_o_choice_switch": p_index != o_index,
        "p_margin_runner": result.get("P_margin_second_minus_best"), "o_margin_runner": result.get("O_margin_second_minus_best"),
        "raw_margin": raw_margin, "unique_physical_action_count": len(unique), "physical_action_duplicate_count": 300 - len(unique),
        "raw_margin_iqr_unique_P": iqr, "normalized_margin": normalized_margin, "near_tie_score": near_tie_score,
        "near_tie_threshold": threshold, "near_tie_flagged": bool(near_tie_score is not None and near_tie_score >= threshold),
        "frozen_probability_near_tie": _frozen_probability(frozen, "near_tie_score", model, near_tie_score) if near_tie_score is not None else None,
        "frozen_probability_raw_margin": _frozen_probability(frozen, "raw_margin", model, -math.log10(max(raw_margin, 1e-12))) if raw_margin is not None else None,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ProductionNotReady(f"expected {EXPECTED_ROWS} manifest rows")
    expected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for item in rows:
        if not isinstance(item, Mapping) or any(key not in item for key in ("case_id", "case_seed", "menu_seed", "model")):
            raise ProductionNotReady("manifest row lacks required identity")
        case_id, case_seed, menu_seed, model = str(item["case_id"]), int(item["case_seed"]), int(item["menu_seed"]), str(item["model"])
        case_index = int(case_id.rsplit("_", 1)[1]) if case_id.startswith("wall_") else -1
        key = (case_id, case_seed, menu_seed, model)
        if case_index not in range(CASE_COUNT) or case_seed != 2026091100 + case_index or menu_seed not in MENU_SEEDS or model not in MODELS or key in seen:
            raise ProductionNotReady(f"manifest row is outside the frozen Wall design: {item}")
        seen.add(key)
        expected.append(dict(item))
    if len({row["case_id"] for row in expected}) != CASE_COUNT or len(seen) != EXPECTED_ROWS:
        raise ProductionNotReady("manifest does not cover exactly 30 cases x 3 menus x 2 models")
    return expected


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "item"


def _result_path(run_dir: Path, item: Mapping[str, Any]) -> Path:
    label = item.get("decision_id") or f"{item['case_id']}__seed{item['case_seed']}__menu{item['menu_seed']}__{item['model']}"
    return run_dir / "results" / f"{_safe_name(str(label))}.json"


def _attempt_dir(run_dir: Path, index: int, item: Mapping[str, Any]) -> Path:
    key = f"{item['case_id']}::{item['case_seed']}::{item['menu_seed']}::{item['model']}"
    return run_dir / "attempts" / f"{index:05d}_{_safe_name(key)}"


def _latest_attempt(directory: Path) -> dict[str, Any] | None:
    values: list[dict[str, Any]] = []
    for path in directory.glob("*/attempt.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            values.append(value)
    return max(values, key=lambda value: int(value.get("attempt", 0)), default=None)


def load_verified_records(run_dir: Path, frozen: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # ``protocol.json`` is an immutable design-freeze record and may remain
    # execution_ready=false.  If the runner has emitted final bindings, honor
    # their explicit readiness flag; otherwise the complete verified-result
    # grid below is the execution gate.
    bindings_path = run_dir / "execution_bindings.json"
    bindings: Mapping[str, Any] | None = None
    if bindings_path.exists():
        try:
            loaded = json.loads(bindings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProductionNotReady(f"invalid execution_bindings.json: {exc}") from exc
        if not isinstance(loaded, Mapping) or loaded.get("execution_ready") is not True:
            raise ProductionNotReady("execution_bindings.json is present but execution_ready is not true")
        bindings = loaded
    status_path = run_dir / "status.json"
    try:
        run_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionNotReady(f"missing or invalid finalized status.json: {exc}") from exc
    if not isinstance(run_status, Mapping) or run_status.get("status") not in {"complete", "complete_with_failures"}:
        raise ProductionNotReady("run status is not finalized; running, paused, or blocked runs cannot be analyzed")
    if run_status.get("pending") not in (None, 0):
        raise ProductionNotReady("finalized status still reports pending manifest items")
    protocol_path = run_dir / "protocol.json"
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionNotReady(f"missing or invalid protocol.json: {exc}") from exc
    frozen_hash = sha256_file(FROZEN_PATH).lower()
    protocol_frozen_hash = str(protocol.get("diagnostic", {}).get("frozen_sha256", "")).lower() if isinstance(protocol, Mapping) else ""
    if protocol_frozen_hash != frozen_hash:
        raise ProductionNotReady("protocol frozen threshold hash does not match frozen_near_tie.json")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    items = validate_manifest(manifest)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    terminal_failures: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        path = _result_path(run_dir, item)
        receipt = _latest_attempt(_attempt_dir(run_dir, index, item))
        if receipt is not None and receipt.get("item") is not None and receipt.get("item") != item:
            raise ProductionNotReady(f"attempt receipt identity mismatch for {item['case_id']} menu {item['menu_seed']} {item['model']}")
        if not path.exists():
            if receipt and receipt.get("status") == "failed" and receipt.get("finished_at"):
                terminal_failures.append({"item": dict(item), "attempt": receipt.get("attempt"), "status": receipt.get("status"), "failure_class": receipt.get("failure_class"), "error": receipt.get("error")})
                continue
            unfinished.append({"item": dict(item), "receipt": receipt})
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            # Only final files under results/ are evidence. Attempts, logs, and
            # partially written JSON are intentionally ignored.
            if any(result.get(key) != item[key] for key in ("case_id", "case_seed", "menu_seed", "model")):
                raise ValueError("identity mismatch")
            if "decision_id" in item and result.get("decision_id") != item["decision_id"]:
                raise ValueError("decision_id mismatch")
            evidence_hash = hashlib.sha256(canonical(result)).hexdigest()
            if not receipt or receipt.get("status") != "complete" or receipt.get("result_sha256", "").lower() != evidence_hash:
                raise ValueError("final result lacks a matching complete attempt receipt/hash")
            records.append(record_from_result(item, result, frozen))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ProductionNotReady(f"invalid or unverifiable final result {path}: {exc}") from exc
    if unfinished:
        raise ProductionNotReady(f"{len(unfinished)} manifest items are unattempted, running, or interrupted")
    source = {
        "manifest": manifest, "run_status": dict(run_status),
        "verified_results_gate": "every manifest item has a final result with matching complete receipt/hash or a terminal failed receipt",
        "valid_result_rows": len(records), "terminal_failure_rows": len(terminal_failures),
        "terminal_failures": terminal_failures,
    }
    if bindings is not None:
        source["execution_bindings"] = bindings
        source["execution_bindings_sha256"] = sha256_file(bindings_path)
    return records, source


def _case_means(records: list[Mapping[str, Any]], outcome: str, *, model: str | None = None) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in records:
        if model is None or row["model"] == model:
            grouped.setdefault(str(row["case_id"]), []).append(float(row[outcome]))
    cases = sorted(grouped)
    return cases, [float(np.mean(grouped[case])) for case in cases]


def flagged_sensitivity(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = ("p_signed_native_menu_gap", "o_signed_native_menu_gap", "o_minus_p_native_utility", "o_minus_p_binary_success")
    flagged: dict[str, dict[str, list[float]]] = {"flagged": {}, "unflagged": {}}
    for group in flagged:
        for outcome in outcomes:
            flagged[group][outcome] = []
    by_case: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in records:
        group = "flagged" if row["near_tie_flagged"] else "unflagged"
        for outcome in outcomes:
            flagged[group][outcome].append(float(row[outcome]))
        case = str(row["case_id"])
        by_case.setdefault(case, {"flagged": {o: [] for o in outcomes}, "unflagged": {o: [] for o in outcomes}})
        for outcome in outcomes:
            by_case[case][group][outcome].append(float(row[outcome]))
    complete_cases = [case for case in sorted(by_case) if by_case[case]["flagged"][outcomes[0]] and by_case[case]["unflagged"][outcomes[0]]]
    differences = {outcome: [float(np.mean(by_case[case]["flagged"][outcome]) - np.mean(by_case[case]["unflagged"][outcome])) for case in complete_cases] for outcome in outcomes}
    return {
        "row_denominators": {group: len(values[outcomes[0]]) for group, values in flagged.items()},
        "case_denominators": {group: sum(bool(by_case[case][group][outcomes[0]]) for case in by_case) for group in ("flagged", "unflagged")},
        "paired_case_denominator": len(complete_cases), "paired_cases": complete_cases,
        "row_means": {group: {o: float(np.mean(values[o])) if values[o] else None for o in outcomes} for group, values in flagged.items()},
        "flagged_minus_unflagged_case_effects": {o: _stats(values, f"flagged:{o}") for o, values in differences.items()},
    }


def analyze_records(records: list[Mapping[str, Any]], frozen: Mapping[str, Any], *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    coverage = {case: sum(1 for row in records if row["case_id"] == case) for case in sorted({str(row["case_id"]) for row in records})}
    complete_cases = [case for case, count in coverage.items() if count == len(MODELS) * len(MENU_SEEDS)]
    complete_records = [row for row in records if str(row["case_id"]) in complete_cases]
    outcomes = {
        "o_minus_p_native_utility": "O−P native utility",
        "o_minus_p_binary_success": "O−P binary success",
        "p_minus_uniform_native_utility": "P−uniform native utility",
        "p_minus_uniform_binary_success": "P−uniform binary success",
    }
    pooled: dict[str, Any] = {}
    per_model: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    case_values: dict[str, list[float]] = {}
    for outcome, label in outcomes.items():
        _, values = _case_means(complete_records, outcome)
        case_values[outcome] = values
        pooled[outcome] = _stats(values, f"pooled:{outcome}")
        for model in MODELS:
            _, model_values = _case_means(complete_records, outcome, model=model)
            per_model[model][outcome] = _stats(model_values, f"{model}:{outcome}")
    pooled["holm_adjusted_p"] = {
        "O_minus_P_family": _holm({"native_utility": pooled["o_minus_p_native_utility"]["two_sided_sign_flip_p"], "binary_success": pooled["o_minus_p_binary_success"]["two_sided_sign_flip_p"]}),
        "P_minus_uniform_family": _holm({"native_utility": pooled["p_minus_uniform_native_utility"]["two_sided_sign_flip_p"], "binary_success": pooled["p_minus_uniform_binary_success"]["two_sided_sign_flip_p"]}),
    }
    source_data = dict(source or {})
    diagnostics = {
        "valid_rows": len(records), "expected_rows": EXPECTED_ROWS, "case_count": len(coverage), "complete_case_count": len(complete_cases),
        "primary_support_threshold_cases": MIN_PRIMARY_CASES, "primary_support_quality_passed": len(complete_cases) >= MIN_PRIMARY_CASES,
        "coverage_by_case": coverage,
        "coverage_by_case_model": {case: {model: sum(1 for row in records if row["case_id"] == case and row["model"] == model) for model in MODELS} for case in coverage},
        "coverage_by_model": {model: sum(row["model"] == model for row in records) for model in MODELS},
        "complete_case_ids": complete_cases,
        "incomplete_case_ids": [case for case, count in coverage.items() if count != 6],
        "terminal_failure_rows": int(source_data.get("terminal_failure_rows", 0)),
        "conditional_on_valid_case_limitation": "Per-case means condition on verified final rows; missing physical outcomes are not imputed.",
    }
    raw_scores = [row["near_tie_score"] for row in records]
    switches = [row["p_choice_o_choice_switch"] for row in records]
    raw_margins = [(-float(row["raw_margin"]) if row["raw_margin"] is not None else float("nan")) for row in records]
    raw_auroc = {"near_tie_score": _auroc(raw_scores, switches), "negative_raw_margin": _auroc(raw_margins, switches)}
    brier = {}
    for key in ("frozen_probability_near_tie", "frozen_probability_raw_margin"):
        pairs = [(float(row[key]), bool(row["p_choice_o_choice_switch"])) for row in records if row[key] is not None]
        brier[key] = float(np.mean([(p - y) ** 2 for p, y in pairs])) if pairs else None
    return {
        "analysis_name": "Wall v1 exploratory decision audit analysis", "status": "complete_exploratory" if len(complete_cases) >= MIN_PRIMARY_CASES else "incomplete_exploratory", "scientific_result": False,
        "source": dict(source or {}), "frozen_threshold_source": {"path": str(FROZEN_PATH), "sha256": sha256_file(FROZEN_PATH)},
        "analysis_plan_source": {"path": str(PLAN_PATH), "sha256": sha256_file(PLAN_PATH)},
        "protocol": {"case_unit": "case_id", "case_aggregation": "mean over 3 menu seeds x 2 models", "candidate_count": 300, "primary_support_cases": MIN_PRIMARY_CASES},
        "uncertainty": {"bootstrap_replicates": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED, "sign_flip_seed": SIGN_FLIP_SEED, "monte_carlo_sign_flips": MONTE_CARLO_SIGN_FLIPS, "intervals": "unadjusted percentile 95% CI", "sign_flip_assumption": "paired case-level sign symmetry/exchangeability under the null"},
        "coverage": diagnostics, "primary": {"pooled": pooled, "model_specific": per_model, "complete_case_ids": complete_cases},
        "diagnostics": {"raw_auroc_for_switch": raw_auroc, "raw_score_definitions": {"near_tie_score": "frozen Push-T score", "negative_raw_margin": "- raw P margin after distinct physical-action deduplication"}, "frozen_logistic_brier": brier, "switch_count": int(sum(switches)), "switch_rate": float(np.mean(switches)) if switches else None},
        "flagged_vs_unflagged_regret": flagged_sensitivity(list(complete_records)),
        "interpretation_guards": ["Exploratory Wall analysis; no Wall threshold retuning or result-dependent subset selection.", "Only finalized runs are analyzed; unattempted, running, or interrupted rows block the final gate.", "Primary effects use only complete six-record case clusters; terminal failures remain in coverage and are never imputed.", "Native utility uses -state distance; binary success is a simulator outcome.", "Near-tie flags and switch prediction do not establish physical risk; no physical-oracle claim is made.", "Per-model contrasts are descriptive; planned Holm families apply only to the two pooled O−P and two pooled P−uniform outcomes.", "This new Wall data is not assumed to be positive before analysis."],
    }


def markdown(result: Mapping[str, Any]) -> str:
    pooled = result["primary"]["pooled"]
    cov = result["coverage"]
    def line(label: str, key: str) -> str:
        item = pooled[key]
        ci = item["percentile_bootstrap_95_ci_unadjusted"]
        return f"- {label}: mean {item['mean']:.6g}, unadjusted 95% CI [{ci[0]:.6g}, {ci[1]:.6g}], sign-flip p={item['two_sided_sign_flip_p']:.6g}"
    return "\n".join([
        "# Wall v1 exploratory decision audit",
        "",
        "This analysis consumes only finalized-run artifacts: each manifest item must have either a final result whose hash matches a complete attempt receipt or a terminal failed receipt. It averages valid outcomes over three menu seeds and two predictors within complete cases, then treats the case as the inferential unit.",
        "",
        f"Coverage: {cov['valid_rows']}/{cov['expected_rows']} manifest rows have valid final results, {cov.get('terminal_failure_rows', 0)} have terminal failures, and {cov['complete_case_count']}/{cov['case_count']} cases have all six records. Primary support quality requires at least {cov['primary_support_threshold_cases']} complete cases and is {'met' if cov['primary_support_quality_passed'] else 'not met'}. Terminal failures and incomplete cases are reported; missing physical outcomes are not imputed.",
        "",
        "Pooled case-level contrasts:",
        "",
        line("O−P native negative-distance utility", "o_minus_p_native_utility"),
        line("O−P binary success", "o_minus_p_binary_success"),
        line("P−uniform native negative-distance utility", "p_minus_uniform_native_utility"),
        line("P−uniform binary success", "p_minus_uniform_binary_success"),
        "",
        f"The planned Holm families are reported in the JSON under `primary.pooled.holm_adjusted_p`: one family for the two O−P outcomes and one for the two P−uniform outcomes. Per-model results are descriptive. Raw AUROC diagnostics and the frozen Push-T logistic scores are reported separately from physical outcomes.",
        "",
        "The frozen Push-T thresholds and logistic coefficients were applied once without Wall retuning. Flagged-versus-unflagged native menu gaps are a sensitivity diagnostic; they do not establish physical risk. The Wall data was not assumed to be positive before analysis, and this exploratory analysis does not make a physical-oracle or generalization claim.",
        "",
    ]) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    started = time.perf_counter()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    if frozen.get("frozen_before_wall_outcomes") is not True:
        raise ValueError("frozen near-tie file is not outcome-blind")
    records, source = load_verified_records(args.run_dir.resolve(), frozen)
    source = {**source, "run_dir": str(args.run_dir.resolve()), "protocol_sha256": sha256_file(args.run_dir / "protocol.json"), "manifest_sha256": sha256_file(args.run_dir / "manifest.json")}
    result = analyze_records(records, frozen, source=source)
    result["runtime"] = {"elapsed_seconds": time.perf_counter() - started, "device": "CPU"}
    output = (args.output or (args.run_dir / "analysis.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary:
        args.summary.resolve().write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "complete_cases": result["coverage"]["complete_case_count"], "elapsed_seconds": result["runtime"]["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
