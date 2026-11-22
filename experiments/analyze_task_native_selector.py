"""CPU-only, posthoc task-native selector diagnostic from immutable saved evidence."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.dont_write_bytecode = True
SPEC = "experiments/task_native_selector_spec_20261002.json"
SEALED = "experiments/official_pusht_v14_recovery_analysis_20260901.json"
DISTANCE = "negative_native_terminal_distance"
SECONDARY = {"pusht": "cumulative_official_task_reward", "wall": "native_terminal_success"}
CONTRASTS = ("a_real_minus_a_pred", "a_state_minus_a_real", "a_state_minus_a_pred")
PAIRS = (("a_pred", "a_real"), ("a_real", "a_state"), ("a_pred", "a_state"))
MODELS = ("jepa", "dino")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action_hashes(actions):
    return [hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest() for action in actions]


def assert_identical(left, right, label):
    for field in left:
        if field not in right or not np.array_equal(left[field], right[field]):
            raise ValueError(f"Physical evidence mismatch: {label}: {field}")


def first_minimum(distances):
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (300,) or not np.isfinite(values).all():
        raise ValueError("Native distance must contain exactly 300 finite candidates")
    return int(np.argmin(values))


def set_analysis_mask(rows, environment):
    valid = [row for row in rows if row["valid"]]
    groups = defaultdict(list)
    for row in valid:
        groups[row["physical_set_id"] if environment == "pusht" else row["cluster_id"]].append(row)
    eligible = set()
    for key, group in groups.items():
        identities = {(row["menu_seed"], row["model"]) for row in group}
        if environment == "pusht":
            complete = len(group) == 2 and {row["model"] for row in group} == set(MODELS)
        else:
            complete = len(group) == 6 and identities == {(seed, model) for seed in (1, 2, 3) for model in MODELS}
        if complete:
            eligible.add(key)
    for row in rows:
        key = row["physical_set_id"] if environment == "pusht" else row["cluster_id"]
        row["analysis_eligible"] = row["valid"] and key in eligible
    return sorted({row["cluster_id"] for row in rows if environment == "pusht" or row["analysis_eligible"]})


def aggregate(rows, field, clusters, environment, model=None):
    """Fixed 15-cell Push-T contrast denominator; complete-case Wall averaging."""
    models = (model,) if model else MODELS
    values = []
    for cluster in clusters:
        model_means = []
        for name in models:
            selected = [row for row in rows if row["analysis_eligible"] and row["cluster_id"] == cluster and row["model"] == name]
            if environment == "pusht":
                # Missing cells contribute zero contrast evidence, never selector outcomes.
                cells = sorted({(row.get("offset", 0), row["menu_seed"]) for row in rows if row["cluster_id"] == cluster})
                if len(cells) > 15:
                    raise ValueError("Push-T trajectory exceeds its 15 scheduled cells")
                observed = {(row.get("offset", 0), row["menu_seed"]): row[field] for row in selected}
                evidence = [observed.get(cell, 0.0) for cell in cells] + [0.0] * (15 - len(cells))
                # Match the frozen NumPy mean, including zeros in scheduled-cell order.
                model_means.append(float(np.mean(evidence)))
            else:
                if len(selected) != 3:
                    raise ValueError("Wall cluster must contain three verified records per model")
                model_means.append(float(np.mean([row[field] for row in selected])))
        values.append(float(np.mean(model_means)))
    return values


def check_stats(actual, expected, label):
    differences = {}
    for key, value in actual.items():
        if key not in expected or not np.allclose(value, expected[key], atol=1e-10, rtol=0):
            differences[key] = {"actual": value, "expected": expected.get(key)}
    if differences:
        raise ValueError(f"Legacy reproduction failed: {label}: {differences}")


def legacy_pusht(root, rows, clusters):
    # Reuse the frozen pure functions without importing its simulator-related modules.
    source = root / "experiments/analyze_official_pusht_v14.py"
    names = {"_finite", "_sign_flip", "_bootstrap_ci", "_stats", "_holm"}
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError("Missing frozen inference definitions")
    namespace = {"np": np, "hashlib": hashlib, "Any": Any, "Mapping": Mapping, "BOOTSTRAP_REPLICATES": 10000}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    stats_function, holm_function = namespace["_stats"], namespace["_holm"]

    reference = read_json(root / SEALED)
    sections = (("endpoint_correction", "raw_inventory", "raw"), ("predicted_minus_uniform", "P_vs_uniform_usefulness", "usefulness"))
    report = {}
    for name, old_section, seed_label in sections:
        actual = {}
        for outcome, old_label in ((DISTANCE, "Delta_G"), (SECONDARY["pusht"], "Delta_J")):
            values = aggregate(rows, f"{name}__{outcome}", clusters, "pusht")
            stats = stats_function(values, f"{seed_label}:pooled:{old_label}")
            check_stats(stats, reference[old_section][old_label], f"{old_section}.{old_label}")
            actual[old_label] = stats
        actual["holm_adjusted_p"] = holm_function({key: value["two_sided_exact_sign_flip_p"] for key, value in actual.items()})
        check_stats(actual["holm_adjusted_p"], reference[old_section]["holm_adjusted_p"], old_section)
        report[name] = actual
    return report


def base_row(environment, source, *, cluster, set_id, model, valid):
    return {"environment": environment, "decision_id": source["decision_id"], "cluster_id": cluster,
            "physical_set_id": set_id, "model": model, "menu_seed": int(source["menu_seed"]),
            "offset": source.get("offset"), "case_seed": source.get("case_seed"), "valid": valid,
            "analysis_eligible": False, "failure_reason": source.get("failure_first_line"),
            "a_pred": None, "a_real": None, "a_state": None}


def add_old_contrasts(row):
    for outcome, vector in ((DISTANCE, -row["distances"]), (SECONDARY[row["environment"]], row["secondary"])):
        row[f"endpoint_correction__{outcome}"] = float(vector[row["a_real"]] - vector[row["a_pred"]])
        row[f"predicted_minus_uniform__{outcome}"] = float(vector[row["a_pred"]] - np.mean(vector))


def load_pusht(root, bundle):
    verifier = load_module(bundle / "recompute_pusht_bundle.py", "selector_legacy_pusht_bundle")
    binding = read_json(bundle / "provenance.json")
    for name, expected in binding["bundle_sha256"].items():
        if sha256(bundle / name) != expected:
            raise ValueError(f"Push-T input hash mismatch: {name}")
    if sha256(root / SEALED) != binding["source_analysis_sha256_recorded"]:
        raise ValueError("Sealed Push-T analysis hash mismatch")
    # Existing verifier checks original abstention, scores, returns, hashes and all old cluster vectors.
    verified = verifier.recompute(bundle)
    ledger = read_json(bundle / "rows.json")
    with np.load(bundle / "candidates.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    rows, evidence, cells = [], {}, defaultdict(list)
    model_names = {"official_jepa_wm": "jepa", "official_dino_wm": "dino"}
    for source in ledger["rows"]:
        set_id = f"{source['trajectory_id']}::offset{source['offset']}::seed{source['menu_seed']}"
        row = base_row("pusht", source, cluster=source["trajectory_id"], set_id=set_id, model=model_names[source["model"]], valid=source["valid"])
        if row["valid"]:
            index = source["array_index"]
            row.update(a_pred=source["p_index"], a_real=source["o_index"], distances=data["native_distances"][index],
                       secondary=data["native_returns"][index], physical_hashes=action_hashes(data["actions_physical"][index]),
                       predicted_cost=data["captured_p_scores"][index], realized_cost=data["costs"][index, :, 1] + ledger["weight"] * data["costs"][index, :, 3])
            evidence[row["decision_id"]] = {key: data[key][index] for key in ("actions_normalized", "actions_physical", "normalized_action_sha256", "normalized_plan_multiplicity", "final_states", "goal_states", "reward_traces", "native_returns", "native_distances")}
            cells[set_id].append(row)
            add_old_contrasts(row)
        rows.append(row)
    paired = 0
    for key, group in cells.items():
        if len(group) == 2:
            assert_identical(evidence[group[0]["decision_id"]], evidence[group[1]["decision_id"]], key)
            paired += 1
    clusters = set_analysis_mask(rows, "pusht")
    if paired != 281 or len(clusters) != 20 or sum(row["analysis_eligible"] for row in rows) != 562:
        raise ValueError("Push-T support changed")
    legacy = legacy_pusht(root, rows, clusters)
    return rows, clusters, {"counts": verified["counts"], "cross_model_identical_sets": paired,
            "identity_fields": list(next(iter(evidence.values()))), "old_headlines": legacy,
            "limitations": "Initial states and raw observations are absent from compact bundle; not independently revalidated. Source NPZ commitments were recorded at export, not full-source rehashed here."}


def load_wall(bundle):
    manifest = read_json(bundle / "export_manifest.json")
    for item in manifest["files"]:
        if sha256(bundle / item["path"]) != item["sha256"]:
            raise ValueError(f"Wall input hash mismatch: {item['path']}")
    verifier = load_module(bundle / "recompute_wall.py", "selector_legacy_wall_bundle")
    verified = verifier.verify(bundle)
    # Preserve JSONL alignment with arrays before grouping, then validate candidate indices.
    candidates = [json.loads(line) for line in (bundle / "candidate_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[candidate["decision_id"]].append((index, candidate))
    summaries = {row["decision_id"]: row for row in (json.loads(line) for line in (bundle / "row_summaries.jsonl").read_text(encoding="utf-8").splitlines())}
    with np.load(bundle / "candidate_arrays.npz", allow_pickle=False) as archive:
        actions, states = archive["actions_physical"], archive["terminal_state"]
    if actions.shape != (53700, 30, 2) or states.shape != (53700, 2):
        raise ValueError("Wall array dimensions changed")
    failures = {entry["item"]["decision_id"]: entry for entry in read_json(bundle / "metadata/failure_ledger.json")["rows"]}
    rows, evidence, cells = [], {}, defaultdict(list)
    for source in read_json(bundle / "metadata/manifest.json")["rows"]:
        set_id = f"{source['case_id']}::seed{source['menu_seed']}"
        row = base_row("wall", source, cluster=source["case_id"], set_id=set_id, model=source["model"], valid=source["decision_id"] in groups)
        if row["valid"]:
            entries = groups[row["decision_id"]]
            indices, details = zip(*entries)
            if [candidate["candidate"] for candidate in details] != list(range(300)):
                raise ValueError("Wall candidate/array order mismatch")
            hashes = action_hashes(actions[list(indices)])
            if hashes != [candidate["actions_physical_sha256"] for candidate in details]:
                raise ValueError("Wall physical action hash mismatch")
            row.update(a_pred=details[0]["predicted_choice_P"], a_real=details[0]["encoded_real_choice_O"],
                       distances=np.asarray([candidate["native_terminal_distance"] for candidate in details], dtype=np.float64),
                       secondary=np.asarray([candidate["native_terminal_success"] for candidate in details], dtype=np.float64),
                       predicted_cost=np.asarray([candidate["predicted_cost_P"] for candidate in details]),
                       realized_cost=np.asarray([candidate["encoded_real_cost_O"] for candidate in details]), physical_hashes=hashes)
            if not np.array_equal(row["secondary"], row["distances"] < 4.5):
                raise ValueError("Wall completion does not equal strict native-distance threshold")
            summary = summaries[row["decision_id"]]
            evidence[row["decision_id"]] = {"physical_actions": actions[list(indices)], "terminal_state_export_float32": states[list(indices)],
                    "native_distance": row["distances"], "success": row["secondary"], "physical_hashes": hashes,
                    "initial_state": summary["initial_state"], "goal_state": summary["goal_state"]}
            cells[set_id].append(row)
            add_old_contrasts(row)
        else:
            if row["decision_id"] not in failures:
                raise ValueError("Unaccounted missing Wall result")
            row["failure_reason"] = failures[row["decision_id"]]["failure_class"]
        rows.append(row)
    paired = 0
    for key, group in cells.items():
        if len(group) == 2:
            assert_identical(evidence[group[0]["decision_id"]], evidence[group[1]["decision_id"]], key)
            paired += 1
    clusters = set_analysis_mask(rows, "wall")
    if paired != 89 or len(clusters) != 29 or sum(row["analysis_eligible"] for row in rows) != 174:
        raise ValueError("Wall support changed")
    mapping = {"endpoint_correction": "o_minus_p", "predicted_minus_uniform": "p_minus_uniform"}
    for name, old_prefix in mapping.items():
        for outcome, old_suffix in ((DISTANCE, "native_utility"), (SECONDARY["wall"], "binary_success")):
            values = aggregate(rows, f"{name}__{outcome}", clusters, "wall")
            expected = verified["primary_effects"][f"{old_prefix}_{old_suffix}"]["case_values"]
            if not np.allclose(values, expected, atol=1e-12, rtol=0):
                raise ValueError("Wall aggregation does not reproduce original case effects")
    return rows, clusters, {"counts": {"issued": 180, "valid": 179, "invalid": 1, "complete_case_count": 29, "analysis_rows": 174},
            "cross_model_identical_sets": paired, "old_headlines": verified,
            "identity_fields": list(next(iter(evidence.values()))),
            "limitations": "Exported terminal states are float32; native distances remain authoritative stored float64. One failed JEPA row cannot be cross-model verified."}


def preflight(root, spec_path):
    started = utc_now()
    spec = read_json(spec_path)
    if spec["status"] != "post_hoc_specified_before_new_selector_outcome_calculation" or spec["inference"]["new_p_values"] is not False:
        raise ValueError("Unexpected diagnostic specification")
    bundles = {environment: root / relative for environment, relative in spec["inputs"].items()}
    data, clusters, reports = {}, {}, {}
    for environment in ("pusht", "wall"):
        print(f"Preflight: verifying {environment} retained evidence and legacy results", flush=True)
        data[environment], clusters[environment], reports[environment] = (load_pusht(root, bundles[environment]) if environment == "pusht" else load_wall(bundles[environment]))
    sources = [spec_path, root / SEALED, root / "experiments/analyze_official_pusht_v14.py", Path(__file__)]
    test_path = Path(__file__).with_name("test_task_native_selector.py")
    if test_path.is_file():
        sources.append(test_path)
    for bundle in bundles.values():
        sources.extend(sorted(path for path in bundle.rglob("*") if path.is_file()))
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    report = {"status": "passed_before_new_selector_calculation", "started_utc": started, "finished_utc": utc_now(),
              "input_root": str(root), "spec_path": str(spec_path), "spec_sha256": sha256(spec_path), "source_sha256": hashes,
              "environments": reports, "new_selector_outcomes_computed": False,
              "legacy_comparison_absolute_tolerance": 1e-10, "specification": spec}
    return data, clusters, report


def derive_selector(row):
    if not row["valid"]:
        return row
    selected = first_minimum(row["distances"])
    row["a_state"] = selected
    for selector, values in (("a_pred", row["predicted_cost"]), ("a_real", row["realized_cost"]), ("a_state", row["distances"])):
        if not 0 <= row[selector] < 300 or row[selector] != int(np.argmin(values)):
            raise ValueError(f"Selector membership/first-minimum mismatch: {row['decision_id']} {selector}")
        tied = np.flatnonzero(values == np.min(values))
        row[f"{selector}_exact_minimum_tie_count"] = int(len(tied))
        row[f"{selector}_distinct_physical_minimum_tie_count"] = len({row["physical_hashes"][index] for index in tied})
        row[f"{selector}_physical_action_sha256"] = row["physical_hashes"][row[selector]]
    for outcome, vector in ((DISTANCE, -row["distances"]), (SECONDARY[row["environment"]], row["secondary"])):
        values = [float(vector[row[selector]]) for selector in ("a_pred", "a_real", "a_state")]
        for selector, value in zip(("a_pred", "a_real", "a_state"), values):
            row[f"{selector}__{outcome}"] = value
        effects = (values[1] - values[0], values[2] - values[1], values[2] - values[0])
        if not np.isclose(effects[0] + effects[1], effects[2], atol=1e-10, rtol=1e-12):
            raise ValueError("Row contrast closure failed")
        if outcome == DISTANCE and min(effects[1:]) < 0:
            raise ValueError("Negative terminal-distance headroom")
        for contrast, value in zip(CONTRASTS, effects):
            row[f"{contrast}__{outcome}"] = value
    for left, right in PAIRS:
        row[f"{left}_vs_{right}__candidate_index"] = row[left] != row[right]
        row[f"{left}_vs_{right}__physical_action"] = row["physical_hashes"][row[left]] != row["physical_hashes"][row[right]]
    return row


def summarize(rows, clusters, environment, seed):
    draws = np.random.default_rng(seed).integers(0, len(clusters), size=(10000, len(clusters)))
    summaries, cluster_rows = [], []
    for model in (None, *MODELS):
        for outcome in (DISTANCE, SECONDARY[environment]):
            vectors = []
            for contrast in CONTRASTS:
                vector = np.asarray(aggregate(rows, f"{contrast}__{outcome}", clusters, environment, model))
                vectors.append(vector)
                interval = np.quantile(vector[draws].mean(axis=1), [.025, .975])
                summaries.append({"environment": environment, "model": model or "pooled", "outcome": outcome, "contrast": contrast,
                    "mean": float(vector.mean()), "ci_low": float(interval[0]), "ci_high": float(interval[1]), "cluster_count": len(clusters),
                    "bootstrap_seed": seed, "bootstrap_draws": 10000, "interval_type": "posthoc unadjusted percentile 95% whole-cluster bootstrap"})
                cluster_rows.extend({"environment": environment, "model": model or "pooled", "cluster_id": cluster, "outcome": outcome,
                    "contrast": contrast, "effect": float(value)} for cluster, value in zip(clusters, vector))
            if not np.allclose(vectors[0] + vectors[1], vectors[2], atol=1e-10, rtol=1e-12):
                raise ValueError("Cluster contrast closure failed")
    return summaries, cluster_rows, hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()


def disagreements(rows):
    report = []
    for scope in ("all_valid", "analysis_mask"):
        for model in (None, *MODELS):
            selected = [row for row in rows if row["valid"] and (scope == "all_valid" or row["analysis_eligible"]) and (model is None or row["model"] == model)]
            for left, right in PAIRS:
                for definition in ("candidate_index", "physical_action"):
                    count = sum(row[f"{left}_vs_{right}__{definition}"] for row in selected)
                    report.append({"environment": rows[0]["environment"], "scope": scope, "model": model or "pooled", "comparison": f"{left}_vs_{right}",
                                   "definition": definition, "disagreements": count, "denominator": len(selected), "rate": count / len(selected) if selected else None})
    return report


def physical_support(rows):
    report = {}
    for scope in ("all_valid", "analysis_mask"):
        selected = [row for row in rows if row["valid"] and (scope == "all_valid" or row["analysis_eligible"])]
        sets = {row["physical_set_id"]: row for row in selected}
        counts = [len(set(row["physical_hashes"])) for row in sets.values()]
        report[scope] = {"model_rows": len(selected), "physical_candidate_sets": len(sets), "ordered_candidate_instances": 300 * len(sets),
                         "distinct_physical_action_instances_within_sets": sum(counts), "duplicate_action_instances_within_sets": 300 * len(sets) - sum(counts),
                         "independent_clusters": len({row["cluster_id"] for row in selected}),
                         "unique_action_count_per_set_distribution": dict(Counter(counts)),
                         "ties_by_selector": {selector: {"model_rows_with_exact_ties": sum(row[f"{selector}_exact_minimum_tie_count"] > 1 for row in selected),
                                "model_rows_with_distinct_physical_ties": sum(row[f"{selector}_distinct_physical_minimum_tie_count"] > 1 for row in selected),
                                "maximum_exact_minimum_tie_count": max(row[f"{selector}_exact_minimum_tie_count"] for row in selected)} for selector in ("a_pred", "a_real", "a_state")}}
        if rows[0]["environment"] == "wall":
            successful = [row for row in sets.values() if np.any(row["secondary"])]
            report[scope]["completion_support"] = {"distinct_successful_physical_candidates_within_sets": sum(len({digest for digest, success in zip(row["physical_hashes"], row["secondary"]) if success}) for row in successful),
                "physical_sets_with_any_success": len(successful), "cases_with_any_success": sorted({row["cluster_id"] for row in successful}),
                "success_is_strict_distance_threshold": True, "threshold": 4.5,
                "selected_success_model_row_counts": {selector: sum(bool(row["secondary"][row[selector]]) for row in selected) for selector in ("a_pred", "a_real", "a_state")}}
    report["counting_rule"] = "Collapse the two models within each case/offset/seed. Deduplicate physical actions only for support counts; all 300 ordered candidates remain in every selector. Counts across different sets are candidate instances, not independent trials."
    return report


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_markdown(summaries):
    labels = {DISTANCE: "Negative terminal distance", SECONDARY["pusht"]: "Cumulative task reward", SECONDARY["wall"]: "Completion probability"}
    lines = ["| Environment / outcome | Realized − predicted | State − realized | State − predicted |", "|---|---:|---:|---:|"]
    for environment in ("pusht", "wall"):
        for outcome in (DISTANCE, SECONDARY[environment]):
            values = [next(row for row in summaries if row["environment"] == environment and row["outcome"] == outcome and row["contrast"] == contrast and row["model"] == "pooled") for contrast in CONTRASTS]
            cells = [f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]" for row in values]
            lines.append(f"| {'Push-T' if environment == 'pusht' else 'Wall'} / {labels[outcome]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_figure(path, summaries):
    # Standalone vector figure; panels use separate axes because outcome units differ.
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="820" viewBox="0 0 1100 820">',
             '<rect width="1100" height="820" fill="white"/>', '<g font-family="Arial, sans-serif" fill="#202020">',
             '<text x="40" y="35" font-size="28">Retrospective intervention on fixed candidate sets</text>']
    for x, label in ((60, "Predicted latent"), (420, "Realized latent"), (800, "Task-native state")):
        parts.append(f'<text x="{x}" y="85" font-size="25">{label}</text>')
    parts.extend(['<text x="335" y="85" font-size="30">→</text>', '<text x="710" y="85" font-size="30">→</text>',
                  '<text x="40" y="128" font-size="21">One distance-selected action; both physical outcomes evaluated at that same action.</text>',
                  '<text x="40" y="158" font-size="21">Task-native state selection is retrospective and non-deployable.</text>'])
    for index, (environment, outcome) in enumerate((("pusht", DISTANCE), ("pusht", SECONDARY["pusht"]), ("wall", DISTANCE), ("wall", SECONDARY["wall"]))):
        left, top = 40 + (index % 2) * 550, 215 + (index // 2) * 255
        rows = [next(row for row in summaries if row["environment"] == environment and row["outcome"] == outcome and row["contrast"] == contrast and row["model"] == "pooled") for contrast in CONTRASTS]
        lo = min(0, *(row["ci_low"] for row in rows)); hi = max(0, *(row["ci_high"] for row in rows)); span = hi - lo or 1
        lo -= .08 * span; hi += .08 * span
        scale = lambda value: left + 158 + 307 * (value - lo) / (hi - lo)
        title = f"{'Push-T' if environment == 'pusht' else 'Wall'}: " + ("negative terminal distance" if outcome == DISTANCE else "cumulative reward" if environment == "pusht" else "completion probability")
        parts.append(f'<text x="{left}" y="{top}" font-size="24">{title}</text>')
        parts.append(f'<line x1="{scale(0):.2f}" x2="{scale(0):.2f}" y1="{top+22}" y2="{top+171}" stroke="#999" stroke-dasharray="4 4"/>')
        for offset, (row, label) in enumerate(zip(rows, ("Real − pred", "State − real", "State − pred"))):
            y = top + 48 + offset * 53
            parts.append(f'<text x="{left}" y="{y+6}" font-size="21">{label}</text>')
            x1, x2, xm = scale(row["ci_low"]), scale(row["ci_high"]), scale(row["mean"])
            parts.append(f'<path d="M{x1:.2f} {y}H{x2:.2f} M{x1:.2f} {y-6}V{y+6} M{x2:.2f} {y-6}V{y+6}" stroke="#202020" stroke-width="1.7" fill="none"/>')
            parts.append(f'<circle cx="{xm:.2f}" cy="{y}" r="5" fill="#202020"/>')
        for value in np.linspace(lo, hi, 4):
            parts.append(f'<text x="{scale(value):.2f}" y="{top+200}" font-size="20" text-anchor="middle">{value:.2f}</text>')
    parts.extend(['<text x="40" y="722" font-size="20">Points: mean outcome differences. Bars: posthoc unadjusted 95% whole-cluster intervals.</text>',
                  '<text x="40" y="751" font-size="20">10,000 resamples; Push-T: 20 trajectories; Wall: 29 complete cases. Higher is better.</text>',
                  '<text x="40" y="780" font-size="20">Distance gains follow by construction. Wall completion is a distance threshold;</text>',
                  '<text x="40" y="807" font-size="20">successful candidate support is limited to two cases.</text>', '</g></svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def write_interpretation(path, summaries, support, preflight_report):
    table = table_markdown(summaries)
    lookup = {(row["environment"], row["outcome"], row["contrast"]): row for row in summaries if row["model"] == "pooled"}
    def effect(environment, outcome, contrast):
        row = lookup[environment, outcome, contrast]
        return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
    text = f"""# Task-native retrospective selector: results and interpretation

The state selector improves its chosen terminal-distance metric in both environments, decreases Push-T cumulative task reward, and adds no Wall completion beyond realized-latent selection. Physical conclusions therefore depend on the objective being evaluated; the diagnostic does not demonstrate a generally better physical decision.

This is a new posthoc camera-ready diagnostic specified before new selector calculation. It uses simulator-realized candidate outcomes and is not deployable. It is an upper bound only for terminal-distance utility within each retained fixed candidate set, not a globally optimal policy.

## What changed and what stayed fixed

One additional selector chooses the first exact minimum of stored native terminal distance over the same ordered 300 candidates. Both physical outcomes are evaluated at that one index; no reward-specific or completion-specific selector is substituted. Candidate generation, original predicted/realized selectors, goals, encoders, validity rules and original results remain unchanged. No simulation, inference, training, CEM updates or closed-loop tests were run.

{table}

Entries are mean effects [posthoc unadjusted 95% whole-cluster percentile intervals]. No new p-values were computed. Original bootstrap intervals and tests were reproduced separately in `preflight.json`; these diagnostic intervals do not replace them.

## Findings and limits

For Push-T, endpoint correction changes native-distance utility by {effect('pusht', DISTANCE, CONTRASTS[0])}; remaining distance headroom is {effect('pusht', DISTANCE, CONTRASTS[1])}. The same state-selected action changes cumulative task reward relative to realized-latent selection by {effect('pusht', SECONDARY['pusht'], CONTRASTS[1])}. Reward and terminal distance are different objectives: the native whole-state distance mixes agent position, block position, raw angle and velocity, and is not a pure manipulation-completion metric. A distance gain follows from selection by construction; its magnitude and the secondary reward effect are empirical findings.

For Wall, remaining distance headroom is {effect('wall', DISTANCE, CONTRASTS[1])}, and the completion difference is {effect('wall', SECONDARY['wall'], CONTRASTS[1])}. Completion is exactly the strict threshold distance < 4.5, so its optimization follows from minimum-distance selection too; it is not independent cross-objective corroboration. Only five distinct successful physical candidate instances appear in four candidate sets across two cases (wall_011 and wall_019). Completion-rate conclusions remain weak.

The intervention bypasses the learned representation and latent objective jointly. Remaining distance headroom is compatible with mismatch at that combined interface; it does not identify the representation alone, the objective alone, or DINOv2 as a cause. It does not establish a deployable fix, performance of full CEM/MPC, closed-loop benefit, or broad generality from two simulation environments. Adverse or null secondary outcomes remain in the table.

## Dependence, missingness and data checks

Push-T retains all 600 issued rows: 577 valid, 23 invalid, 281 paired cells (562 model rows), and 15 unmatched valid rows. Contrasts are averaged equally across models under the same paired mask and divided by 15 scheduled cells per trajectory across all 20 trajectories. Missing cells provide zero contrast evidence under that inherited estimand; no missing selector outcome is assigned zero. Wall retains 179 valid rows and one failed row. Its primary analysis uses 29 complete six-row cases (174 records), excluding wall_028 rather than filling its missing JEPA result.

Cross-model arrays and native outcomes agree exactly for all 281 Push-T pairs and 89 available Wall pairs. Candidate-index and distinct-physical-action disagreement are reported separately, with all-valid and analysis-mask denominators. Duplicate plans remain in selector populations, and physical-support counts collapse duplicated model evidence. Exact-minimum ties are reported in `physical_support.json` and row output. Row and cluster contrasts satisfy endpoint correction + remaining headroom = total headroom.

All committed compact-bundle hashes and original headline means, cluster vectors, confidence intervals, sign-flip tests and original Holm families passed before new selector outcomes were computed. No evidence discrepancy was found. Push-T initial states/raw observations are absent from the compact bundle and were not independently checked here; full source NPZ hashes remain recorded export commitments. Wall exported terminal states are rounded float32, so the analysis uses stored native distances rather than recomputing them from rounded states. Push-T rewards retain the original sequential float32 accumulation. Source commitments, specification hash, runtime and UTC timestamps are recorded in `preflight.json` and `results.json`.
"""
    path.write_text(text, encoding="utf-8")


def validate_output_directory(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    bundles = (root / "output/pusht_mechanism_20260929/bundle", root / "output/wall_recompute_20260910_v3")
    if output == root or any(output.is_relative_to(bundle.resolve()) for bundle in bundles):
        raise ValueError("Output must be outside immutable input bundles")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "analyze"))
    parser.add_argument("--input-root", "--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    validate_output_directory(root, output)
    output.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    data, clusters, checked = preflight(root, root / SPEC)
    write_json(output / "preflight.json", checked)
    write_json(output / "original_results.json", {environment: report["old_headlines"] for environment, report in checked["environments"].items()})
    if args.stage == "preflight":
        print(json.dumps({"status": checked["status"], "output": str(output / "preflight.json")}))
        return 0
    print("Preflight passed. Calculating the posthoc state-selector outcomes.", flush=True)
    if sha256(root / SPEC) != checked["spec_sha256"]:
        raise ValueError("Specification changed during preflight")
    all_summary, all_clusters, all_disagreements, support, bootstrap_hashes = [], [], [], {}, {}
    for environment, seed in (("pusht", 20261002), ("wall", 20261003)):
        for row in data[environment]:
            derive_selector(row)
        summary, cluster_rows, draw_hash = summarize(data[environment], clusters[environment], environment, seed)
        all_summary.extend(summary); all_clusters.extend(cluster_rows)
        all_disagreements.extend(disagreements(data[environment]))
        support[environment] = physical_support(data[environment])
        bootstrap_hashes[environment] = draw_hash
    array_fields = {"distances", "secondary", "physical_hashes", "predicted_cost", "realized_cost"}
    output_rows = [{key: value for key, value in row.items() if key not in array_fields} for rows in data.values() for row in rows]
    result = {"analysis": checked["specification"]["analysis"], "status": "complete_posthoc", "started_utc": started, "finished_utc": utc_now(),
              "spec_sha256": checked["spec_sha256"], "preflight_sha256": sha256(output / "preflight.json"),
              "software": {"python": sys.version, "numpy": np.__version__}, "new_p_values_computed": False,
              "summary": all_summary, "cluster_effects": all_clusters, "selection_disagreements": all_disagreements, "physical_support": support,
              "bootstrap": {"draws": 10000, "seeds": {"pusht": 20261002, "wall": 20261003}, "resampling_index_sha256": bootstrap_hashes,
                            "shared_indices": "Same cluster resampling indices for every outcome, contrast and model within each environment"}}
    write_json(output / "results.json", result)
    write_json(output / "physical_support.json", support)
    write_csv(output / "row_effects.csv", output_rows)
    write_csv(output / "cluster_effects.csv", all_clusters)
    write_csv(output / "summary.csv", all_summary)
    write_csv(output / "selection_disagreements.csv", all_disagreements)
    (output / "compact_table.md").write_text(table_markdown(all_summary) + "\n", encoding="utf-8")
    write_figure(output / "intervention_ladder.svg", all_summary)
    write_interpretation(output / "RESULTS_INTERPRETATION.md", all_summary, support, checked)
    script = Path(__file__).resolve()
    commands = ["$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'", f'& "{sys.executable}" -B "{script.with_name("test_task_native_selector.py")}"',
                *[f'& "{sys.executable}" -B "{script}" {stage} --input-root "{root}" --output-dir "{output}"' for stage in ("preflight", "analyze")]]
    (output / "REPRODUCE.md").write_text("# Reproduction\n\nRun in PowerShell; Python with NumPy is sufficient. Inputs are read-only.\n\n```powershell\n" + "\n".join(commands) + "\n```\n\nAnalysis reruns preflight before computing any new outcomes. A discrepancy raises an error before new calculation. The original frozen analysis helpers are imported only for legacy inference.\n", encoding="utf-8")
    files = [path for path in output.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    write_json(output / "artifact_manifest.json", {"created_utc": utc_now(), "spec_sha256": checked["spec_sha256"],
        "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)]})
    print(json.dumps({"status": result["status"], "output": str(output), "summary": [row for row in all_summary if row["model"] == "pooled"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
