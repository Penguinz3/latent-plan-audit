# Experiment and analysis source

This directory preserves the Push-T fixed-menu audit, its recovery adapter, Wall extension, and later post-hoc analyses. The November 22 release adds unchanged local source files. It does not rerun simulations or change any submitted paper.

## Recompute the numerical results

Use the self-contained [numerical reproduction package](../reproduction/2026-11-22/README.md). It includes the frozen specification, compact saved inputs, the exact analyzer versions used for that package, protocol history, and approved outputs. NumPy is its only third-party dependency. Its input paths resolve inside the extracted package.

The standalone scientific-invariant checks can also be run from this repository:

```bash
python -B experiments/test_task_native_selector.py
```

The default `pytest` suite covers the reusable score tool, near-tie helpers, terminal-reward analysis, and Wall analysis helpers. These CPU tests are distinct from the numerical reproduction and from simulator execution.

## Source map

| Source | Purpose |
|---|---|
| `official_pusht_v14_*`, `official_pusht_v12_common_initial.py`, `official_pusht_*` | Original fixed-menu execution, validation, issuance, and shared primitives |
| `analyze_official_pusht_v14.py` | Frozen original Push-T inference |
| `analyze_official_pusht_v14_recovery.py` | Recovery evidence adapter preserving that inference |
| `wall_decision_audit.py`, `run_wall_audit.py`, `analyze_wall_audit.py` | Wall extension |
| `analyze_pusht_component_posthoc.py`, `analyze_pusht_component_context.py`, `analyze_pusht_terminal_reward.py` | Post-hoc physical-component and reward diagnostics |
| `analyze_pusht_margin_instability.py`, `validate_pusht_near_tie_instability.py` | Post-hoc prediction-only selector-instability diagnostic |
| `analyze_task_native_selector.py`, `test_task_native_selector.py`, `render_task_native_figures.py` | Post-hoc hindsight state-distance selector and figures |
| `../scripts/export_*`, `../scripts/recompute_*` | Compact candidate export and numerical recomputation |

## Original execution environment

The original execution and recovery scripts require the original upstream JEPA-WM/DINO-WM simulator runtime, checkpoints, dataset, planning configurations, and retained artifacts. They contain historical project-path and source-hash bindings. This repository is not a turnkey reconstruction of that GPU environment. Do not run its launchers expecting new experiments from the lightweight `requirements.txt` environment.

The portable package retains the planning configurations and protocol records for inspection. Its reproduction path parses the original Push-T inference source for pure statistical functions instead of importing its simulator dependencies. Exporters require the original retained artifacts; downloading the existing compact package avoids that requirement.

The component, near-tie, and state-selector findings are post-hoc. Near-ties describe selector instability, not physical risk. The hindsight state selector jointly bypasses the learned representation and latent objective; it is not a deployable remedy or an isolated test of the visual encoder.
