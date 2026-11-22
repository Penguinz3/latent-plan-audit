# LatentPlanAudit

Decision-level diagnostics for testing whether correcting a latent world model's prediction also corrects the action selected by a robot planner.

This repository accompanies **Decision-Level Auditing of Latent Robot World Models** and its workshop versions. The original IROS PWMS paper and later PhysWorldAI, RoboPAD, and UncertainWM versions are retained separately.

- [Project page](https://penguinz3.github.io/latent-plan-audit/)
- [Original IROS PWMS paper](paper/decision_level_auditing_iros_pwms.pdf) and [source](paper/source/main.tex)
- [Latest final papers and LaTeX sources, October 22](paper/2026-10-22/README.md): [PhysWorldAI](paper/2026-10-22/physworldai/decision_level_auditing_physworldai_camera_ready_final_20261022.pdf), [RoboPAD](paper/2026-10-22/robopad/decision_level_auditing_robopad_camera_ready_final_20261022.pdf), [UncertainWM](paper/2026-10-22/uncertainwm/decision_level_auditing_uncertainwm_camera_ready_descriptive_20261022.pdf)
- [October 9 preprint](preprint/2026-10-09/decision_level_auditing_preprint.pdf)
- **[Download compact numerical reproduction](https://github.com/Penguinz3/latent-plan-audit/releases/download/v2026.11.22/latent-plan-audit-reproduction-2026-11-22.zip)** (110 MB compressed; [instructions and checksums](reproduction/2026-11-22/README.md))
- [Reusable audit tool](latent_score_decomposition.py)
- [Push-T and Wall experiment source and post-hoc analyses](experiments/README.md)
- [Frozen study specification and sealed summary](frozen_study/)

## What the audit asks

For a fixed action menu, the study compares:

- the action selected from predicted terminal representations;
- the action selected after the same cost is applied to simulator-realized, re-encoded endpoints;
- terminal goal-distance utility and task reward or completion of those selected actions.

This separates prediction fidelity from decision quality without retraining the model or changing its candidate actions. In the Push-T study, the released latent planners beat uniform menu selection, but replacing their predicted terminal latents with realized endpoints did not establish aggregate physical recovery. The final papers add Wall and clearly separated post-hoc component and hindsight state-distance diagnostics. The original files use P/O for predicted/realized scoring and G/J for distance utility/reward; that API remains unchanged.

The hindsight state selector jointly bypasses the learned representation and latent objective. Its native-distance advantage is built into its selection rule; it reduces Push-T cumulative reward and adds no observed Wall completions over realized-latent selection. It is not a deployable fix or an isolated test of DINOv2.

## Reusable tool

`latent_score_decomposition.py` implements an exact decomposition of the change in an MSE planning score:

```text
P - O = 2 mean((z - g)(zhat - z)) + mean((zhat - z)^2)
```

It also provides pair-margin accounting, regret calculations, and a conservative taxonomy for distinguishing preserved decisions from rollout- or objective-mediated reversals.

```python
from latent_score_decomposition import decompose_latent_scores, pair_margin_audit

decomposition = decompose_latent_scores(
    predicted_visual=predicted_latents,
    realized_visual=realized_latents,
    goal_visual=goal_latent,
)

pair_report = pair_margin_audit(
    decomposition["predicted_score"],
    decomposition["observed_score"],
)
```

The tool expects NumPy arrays with candidates on axis zero. Visual latents are required; an optional proprioceptive term can be supplied with its planning weight.

## Quick check

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The focused tests verify the exact score identity, tie behavior, pairwise accounting, near-tie helpers, terminal reward, and Wall analysis helpers. Run `python -B experiments/test_task_native_selector.py` for the standalone scientific-invariant checks. For numerical reproduction from saved evidence, use the separate [download instructions](reproduction/2026-11-22/README.md).

## Repository layout

```text
latent_score_decomposition.py   reusable decision-audit functions
tests/                          focused unit tests
experiments/                    Push-T and Wall execution/analysis source
scripts/                        compact evidence exporters and recomputation
frozen_study/                   preregistration, issuance, manifest, and sealed summary
figures/                        paper and exploratory figures
paper/                          original PWMS paper and dated final versions/sources
reproduction/2026-11-22/         release instructions, manifests, and small result tables
```

## Reproducibility boundary

The November 22 release includes a 110 MB compressed package of saved candidate evidence, frozen specifications, protocol records, exact analyzers, and approved outputs. Verification from a relocated directory reproduced six numerical artifacts byte for byte and every declared numerical section of the diagnostic results. The package uses NumPy and requires no GPU.

This is numerical recomputation from saved evidence, not an independent rerun of model predictions or simulator trajectories. The original execution scripts require upstream code, checkpoints, data, and historical path/hash bindings that the lightweight environment does not reconstruct. See [experiment source notes](experiments/README.md).

The approximately 241 GB of raw per-candidate arrays are **not uploaded**. The compact export retains quantities needed for the included recomputation, not every raw observation or latent tensor. Source hashes are provenance commitments, not a replacement for the source files. Keep a separate backup of raw evidence, checkpoints, original runtime, and any other unpublished work before deleting the original project folder. This public repository and download are not a complete backup of that folder.

The component, near-tie, and state-selector analyses are explicitly post-hoc. Original inference and later descriptive intervals are reported separately. Near-ties indicate selector instability, not physical risk. Retained protocol records are not an independently timestamped preregistration registry; November 22 is their packaging date, not proof of earlier public registration.

## Citation

See [`CITATION.cff`](CITATION.cff). The workshop is non-archival; this repository provides the stable public paper and project record requested by PWMS.
