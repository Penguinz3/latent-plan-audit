# LatentPlanAudit

Decision-level diagnostics for testing whether correcting a latent world model's prediction also corrects the action selected by a robot planner.

This repository accompanies **Decision-Level Auditing of Latent Robot World Models**, accepted at the **IROS 2026 Workshop on Physical World Models for Scaling Embodied AI (PWMS)**.

- [Public workshop paper](paper/decision_level_auditing_iros_pwms.pdf)
- [Paper source](paper/source/main.tex)
- [Reusable audit tool](latent_score_decomposition.py)
- [Push-T experiment and post-hoc analyses](experiments/)
- [Frozen study specification and sealed summary](frozen_study/)

## What the audit asks

For a fixed action menu, the study compares:

- **P:** the action selected from predicted terminal latents;
- **O:** the action selected after the same cost is applied to simulator-realized, re-encoded endpoints;
- **G/J:** the physical outcomes of the two selected actions.

This separates prediction fidelity from decision quality without retraining the model or changing its candidate actions. In the Push-T study, the released latent planners beat uniform menu selection, but replacing their predicted terminal latents with realized endpoints did not establish aggregate physical recovery. A post-hoc component analysis is included separately from the accepted paper.

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

The focused tests verify the exact score identity, tie behavior, pairwise accounting, and the near-tie analysis helpers.

## Repository layout

```text
latent_score_decomposition.py   reusable decision-audit functions
tests/                          focused unit tests
experiments/                    Push-T analyzers, specifications, and results
frozen_study/                   preregistration, issuance, manifest, and sealed summary
figures/                        paper and exploratory figures
paper/                          public PWMS paper and LaTeX source
```

## Reproducibility boundary

The repository contains the frozen study documents, sealed aggregate results, deterministic analyzers, post-hoc specifications, and compact result files. The approximately 241 GB of per-candidate arrays are not duplicated in GitHub. Their paths and cryptographic bindings remain recorded in the sanitized merge index for reconstruction from the retained study artifacts.

The component and near-tie analyses are explicitly post-hoc. They describe stored decision artifacts; they do not identify the encoder, objective, predictor, or controller as a causal source of failure.

## Citation

See [`CITATION.cff`](CITATION.cff). The workshop is non-archival; this repository provides the stable public paper and project record requested by PWMS.
