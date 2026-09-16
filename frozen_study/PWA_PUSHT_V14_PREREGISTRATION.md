# Prediction–World–Action Audit on Push-T (`PWA-PushT-v1.4`)

Status: LOCKED 2026-08-30 / NOT YET LAUNCHED. This document is the
pre-registration for one confirmatory execution after exact-source review.
It is a clean, versioned successor to permanently failed v1.3. No v1.3
artifact, certificate, manifest, issuance record, smoke result, or numerical
outcome is evidence for v1.4, and none may be reused, combined, promoted, or
used to tune v1.4. v1.3 failed before producing usable measurements because
the released replay-encoding helper was called without its required model
argument. v1.4 changes only those calls and disjoint version/provenance paths;
the scientific question, schedule, endpoints, gates, and analysis are unchanged.

## Scientific question and scope

The fixed scientific question is whether substituting the simulator-realized
terminal latent at the scoring interface recovers physical decision quality
lost by the released JEPA-WM and DINO-WM predictors on this Push-T audit.
The object is an outcome-blind ranking audit, not the released final CEM
planner and not improved closed-loop control. Both released predictors use one
shared representation geometry; this is one contact-rich task and does not
establish representation-level or robotics generality.

Each cell uses exactly the first, iteration-0 Gaussian CEM population: 300
proposals, zero mean, unit variance, proposal 0 equal to the mean, horizon 6,
model width 10, frameskip 5, and one population only. There are no elite
updates, later populations, returned optimizer means, outcome-informed menu
changes, NaN repair, or outcome-dependent support tuning.

## Exact frozen schedule

The canonical manifest is regenerated from the released validation lengths
and contains all 20 validation trajectories except trajectory 7, five released
length offsets

```text
offset_k = floor(k * (L - 31) / 4),  k = 0, 1, 2, 3, 4
```

menu seeds 1, 2, and 3, and both models (`official_jepa_wm` and
`official_dino_wm`). This is exactly

```text
20 trajectories × 5 offsets × 3 menu seeds × 2 models = 600 rows.
```

The inferential unit is the trajectory. It is not a candidate, row, offset,
menu, or duplicated model outcome. Seeds vary only the Gaussian proposal
menu. Simulator preparation, expert-goal replay, and candidate replay use
environment seed 1. Initial state, goal state, initial observation, goal
observation, physical actions, terminal states, reward traces, and all
immutable bindings are identical across the two models for a matched
trajectory/offset/seed cell. A cherry-picked or incomplete manifest is not a
v1.4 analysis input.

## Definitions and exact decomposition

For candidate `i`, `P_i` is the released planning cost evaluated on its
predicted terminal visual/proprioceptive latent. `O_i` replaces only that
prediction with the simulator-realized terminal observation encoded by the
same model. `O` is a realized-latent counterfactual, never an oracle.

`G_i` is negative official final-state distance to the fixed goal. `J_i` is
the float32 sum of all 30 official coverage rewards. The planner cost uses
visual MSE plus `alpha = 0.1` times proprioceptive MSE. For each latent
component `c`, with predicted latent `z_hat`, realized latent `z`, and goal
latent `g`, the locked identity is

```text
P_c - O_c = D_c + M_c
D_c = 2 mean((z - g) * (z_hat - z))
M_c = mean((z_hat - z)^2)
P - O = D_visual + M_visual + 0.1 * (D_proprio + M_proprio)
```

All retained arrays must be finite and every decomposition identity residual
must be at most `1e-8`. Captured iteration-0 scores must reproduce `P` within
`2e-5` with `rtol=0`. P/O tied candidate-ID sets use `atol=1e-10` and
`rtol=1e-8`; a P or O selector with more than one exact tied ID abstains.
G/J finite-menu-best values remain defined when G/J themselves tie. For the
secondary mechanism attribution only, a multi-candidate G- or J-best tie set
causes that pair to abstain; first-index ordering is never used to assign a
directional or magnitude mechanism.

For physical outcome `Y` in `{G,J}` and score `q` in `{P,O}`:

```text
R_q^Y = max_i Y_i - Y_argmin(q_i)
Delta^Y = R_P^Y - R_O^Y
```

Positive `Delta` means the realized-latent substitution recovers physical
decision regret on the identical candidate menu. Residual `R_O` is what the
substitution does not repair and may include representation, objective, and
state–action abstraction limitations.

## Authoritative production pass and encoding diagnostics

The live production-pass arrays are authoritative for all P/O/G/J results:
physical actions, terminal observations/states, reward traces, selectors,
bindings, and score vectors are never replaced, averaged, or selected from a
re-encoding. Independently re-encode the serialized initial observation, every
production terminal observation, and the serialized goal observation with
chunk sizes 8 and 4; each initial/terminal/goal re-encode is retained and
checked against the production pass. Both diagnostics anchor directly to the
production pass. Coordinate drift is descriptive only and reports max, RMS,
median, and p95 absolute drift for visual/proprio terminal and goal latents;
it has no latent-coordinate threshold.

Total score and visual/proprio component score drift retain the decision-level
`2e-5` tolerance with `rtol=0`; P/O tied-ID sets must agree exactly. Chunk-8
is the production comparison anchor and chunk-4 is the cross-check. A failed
re-encoding gate invalidates the row/evidence; coordinate magnitude alone does
not. Production values remain the values used for all primary estimates.

## Evidence seal and terminal lifecycle

Before any scientific row, a fresh v1.4 supervisor launches exactly 18 fresh
processes: three canonical excluded-trajectory contracts (menu seeds 1–3),
three repeats per contract, for each model. Repeat 0 is the comparison anchor.
Across repeats, total and visual/proprio component score vectors must have
elementwise max-minus-min at most `2e-5`; normalized/physical actions, raw
observations, terminal states, reward traces, J/G, immutable bindings, P/O
selectors, and exact tie sets must be identical. Repeat decision stability is
a gate, not a descriptive suggestion. Latent coordinate drift remains
descriptive.

A canonical v1.4 certificate issuance tombstone and a separate canonical
output-independent STUDY issuance record are each consumed exactly once before
the first scientific child process. Neither record contains or depends on
analysis output, and the STUDY record is consumed before any alternate full
run can be selected. A missing, failed, interrupted, or already-consumed
issuance permanently blocks the run. There is one full run only: no alternate
full run, retry, relabeling, v1.4 relaxation, or post hoc issuance is allowed.
Both certificate and study output directories must be created, verified as
directories, verified empty, and shown disjoint from every canonical issuance
and ledger path before either tombstone is consumed.

The authoritative certificate-attempt ledger is the canonical issuance-bound
file `experiments/official_pusht_v14_certificate_ledger.jsonl`, outside the
replaceable certificate output directory. It contains exactly 18 canonical
`started` histories, one for each model/contract/repeat child, and each child
has a unique external capability binding the run ID, ordinal, model,
contract/repeat, exact artifact/NPZ/raw-log paths, and started-event hash.
Restoring an older certificate-output snapshot therefore cannot restore an
earlier valid ledger prefix, select scratch evidence, or authorize another
attempt. The certificate worker must pass the shared isolation/origin
preflight, then atomically consume its capability before importing model/task
code. A crash after `started` or capability consumption is terminal for that
attempt; terminal success/failure receipts commit artifact, NPZ, and raw-log
hashes.

The authoritative study ledger is the canonical issuance-bound file
`experiments/official_pusht_v14_study_ledger.jsonl`, outside the replaceable
study output directory. Restoring an older output-directory snapshot therefore
cannot restore an earlier valid ledger prefix or authorize another row. Each
scientific worker must validate exactly one live `started` event and atomically
consume that row's unique capability before importing model/task code. A crash
after `started` or capability consumption is terminal for that row.

The certificate supervisor, full study, and every certificate child must run
through the exact source-path launcher
`experiments/official_pusht_v14_launcher.py` under isolated no-bytecode Python
(`-I -B`, user site disabled, no ambient `PYTHONPATH`). NumPy, PyTorch, YAML,
OmegaConf, and released `app`/`evals` import origins are checked before
irreversible scientific work. The launcher first scans the complete frozen
source surface and writes a one-shot process-bound attestation outside the
replaceable source/output roots; supervisor, study, and children must consume
that attestation before issuance or capability consumption. All v1.4 paths
and source bindings are anchored to the canonical project root
`REPOSITORY_ROOT`, so copied trees cannot issue independently. The released checkout and pinned DINOv2 tree
must contain no untracked or ignored importable source, bytecode cache, native
extension, or symlink; experiment children run with bytecode generation
disabled.

The certificate bundle, both issuance records, canonical manifest,
append-only ledger header and hash chain, completion/failure receipts, every
artifact and sidecar, and paired cross-model physical evidence must all
validate before `sealed=True`. Every model/row expected binding is mandatory:
released official commit and root, checkpoint/config, dataset root/split and
archive/`seq_lengths`/`states` hashes, menu seed, `environment_seed=1`, state
and action segment hashes, independent initial/goal state hashes, independent
initial/goal observation hashes, runtime and hardware fingerprints plus their
hashes, and the pinned DINOv2 source/weights binding. The certificate and
STUDY issuance references, file hashes, and immutable commitments are passed
to artifact validation. A completed or failed receipt must commit and
revalidate file existence states and hashes plus manifest, global bundle,
model certificate, certificate issuance, STUDY issuance, protocol, and source
bundle commitments; a failed receipt is not exempt from these checks.
The 600-row run requires a passed v1.4 bundle before its first row. A valid
completed row has one `started` then one `completed` event, and its receipt
binds artifact/NPZ hashes, manifest, model certificate, global bundle,
protocol, and source bundle. A valid failed row has one `started` then one
terminal `failed` receipt with immutable file-state evidence. Missing,
malformed, replaced, selectively missing, or tampered evidence invalidates the
sealed study; it is not silently sanitized into a scientific zero.

## Primary estimands, invalid-data policy, and validity gates

Raw `Delta-G` and `Delta-J` are co-primary. Each trajectory has exactly 15
offset/menu cells. A cell is matched only when both model rows are valid. A
missing, terminal-failed, non-finite, or tied P/O row makes that cell invalid
for both models. Invalid or unmatched cells contribute zero to both models
under the identical matched 15-cell mask; there is no complete-case deletion.
Here zero is a locked invalid-as-zero convention: it means that the cell
contributes no observed recovery evidence for either model, while preserving
the predeclared 15-cell denominator and equal model exposure. It is not a
claim that the unobserved outcome was physically zero, and it is not replaced
by a model-specific imputation or a post hoc missing-at-random assumption.
Within a trajectory, values are first averaged over all 15 fixed cells for
each model, then the two model trajectory values are pooled with equal weight.
Model-specific estimates use the same matched-cell mask and their own
per-model trajectory values.

The raw analysis is valid only when at least 540 of 600 model rows are valid
under that common mask and at least 18 of 20 trajectories have at least 12 of
15 matched cells. These gates are binding and cannot be waived after seeing
results.

For each co-primary outcome, report the 20 trajectory values, mean, median,
deterministic 10,000-resample percentile trajectory-bootstrap 95% CI, and
exact two-sided trajectory sign-flip p-value. The exact sign-flip test treats
the trajectory-cluster values as exchangeable under the null and assumes
sign symmetry of their null distribution; it enumerates all sign assignments
for the observed trajectory count and does not use candidate-level signs.
Holm correction is applied to the Delta-G/Delta-J family at FWER alpha .05.
Report the same inferential
quantities and CIs separately for each model.

## Mandatory magnitude-bounded subset S

The outcome-blind subset is recomputed independently inside every row:

```text
S = { i : max(abs(normalized_action_i)) <= 3 }
```

S is a magnitude-bounded robustness subset, not an in-distribution or learned
support claim. S requires at least 200 candidates and untied P/O choices. All
winners, regrets, Delta values, exact uniform expectations, ranking metrics,
and normalizations are recomputed inside S from the NPZ arrays. No recorded
artifact decision delta or winner is used. S uses the same paired-mask,
invalid-as-zero, 15-cell trajectory pooling, bootstrap, sign-flip, Holm, and
540/600 plus 18/20 validity rules as raw analysis.

## Claim gates and usefulness

A broad predictive-ranking-recovery claim is permitted only for a passed
`sealed=True` analysis with a passed bundle/repeat gate, and only if all of
the following hold:

1. raw and S validity gates pass;
2. raw and S Holm-adjusted p-values are below .05 for both G and J;
3. raw and S pooled means and lower 95% CI bounds are above zero for both G
   and J;
4. lower model-specific 95% CI bounds are above zero for both models, both
   outcomes, and both raw and S inventories;
5. raw and S pooled signs do not conflict for either outcome; and
6. the score/component/tie-set/repeat decision-stability gate passes.

Failure of any condition blocks the broad claim. Isolated positive findings
remain descriptive and cannot be promoted.

Uniform selection is the exact mean G/J outcome across every candidate in the
current menu, not one lucky sampled candidate. P-minus-uniform is analyzed
with the same matched trajectory procedure and Holm correction. The word
`useful` is permitted only for a passed sealed bundle and repeat gate, and only
when P beats exact uniform for both G and J with a passing validity gate,
adjusted p-values below .05, and positive lower 95% CI bounds. With
`sealed=False`, the output is diagnostic only: broad and usefulness gates are
always false and every raw/S result label is `unstable`/non-claimable.
Failure blocks usefulness language but does not by itself negate a narrowly
reported ranking pathology or recoverability result; however, any failed
integrity, repeat, or validity gate makes every result sublabel and the
overall result `unstable`.

## Standard descriptive metrics and mechanism sensitivity

Standard metrics are descriptive only: score-error MSE; separate visual and
proprioceptive latent-prediction MSE and their 0.1-weighted total; absolute
`|P-O|`; Spearman correlation; pairwise ranking accuracy; normalized
Regret@1; and top-1/5/10 normalized regret. Controls include exact uniform
expectation, P, O, finite-menu G-oracle, finite-menu J-oracle, the expert
goal-defining segment, and hold-position no-op. These metrics cannot change a
primary claim gate.

For each row and each of the three predeclared consequential pairs—P-selected versus O-selected,
P-selected versus finite-menu-best G, and P-selected versus finite-menu-best
J—report predicted margin, realized-latent margin, directional margin,
magnitude margin, identity residual, selection disagreement/reversal, and
directional-versus-magnitude dominance. Recompute the mechanism diagnostics
for production, chunk-8, and chunk-4 encodings. For each row, compare the
directional/magnitude signs and dominance across all three encodings. Emit a
row label (`directional`, `magnitude`, or `balanced`) only when both sign and
dominance are stable; otherwise the row label is exactly `indeterminate`.
Retain the actual per-encoding margin summaries and report aggregate
row-label counts/fractions; do not replace heterogeneous rows with one global
signature. Mechanism results are secondary descriptive analyses and cannot
change the primary claim gate.

## Exact result interpretation table

The formal interpretation uses the following predeclared table for each
co-primary outcome and inventory (raw and S), after validity and claim gates
are shown:

| Result | Required pattern | Permitted wording |
|---|---|---|
| Positive | mean > 0 and lower 95% CI > 0, with required adjusted p/gates | realized-latent substitution recovered the corresponding physical regret on this audit |
| Null | 95% CI includes 0 and no opposite-sign CI conclusion | no reliable recovery was detected |
| Negative | mean < 0 and upper 95% CI < 0, with required adjusted p/gates | substitution worsened the corresponding physical regret on this audit |
| Mixed | raw and S signs conflict, or G and J disagree | outcome-dependent or robustness-sensitive result; no broad claim |
| Unstable | validity/repeatability/integrity gate fails, evidence is insufficient, or CI/p-value is not interpretable | result is unstable/inconclusive; no directional claim |

The table determines result wording, not the pre-registered question or
estimands. Observed results may change the paper's title, emphasis, or
headline wording to accurately reflect the table, but may not change the
formal scientific question, endpoints, decomposition, validity gates,
inferential tests, or claim gate.

## Historical disclosure and claim boundary

The paper must disclose that v1.2 permanently failed its predeclared
same-path RGB/serialization integrity rule and that v1.4 had not launched at
the time of this draft. No v1.2 evidence or numerical outcome may be reused.
The v1.4 study compares two predictors within one shared representation on
one contact-rich Push-T audit. It does not establish representation-level
generality, improved closed-loop control, or universal robotics behavior.
