# PWA-PushT v1.4-R Outcome-Blind Infrastructure Recovery Addendum

Status: recovery protocol declared before any v1.4 scientific outcome arrays or
per-row P/O/G/J values were inspected.

## Purpose

The frozen v1.4 run reached 600 terminal rows but became scientifically
unstable after one fatal CUDA illegal-memory-access poisoned its persistent
worker process. The supervisor then recorded the same infrastructure error for
every remaining untouched row. This addendum permits one separate recovery
attempt for only those mechanically identifiable cascade rows. It does not
alter, delete, or relabel any v1.4 evidence.

## Frozen source evidence

- Original manifest canonical content SHA-256:
  `10bf3d645541d5053e42c538f9c68bad0a526cbe324fd68d17a9a2201c2f25f2`
- Original manifest file SHA-256:
  `d4a8d3563abd61f5575635db6479e069b318317ee691fca7a162e40c23fe71d8`
- Original terminal ledger file SHA-256:
  `86c11bc693878e2e543ae5aeea59b69879ba3a539d56497da16d0ec5ecdb259a`
- Original terminal ledger tail event SHA-256:
  `fd97eed5836e3e04652170c26b5c64e67f1c9ee4480f702b19747874c71b8263`
- Original study issuance file SHA-256:
  `57a5294216275b60681546e725119bc22c83a9131add060a6ec8b72c116d15ce`
- Certificate bundle canonical content SHA-256:
  `7e4c9e76002c9534629360f0547918b2ca0f4eb67bcd83e78939b7d775a06a44`
- Certificate bundle file SHA-256:
  `4db6f36682d6aa1a03cd19c7ddaafd109349f99e7dd6504d15cea274e2957e9a`

At declaration, the original ledger contained 420 completed and 180 failed
rows. No scientific values were used to define this recovery.

## Eligibility rule

A row is eligible if and only if its frozen v1.4 terminal receipt:

1. has `status == "failed"`; and
2. has an `error` whose first line is exactly
   `CUDA error: an illegal memory access was encountered`.

Applied mechanically to the bound ledger, this selects exactly 169 unique
decision IDs: original ordinals 431 through 599 inclusive (84 JEPA-WM and 85
DINO-WM rows). The first selected row is
`v14_official_dino_wm_pusht_val_episode_015_offset_025_seed_03`; the last is
`v14_official_dino_wm_pusht_val_episode_020_offset_124_seed_03`.

No other failure class is eligible. The remaining 11 v1.4 failures stay
invalid. No additional or substitute row may be introduced if a recovery row
fails.

## Execution and terminality

- Each eligible row receives at most one recovery attempt.
- Every row runs in a newly created isolated OS process with a new CUDA
  context. The supervisor process never initializes CUDA.
- A recovery `started` receipt is terminal after interruption; resume may
  finalize it as failed but may not retry it.
- A failed child cannot cause later rows to inherit its CUDA context.
- Recovery artifacts retain the exact original decision identity, model,
  checkpoint, configuration, episode, offset, menu seed, certified candidate
  menu, dataset segments, and protocol bindings.
- Recovery uses a separate issuance, append-only hash-chained ledger, output
  directory, row capabilities, and process logs. Frozen v1.4 files remain
  byte-identical.

## Deterministic merge rule

For each of the 600 original manifest rows:

1. use the original artifact when the original terminal status is completed;
2. otherwise, use a completed, fully validated recovery artifact only when the
   original row satisfies the exact eligibility rule above;
3. otherwise mark the row invalid under the original zero/matched-mask policy.

Original and recovery failures remain visible in provenance. A recovery does
not erase or rewrite its original failed receipt. Duplicate recovery evidence,
changed bindings, missing lineage, selection drift, or a broken hash chain
makes the affected row invalid. There is no outcome-dependent choice between
original and recovered values.

If all 169 recovery rows pass, the combined evidence contains at most 589 valid
model rows; the preregistered 540/600 and trajectory-level validity gates still
apply unchanged. The final analysis remains outcome-blind until the recovery
ledger and merged evidence are sealed and validated.
