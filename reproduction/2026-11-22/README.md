# Numerical reproduction release, November 22, 2026

**[Download the package](https://github.com/Penguinz3/latent-plan-audit/releases/download/v2026.11.22/latent-plan-audit-reproduction-2026-11-22.zip)** · [GitHub release](https://github.com/Penguinz3/latent-plan-audit/releases/tag/v2026.11.22)

The ZIP is **109,851,585 bytes** (about 110 MB, 105 MiB), with about 160 MB of uncompressed files. It is a release attachment, so cloning the source repository does not download the data. The approximately 241 GB of full raw candidate arrays are excluded.

SHA-256 of the ZIP:

```text
1e26b13030d411dbbbbc0d65ed4aa78dda69153cdbbf59563e95968951583d02
```

[release.json](release.json) records the asset size/hash. [package-manifest.json](package-manifest.json) lists file hashes and sizes, and the ZIP also contains `SHA256SUMS`. [source-and-paper-manifest.json](source-and-paper-manifest.json) records copied source and paper hashes.

## Run

Extract the ZIP and enter `latent-plan-audit-reproduction`. Use Python with NumPy; the tested runtime was Python 3.12.13, NumPy 2.5.1 on Windows. The package includes a pinned `requirements.txt` if creating a new environment.

In PowerShell:

```powershell
# Optional setup in a new environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:PYTHONPATH = ''
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -B experiments/test_task_native_selector.py
python -I -B experiments/analyze_task_native_selector.py analyze --root . --output-dir regenerated
python -I -B verify_reproduction.py
```

For a POSIX shell, use the same Python commands after `unset PYTHONPATH` and `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Activate a virtual environment using its platform-specific path.

The [verification record](verification.json) reused an existing interpreter and installed NumPy with isolated analyzer imports. It was not a new dependency installation or simulator rerun. Generated timestamps and absolute path metadata can differ between runs.

## Contents and verified scope

- Compact Push-T candidate costs, actions, terminal states, distances, and reward traces, with original result and export commitments.
- Compact Wall candidate arrays/tables and frozen analysis/provenance records.
- Frozen task-state selector specification, analyzer, invariant checks, and approved result tables.
- Original protocol documents, configurations, issuance records, and execution/recovery ledgers retained for inspection.
- A verifier requiring exact hashes for four numerical CSVs, physical-support JSON, and original-results JSON; it also compares every declared numerical section of the diagnostic result and checks that analysis inputs resolve within the package.

Small inspectable outputs are mirrored here: [summary.csv](summary.csv), [physical_support.json](physical_support.json), and [original_results.json](original_results.json). The post-hoc summary uses unadjusted descriptive intervals. The original-results file separately preserves the original confidence intervals, tests, and multiplicity correction.

## Limits

This package recomputes results from saved evidence. It performs no model inference, simulation, training, optimizer update, or closed-loop evaluation. Initial Push-T states and full raw observations are absent. Wall exported states are rounded float32; stored native distances are authoritative. Historical source paths in metadata are provenance strings, not required reproduction paths.

The hindsight state selector minimizes stored native terminal distance among the same candidates. Its distance advantage is definitional. It reduces Push-T cumulative reward and adds no observed Wall completions beyond realized-latent selection. It does not establish a deployable remedy or isolate a representation defect. Original populations, missingness rules, and trajectory/case clustering are preserved; no new p-values are added.

Historical protocol records retain their original local-review wording. Their inclusion in this public release establishes a packaging date, not independent proof of preregistration timing or outcome blindness. Full raw observations, latent tensors, checkpoints, GPU runtime, and the complete original project history still require a separate backup.
