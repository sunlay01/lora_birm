# CIFAR Entrypoints

This folder contains multiple generations of Spurious CIFAR / CifarMnist scripts. Not all of them should be treated as active entrypoints.

## Active Entrypoints

Use these scripts first.

### 1. `official_cifarmnist_comparison.py`

Purpose:
- shared engine for ERM, IRMv1, BIRM, and LoRA-BIRM variants
- baseline comparison runner
- common implementation source for several other wrappers

Use it when:
- you need the old full baseline table
- you need a negative-control rerun of the old `IRMv1 -> LoRA-BIRM` route
- you need the core implementation that other scripts import

Do not treat it as:
- the current main thesis route for CIFAR

### 2. `run_hybrid_env_official_schedule.py`

Purpose:
- current main screening script for the CIFAR thesis line
- compares a small set of hybrid-objective variants under a short-budget snapshot-style protocol

Use it when:
- you want to know whether the current "new objective + low-rank snapshot" route is alive
- you want a first-pass winner among hybrid variants

Primary metrics:
- `best_mean`
- `best_min`
- `final_mean`

Current interpretation:
- this is the main CIFAR screening entrypoint
- if this script does not produce a convincing winner, do not jump straight to 5-seed expansion

### 3. `validate_notebook_peak_multiseed.py`

Purpose:
- multiseed validation for the notebook-style peak / snapshot behavior
- checks whether the local-peak effect is real beyond a single seed

Use it when:
- you want to support the thesis claim about repeated local historical peaks
- you want a peak-diagnostic script rather than a full stable-final model script

Do not treat it as:
- a proof of stable dominance over BIRM

### 4. `run_hybrid_worst_env_schedule.py`

Purpose:
- targeted stability check for worst-environment weighting
- asks whether the weakest seed / weakest environment can be lifted

Use it when:
- `run_hybrid_env_official_schedule.py` shows peak potential but poor `best_min`
- you want to test whether worst-environment weighting improves stability

Primary metrics:
- `best_min`
- `best_mean`
- `final_mean`

Current interpretation:
- this is a stability stress test, not the first script to run

### 5. `run_hybrid_env_weight_grid.py`

Purpose:
- attribution / ablation script for variance-weight and gradient-penalty weight choices

Use it when:
- you need a diagnostic figure or table about objective weights
- you want to understand why a hybrid variant peaks or fails

Do not treat it as:
- the default main rerun script

## Recommended Cluster Order

If the goal is to support the current thesis wording, use this order:

1. `python cifar/run_hybrid_env_official_schedule.py`
2. `python cifar/validate_notebook_peak_multiseed.py`
3. `python cifar/run_hybrid_worst_env_schedule.py`
4. `python cifar/run_hybrid_env_weight_grid.py`

Only run the old route when you explicitly need a negative result or baseline refresh:

5. `python cifar/official_cifarmnist_comparison.py`

## What Each Script Is Supposed To Support

### Thesis-safe support

- `run_hybrid_env_official_schedule.py`
  - supports: hybrid route has peak potential under short budget
- `validate_notebook_peak_multiseed.py`
  - supports: repeated local historical peaks above BIRM can appear in screening-style runs
- `run_hybrid_worst_env_schedule.py`
  - supports or refutes: whether worst-environment weighting really improves stability
- `run_hybrid_env_weight_grid.py`
  - supports: loss-component attribution, not headline claims

### Negative-control support

- `official_cifarmnist_comparison.py`
  - supports: the older two-stage route is unstable and should not be over-claimed

## Legacy / One-Off Scripts

These are not the preferred first entrypoints anymore:

- `grid_search_notebook_style_lora_birm.py`
- `official_cifarmnist_lora_v2.py`
- `run_hybrid_env_multiseed.py`
- `run_hybrid_env_penalty_smoke.py`
- `run_notebook_style_hybrid_stabilize.py`
- `run_notebook_style_lora_birm.py`
- `run_notebook_style_peak_earlystop.py`
- `run_notebook_style_peak_longdiagnostic.py`
- `run_peak_then_stabilize_hybrid.py`
- `validate_notebook_peak_20seeds.py`

They can still be useful for archaeology or narrow diagnostics, but cluster time should not start there.

## Data And Outputs

Expected local data paths:

- `/root/data/cifar-10-batches-py`
- `/root/data/cifar-10-python.tar.gz`
- `/root/data/MNIST`

Important outputs to keep from every run:

- run summary CSV
- step history CSV
- grouped or final-table CSV
- best checkpoint per seed if saved

Minimum statistics to report back:

- mean best OOD accuracy
- std of best OOD accuracy
- mean final OOD accuracy
- worst-group or `test_minacc` style metric
- best-step distribution
- runtime per seed
