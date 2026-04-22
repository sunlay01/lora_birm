# LoRA-BIRM: CMNIST, Spurious CIFAR, and Qwen Experiments

This repository collects our working research code for:

- CMNIST causal/OOD experiments
- Spurious CIFAR / CifarMnist experiments based on BIRM and LoRA-BIRM variants
- a Qwen notebook prototype adapted to our LoRA-BIRM Snapshot idea

The codebase is organized as a research repository rather than a polished library. The goal is to preserve the experiment scripts and notebooks that produced the main observations in our thesis work.

## Repository Structure

```text
.
├── cmnist/
│   └── cmnist.ipynb
├── cifar/
│   ├── official_cifarmnist_comparison.py
│   ├── official_cifarmnist_lora_v2.py
│   ├── run_*.py
│   ├── validate_*.py
│   ├── grid_search_*.py
│   └── birm_official/
│       ├── model.py
│       ├── utils.py
│       └── dataset_scripts/
└── qwen/
    ├── Untitled.ipynb
    ├── LLM.py
    ├── README.md
    ├── run_qwen_erm_irmv1_birm_baselines.py
    ├── run_qwen_lora_birm_only.py
    ├── qwen_lora_birm_snapshot_run.py
    ├── qwen_lora_birm_stable_run.py
    └── results/
```

## Main Components

### 1. `cmnist/`

- `cmnist.ipynb`
  - notebook for CMNIST experiments
  - useful for validating the low-dimensional causal generalization mechanism before scaling to harder visual settings

### 2. `cifar/`

- `official_cifarmnist_comparison.py`
  - main comparison script for ERM, IRMv1, BIRM, and our LoRA-style extensions on Spurious CIFAR / CifarMnist
- `official_cifarmnist_lora_v2.py`
  - LoRA-based BIRM variant on the CifarMnist setting
- `run_notebook_style_*.py`, `run_peak_then_stabilize_hybrid.py`, `run_hybrid_env_*.py`
  - ablations, stabilization runs, hybrid objectives, and snapshot-style search scripts
- `validate_notebook_peak_*.py`
  - multiseed validation utilities
- `birm_official/`
  - minimal vendored dependency subset from the original BIRM codebase required by the CIFAR scripts in this repository

### 3. `qwen/`

- `Untitled.ipynb`
  - Qwen2.5-3B notebook prototype
  - currently adapted to a LoRA-BIRM Snapshot style training loop with:
    - low-rank Bayesian head
    - environment variance regularization
    - EMA candidate snapshots
    - best-snapshot model recovery
- `LLM.py`
  - small helper wrapper around local Qwen loading
- `run_qwen_erm_irmv1_birm_baselines.py`
  - script used to compare `ERM`, `IRMv1`, and `BIRM` on the Qwen shortcut/reversed-test setup
- `run_qwen_lora_birm_only.py`
  - main standalone Qwen LoRA-BIRM training script
- `qwen_lora_birm_snapshot_run.py`, `qwen_lora_birm_stable_run.py`
  - snapshot-style and stabilized Qwen LoRA-BIRM variants
- `results/`
  - lightweight Qwen result summaries and step-history CSVs

## Environment

This repository was developed in a local research environment rather than a fresh package-managed project. A practical starting point is:

- Python 3.10+
- PyTorch with CUDA support
- `transformers`
- `peft`
- `pandas`
- `numpy`
- `scikit-learn`
- `torchvision`
- `modelscope`
- `langchain`
- `jupyter`

See `requirements.txt` for a lightweight dependency list.

## Data Assumptions

This repository does **not** include datasets, checkpoints, logs, or large experiment artifacts.

Several scripts assume local data already exists in paths such as:

- `/root/data/cifar-10-batches-py`
- `/root/data/cifar-10-python.tar.gz`
- `/root/data/MNIST`
- parquet files placed in the working directory for the Qwen notebook

You will likely need to adapt these paths for another machine.

## Running the Code

### CMNIST

Open and run:

```bash
jupyter notebook cmnist/cmnist.ipynb
```

### Spurious CIFAR / CifarMnist

Examples:

```bash
python cifar/official_cifarmnist_comparison.py
python cifar/official_cifarmnist_lora_v2.py
python cifar/run_hybrid_env_official_schedule.py
python cifar/validate_notebook_peak_multiseed.py
```

Notes:

- the CIFAR scripts now prefer the vendored `cifar/birm_official/` dependency tree inside this repository
- if an old local checkout of `Bayesian-Invariant-Risk-Minmization` exists under `/root`, the scripts can still fall back to it

### Qwen Notebook

Open and run:

```bash
jupyter notebook qwen/Untitled.ipynb
```

The active notebook cell is the LoRA-BIRM Snapshot version. Historical notebook cells were intentionally disabled to avoid running outdated code paths during `Run All`.

### Qwen Scripts

Examples:

```bash
python qwen/run_qwen_erm_irmv1_birm_baselines.py
python qwen/run_qwen_lora_birm_only.py
python qwen/qwen_lora_birm_stable_run.py
```

Notes:

- the Qwen scripts assume local parquet data files and a local/offline model environment
- only lightweight CSV/JSON summaries are tracked in git; checkpoints are excluded

## What Is Excluded

The repository intentionally excludes:

- training artifacts
- checkpoint files
- logs
- large CSV summaries
- parquet datasets
- cached pretrained model weights
- generated figures and PDFs

## Provenance Note

The `cifar/birm_official/` folder contains the minimal subset of the original BIRM implementation needed to run the Spurious CIFAR scripts in this repository. We keep it here only to make this research repo self-contained enough to execute the main CIFAR experiments.

## Cluster Handoff (2026-04-22)

This section is for handing the repo to another Codex instance running on a GPU cluster.

The thesis wording was already tightened and pushed to GitHub in commit `7ccb9c9` (`Refine thesis experimental claims`). The current thesis source is `docs/thesis/main.tex`, and the compiled PDF is `docs/thesis/main.pdf`.

### Current Thesis Claims To Support

- CMNIST is only used as a small-scale trend-validation chapter. The safe claim is that LoRA-BIRM shows an "accuracy not worse and cost not higher" trend relative to Full-BIRM on this toy setting. Do not write it as a definitive efficiency result.
- Spurious CIFAR is currently framed around three axes: OOD accuracy, cross-seed stability, and training-budget compression.
- The current safe Spurious CIFAR claim is: LoRA-BIRM snapshot / hybrid variants can approach BIRM's process-best region while substantially reducing single-seed runtime under the current snapshot-selection protocol.
- The current safe Spurious CIFAR claim is not a formal complexity or efficiency theorem. It is explicitly a "budget compression under the current diagnostic protocol" statement.
- The current thesis text still admits that stability across seeds and worst-group environments is not solved. This is the main place where reruns can materially improve the paper.
- Qwen is currently written as a single-run historical-peak comparison only.
- The current safe Qwen claim is: LoRA-BIRM reaches a higher historical peak at similar `time-to-best`, not that it is faster, not that it is statistically significant, and not that it has already passed a clean independent-validation selection protocol.

### Current Tracked Results

- Qwen frozen-head baselines in `qwen/results/qwen_erm_irmv1_birm_baseline_results.csv`:
- `BIRM = 50.15%`
- `ERM = 51.73%`
- `IRMv1 = 51.77%`
- Qwen stable LoRA-BIRM run in `qwen/results/qwen_lora_birm_stable_400_lr2e-5_pen3_ema98/summary.csv`:
- `best_acc = 53.93%`
- `best_step = 175`
- `max_steps = 400`
- `seed = 42`
- These numbers are already reflected conservatively in the thesis text.

### Priority Order On The Cluster

1. Rerun Spurious CIFAR first.
2. Rerun CMNIST only if we need a clean sanity-check table or a regenerated figure/table for consistency.
3. Rerun Qwen only after the CIFAR protocol is stable, because Qwen is the most expensive and the thesis already treats it as preliminary.

### What To Run For Spurious CIFAR

See `cifar/README.md` first. That file is the current source of truth for which CIFAR scripts are active entrypoints and which ones are legacy one-off wrappers.

Start from these scripts:

```bash
python cifar/run_hybrid_env_official_schedule.py
python cifar/validate_notebook_peak_multiseed.py
python cifar/run_hybrid_worst_env_schedule.py
```

Use `python cifar/official_cifarmnist_comparison.py` only when you explicitly need the old baseline / negative-control route.
The CIFAR code expects local data in paths such as:

- `/root/data/cifar-10-batches-py`
- `/root/data/cifar-10-python.tar.gz`
- `/root/data/MNIST`

Keep all generated CSV artifacts. The important ones are:

- run summary CSV
- step history CSV
- grouped / final table CSV
- best checkpoint path per seed if the script saves it

For every method or variant, extract at least:

- mean best OOD accuracy across seeds
- standard deviation across seeds
- mean final OOD accuracy across seeds
- mean `test_minacc` / worst-group-style metric across seeds
- best-step distribution across seeds
- runtime per seed

The key research question is:

- can LoRA-BIRM variants still approach BIRM's process-best OOD region after a clean rerun, while preserving the large runtime reduction claimed under the current snapshot-style protocol?

### Suggested Spurious CIFAR Workflow

1. Run a small screening pass first with the official schedule / snapshot-style script to see which LoRA-BIRM variant still looks alive.
2. If one variant is clearly better, rerun it with more seeds and keep the baselines fixed.
3. Compare not only the single best seed, but also the mean, variance, and worst-seed behavior.
4. Keep both process-best and final-step numbers. The thesis text currently relies on process-best diagnostics, so do not throw that information away.
5. If the rerun breaks the old claim, update the thesis claim downward rather than forcing the data to match the old narrative.

### What To Run For Qwen

Start from these scripts:

```bash
python qwen/run_qwen_erm_irmv1_birm_baselines.py
python qwen/qwen_lora_birm_stable_run.py
```

Important Qwen notes:

- the current comparison is end-to-end and does not use feature caching
- wall-clock is dominated by the large model forward/backward pass and the full reversed-test evaluation
- for this reason, the current thesis language should stay on "similar `time-to-best` with a higher historical peak" rather than "better efficiency"
- if rerunning, keep `summary.csv`, `step_history.csv`, and the exact config JSON for every run
- if resources allow, add multi-seed repeats; if resources do not allow that, at least reproduce the current single-run result cleanly

### Deliverables Expected From The Cluster

- a short note listing exactly which scripts were run
- the exact command lines and any environment changes needed to make them work
- the produced CSV paths
- a compact table comparing baselines and LoRA-BIRM variants
- a conclusion on whether the current thesis wording is still supportable, should be strengthened, or should be weakened
