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
