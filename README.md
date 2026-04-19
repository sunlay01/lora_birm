# CMNIST and Spurious CIFAR Experiments

This repository contains the working code used for our CMNIST and Spurious CIFAR experiments, plus the Qwen notebook prototype used to test the same causal/OOD training ideas on a language model.

## Structure

- `cmnist/`
  - `cmnist.ipynb`: CMNIST experiment notebook.
- `cifar/`
  - `official_cifarmnist_comparison.py`: main Spurious CIFAR / CifarMnist comparison script for ERM, IRMv1, and BIRM-style baselines.
  - `official_cifarmnist_lora_v2.py`: LoRA-based BIRM variant on the CifarMnist setup.
  - `run_*.py`, `validate_*.py`, `grid_search_*.py`: follow-up ablations, stabilization runs, and validation scripts for snapshot, hybrid, and multiseed experiments.
- `qwen/`
  - `Untitled.ipynb`: Qwen notebook adapted to our LoRA-BIRM Snapshot algorithm.
  - `LLM.py`: auxiliary LLM-related code.

## Notes

- This repository intentionally excludes datasets, cached model weights, checkpoints, logs, and large artifact directories.
- The codebase is a research workspace snapshot, so several scripts are experiment-oriented rather than packaged as a library.
- Some scripts assume datasets already exist locally in paths such as `/root/data`, `/root/data_dir`, or parquet files in the working directory.

## Excluded From Version Control

The following categories are intentionally not included:

- training artifacts and checkpoints
- `official_cifarmnist_artifacts/`
- dataset folders and archives
- Python cache files
- temporary notebook backups

