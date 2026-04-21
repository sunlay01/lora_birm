# Qwen Experiments

This directory collects the Qwen-side research code used for our LoRA-BIRM experiments on the shortcut/reversed-test setup.

## Main Files

- `LLM.py`
  - local helper wrapper for loading Qwen models
- `Untitled.ipynb`
  - original notebook prototype
  - currently has local uncommitted edits in the working tree and is intentionally left untouched by the upload workflow
- `run_qwen_erm_irmv1_birm_baselines.py`
  - baseline comparison script for `ERM`, `IRMv1`, and `BIRM`
- `run_qwen_lora_birm_only.py`
  - main script for the Qwen LoRA-BIRM run used in the thesis writeup
- `qwen_lora_birm_snapshot_run.py`
  - earlier snapshot-style Qwen experiment script
- `qwen_lora_birm_stable_run.py`
  - stabilized Qwen LoRA-BIRM snapshot script

## Results Included

The `results/` folder contains lightweight reproducibility artifacts only:

- baseline summary CSV
- stable LoRA-BIRM summary CSV
- stable LoRA-BIRM step history CSV
- stable LoRA-BIRM config JSON

Large checkpoints and adapter weights are intentionally excluded from the repository.

## Notes

- The scripts assume local access to the parquet files used in the Qwen toxicity shortcut experiment.
- Paths are still research-environment oriented and may need adjustment on another machine.
- The thesis chapter uses the stable LoRA-BIRM run where the best checkpoint occurs at step `175`.
