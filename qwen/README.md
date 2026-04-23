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
- stable LoRA-BIRM reference summaries / step histories / configs
- stable LoRA-BIRM `300`-step seed sweep summaries / step histories / configs
- an aggregate `300`-step seed-sweep summary CSV

Large checkpoints and adapter weights are intentionally excluded from the repository.

## Current Stable Reference

- The thesis chapter currently uses the stable LoRA-BIRM run in `results/qwen_lora_birm_stable_400_lr2e-5_pen3_ema98/`.
- That reference run is `seed=42`, `best_acc=0.5393`, `best_step=175`, `best_tag=current`, with runtime about `85.2 min`.
- We now also keep a full `400`-step cluster reference for `seed=43` in `results/qwen_lora_birm_stable_400_lr2e-5_pen3_ema98_seed43/`.
- The completed `300`-step seed sweep is archived under `results/qwen_lora_birm_stable_300_lr2e-5_pen3_ema98_seed43/` through `results/qwen_lora_birm_stable_300_lr2e-5_pen3_ema98_seed47/`, with a combined table in `results/qwen_lora_birm_stable_300_lr2e-5_pen3_ema98_seed_sweep_summary.csv`.
- The main question left open is seed sensitivity: how often the best snapshot appears early, and how large the late-stage drop is across seeds.

## Next Cluster Run Plan

- Keep `seed43` as the full `400`-step reference run.
- Continue the remaining sweep on seeds `44 45 46 47` with `300` steps.
- Do not change any hyperparameters besides `max_steps`. Keep the same reversed-shortcut dataset construction, same `eval_interval=25`, same LoRA/BIRM settings.
- Each seed must write to its own output directory so summaries and adapters do not overwrite each other.
- The stable script now supports these environment variables:
  - `QWEN_SEED`
  - `QWEN_MAX_STEPS`
  - `QWEN_RUN_NAME`
  - `QWEN_OUT_ROOT`
  - `QWEN_MODEL_CACHE_DIR`
  - `QWEN_DATA_PARQUET`

From inside `qwen/`, a single `300`-step continuation run should look like:

```bash
QWEN_SEED=44 \
QWEN_MAX_STEPS=300 \
QWEN_RUN_NAME=qwen_lora_birm_stable_300_lr2e-5_pen3_ema98_seed44 \
QWEN_OUT_ROOT=/root/qwen_lora_birm_tuning \
QWEN_MODEL_CACHE_DIR=/root/autodl-tmp/modelscope_cache \
QWEN_DATA_PARQUET=/root/train-00000-of-00001.parquet \
python qwen_lora_birm_stable_run.py | tee seed44_300.log
```

If the cluster gives multiple GPUs/jobs, launch one seed per job. If only one GPU is available, run them sequentially:

```bash
for seed in 44 45 46 47; do
  QWEN_SEED=$seed \
  QWEN_MAX_STEPS=300 \
  QWEN_RUN_NAME=qwen_lora_birm_stable_300_lr2e-5_pen3_ema98_seed${seed} \
  QWEN_OUT_ROOT=/root/qwen_lora_birm_tuning \
  QWEN_MODEL_CACHE_DIR=/root/autodl-tmp/modelscope_cache \
  QWEN_DATA_PARQUET=/root/train-00000-of-00001.parquet \
  python qwen_lora_birm_stable_run.py | tee seed${seed}_300.log
done
```

## What To Bring Back

- For each seed, save these lightweight artifacts:
  - `summary.csv`
  - `step_history.csv`
  - the stdout log, e.g. `seed43_300.log`
- The large adapter directory can stay on the cluster unless we later decide one seed is worth archiving.
- After the sweep finishes, merge all `summary.csv` files into one table for comparison. A simple collector is:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/root/qwen_lora_birm_tuning")
run_dirs = sorted(root.glob("qwen_lora_birm_stable_*_lr2e-5_pen3_ema98_seed*/summary.csv"))
rows = []
for path in run_dirs:
    df = pd.read_csv(path)
    if not df.empty:
        rows.append(df.iloc[0].to_dict())

out = pd.DataFrame(rows).sort_values(["seed", "max_steps"])
out.to_csv("qwen_lora_birm_seed_sweep_summary.csv", index=False)
print(out[["seed", "max_steps", "best_acc", "best_step", "best_tag", "runtime_min"]])
PY
```

## What We Check After Dinner

- Mean and std of `best_acc` across seeds.
- Whether `best_step` keeps clustering in the mid-training region rather than near the end.
- Whether the final selected snapshot in the last row of `step_history.csv` is usually worse than the run best, which supports the snapshot-selection narrative.
- Whether any seed collapses or behaves qualitatively differently enough to weaken the current thesis claim.
