import itertools
from dataclasses import replace

import pandas as pd

import _pathfix  # noqa: F401

from official_cifarmnist_comparison import CONFIG, ExperimentConfig, run_lora_method


def main():
    base = ExperimentConfig(
        **{
            **CONFIG.__dict__,
            "methods": ("LoRA-BIRM Notebook-Variance",),
            "seeds": (11,),
            "steps": 300,
            "print_every": 25,
            "penalty_anneal_iters": 0,
            "lora_mc_samples": 3,
            "output_prefix": "official_cifarmnist_notebook_grid",
        }
    )

    lrs = (0.003, 0.01, 0.03)
    penalties = (1000.0, 3000.0, 10000.0)
    kl_weights = (0.0003, 0.001)

    summaries = []
    histories = []
    for lr, penalty_weight, kl_weight in itertools.product(lrs, penalties, kl_weights):
        cfg = replace(
            base,
            lr=lr,
            penalty_weight=penalty_weight,
            lora_kl_weight=kl_weight,
        )
        tag = f"lr={lr:g} p={penalty_weight:g} kl={kl_weight:g}"
        print(f"\n===== Notebook-Variance {tag} =====", flush=True)
        summary, history = run_lora_method("LoRA-BIRM Notebook-Variance", cfg, seed=11)
        summary["lr"] = lr
        summary["penalty_weight_cfg"] = penalty_weight
        summary["lora_kl_weight_cfg"] = kl_weight
        summary["tag"] = tag
        summaries.append(summary)

        history = history.copy()
        history["lr"] = lr
        history["penalty_weight_cfg"] = penalty_weight
        history["lora_kl_weight_cfg"] = kl_weight
        history["tag"] = tag
        histories.append(history)
        print(
            f"best={100.0 * summary['best_test_acc']:.2f}% "
            f"step={summary['best_step']} "
            f"runtime={summary['runtime_sec']:.1f}s",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        ["best_test_acc", "final_test_acc"], ascending=False
    )
    history_df = pd.concat(histories, ignore_index=True)

    summary_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_notebook_grid_run_summary.csv"
    history_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_notebook_grid_step_history.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)

    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print("\nTop configs:")
    print(
        summary_df[
            [
                "tag",
                "best_test_acc",
                "best_step",
                "final_test_acc",
                "runtime_sec",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
