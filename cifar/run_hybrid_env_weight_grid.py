import itertools
from dataclasses import replace

import pandas as pd

from official_cifarmnist_comparison import ExperimentConfig, run_lora_method


def main():
    base = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=120,
        print_every=20,
        penalty_weight=3000.0,
        penalty_anneal_iters=0,
        l2_regularizer_weight=0.001,
        lr=0.01,
        step_gamma=1.0,
        opt="sgd",
        envs_num=2,
        image_scale=64,
        hidden_dim=16,
        grayscale_model=0,
        seeds=(11, 17, 23),
        methods=("LoRA-BIRM Hybrid-Env",),
        lora_rank=16,
        lora_alpha=16.0,
        lora_mc_samples=3,
        lora_kl_weight=0.0003,
        hybrid_variance_weight=1.0,
        hybrid_grad_penalty_weight=1.0,
        route_b_warmup_steps=400,
        output_prefix="official_cifarmnist_hybrid_env_weight_grid",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )

    variance_weights = (0.3, 1.0, 3.0)
    grad_weights = (0.3, 1.0, 3.0)

    summaries = []
    histories = []
    for variance_weight, grad_weight in itertools.product(variance_weights, grad_weights):
        config = replace(
            base,
            hybrid_variance_weight=variance_weight,
            hybrid_grad_penalty_weight=grad_weight,
        )
        tag = f"var={variance_weight:g} grad={grad_weight:g}"
        for seed in config.seeds:
            print(f"\n===== Hybrid-Env {tag} | seed={seed} =====", flush=True)
            summary, history = run_lora_method("LoRA-BIRM Hybrid-Env", config, seed)
            summary["tag"] = tag
            summary["hybrid_variance_weight_cfg"] = variance_weight
            summary["hybrid_grad_penalty_weight_cfg"] = grad_weight
            summaries.append(summary)

            history = history.copy()
            history["tag"] = tag
            history["hybrid_variance_weight_cfg"] = variance_weight
            history["hybrid_grad_penalty_weight_cfg"] = grad_weight
            histories.append(history)
            print(
                f"best={100.0 * summary['best_test_acc']:.2f}% "
                f"final={100.0 * summary['final_test_acc']:.2f}% "
                f"step={summary['best_step']}",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True)
    grouped = (
        summary_df.groupby(
            ["tag", "hybrid_variance_weight_cfg", "hybrid_grad_penalty_weight_cfg"],
            sort=False,
        )
        .agg(
            best_mean=("best_test_acc", "mean"),
            best_std=("best_test_acc", lambda x: x.std(ddof=0)),
            best_min=("best_test_acc", "min"),
            best_max=("best_test_acc", "max"),
            final_mean=("final_test_acc", "mean"),
            final_std=("final_test_acc", lambda x: x.std(ddof=0)),
            best_step_mean=("best_step", "mean"),
            num_runs=("seed", "count"),
        )
        .reset_index()
        .sort_values(["best_mean", "final_mean"], ascending=False)
    )

    summary_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_hybrid_env_weight_grid_run_summary.csv"
    history_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_hybrid_env_weight_grid_step_history.csv"
    grouped_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_hybrid_env_weight_grid_grouped.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    grouped.to_csv(grouped_path, index=False)

    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print("saved:", grouped_path)
    print("\nGrouped results:")
    print(
        grouped[
            [
                "tag",
                "best_mean",
                "best_std",
                "best_min",
                "best_max",
                "final_mean",
                "best_step_mean",
                "num_runs",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
