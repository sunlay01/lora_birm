from official_cifarmnist_comparison import CONFIG, ExperimentConfig, build_final_table, run_all


def main():
    config = ExperimentConfig(
        **{
            **CONFIG.__dict__,
            "methods": ("LoRA-BIRM Notebook-Variance",),
            "seeds": (11,),
            "steps": 1000,
            "print_every": 50,
            "penalty_weight": 10000.0,
            "penalty_anneal_iters": 0,
            "lora_mc_samples": 3,
            "lora_rank": 16,
            "lora_alpha": 16.0,
            "lora_kl_weight": 0.001,
            "output_prefix": "official_cifarmnist_notebook_style",
        }
    )
    summary_df, history_df = run_all(config)
    final_df = build_final_table(summary_df, config.methods)
    summary_path = f"/root/official_cifarmnist_artifacts/{config.output_prefix}_run_summary.csv"
    history_path = f"/root/official_cifarmnist_artifacts/{config.output_prefix}_step_history.csv"
    final_path = f"/root/official_cifarmnist_artifacts/{config.output_prefix}_final_table.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    final_df.to_csv(final_path, index=False)
    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print("saved:", final_path)
    print("\nFinal table:")
    print(final_df.sort_values("best_ood_acc_mean_pct", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
