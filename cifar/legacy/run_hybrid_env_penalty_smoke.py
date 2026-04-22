from official_cifarmnist_comparison import ExperimentConfig, build_final_table, run_all


def main():
    config = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=120,
        print_every=5,
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
        seeds=(11,),
        methods=("LoRA-BIRM Hybrid-Env",),
        lora_rank=16,
        lora_alpha=16.0,
        lora_mc_samples=3,
        lora_kl_weight=0.0003,
        hybrid_variance_weight=1.0,
        hybrid_grad_penalty_weight=1.0,
        route_b_warmup_steps=400,
        output_prefix="official_cifarmnist_hybrid_env_smoke",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )
    summary_df, history_df = run_all(config)
    final_df = build_final_table(summary_df, config.methods)
    summary_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_hybrid_env_smoke_run_summary.csv"
    history_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_hybrid_env_smoke_step_history.csv"
    final_path = "/root/official_cifarmnist_artifacts/official_cifarmnist_hybrid_env_smoke_final_table.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    final_df.to_csv(final_path, index=False)
    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print("saved:", final_path)
    print("\nFinal table:")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
