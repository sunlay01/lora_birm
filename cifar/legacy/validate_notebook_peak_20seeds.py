import time
from dataclasses import asdict

import pandas as pd
import torch

from official_cifarmnist_comparison import (
    ARTIFACT_DIR,
    DEVICE,
    ExperimentConfig,
    OfficialResNetLoRA,
    build_dp,
    build_flags,
    build_optimizer,
    eval_acc_class,
    lora_notebook_variance_objective,
    set_seed,
    weight_norm_squared,
)


def evaluate_model(model, dp):
    model.eval()
    test_x, test_y, test_g, test_c = dp.fetch_test()
    with torch.no_grad():
        test_logits = model(test_x, sample=False)
    test_acc, test_minacc, test_majacc = eval_acc_class(test_logits, test_y, test_c)
    return (
        float(test_acc.detach().cpu().item()),
        float(test_minacc.detach().cpu().item()),
        float(test_majacc.detach().cpu().item()),
    )


def train_step(model, optimizer, flags, config, dp):
    model.train()
    train_x, train_y, train_g, _train_c = dp.fetch_train()
    features = model.features(train_x)
    train_nll, train_penalty = lora_notebook_variance_objective(
        model, features, train_y, train_g, config
    )
    kl_loss = model.kl_divergence()

    loss = train_nll.clone()
    loss = loss + flags.l2_regularizer_weight * weight_norm_squared(model)
    loss = loss + config.lora_kl_weight * kl_loss
    loss = loss + config.penalty_weight * train_penalty
    if config.penalty_weight > 1.0:
        loss = loss / config.penalty_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item())


def main():
    seeds = tuple(range(1, 41, 2))
    config = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=8,
        print_every=1,
        penalty_weight=5000.0,
        penalty_anneal_iters=0,
        l2_regularizer_weight=0.001,
        lr=0.03,
        step_gamma=1.0,
        opt="sgd",
        envs_num=2,
        image_scale=64,
        hidden_dim=16,
        grayscale_model=0,
        seeds=seeds,
        methods=("LoRA-BIRM Notebook-Variance",),
        lora_rank=16,
        lora_alpha=16.0,
        lora_mc_samples=3,
        lora_kl_weight=0.0003,
        route_b_warmup_steps=400,
        output_prefix="official_cifarmnist_notebook_peak_20seeds",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )

    print("device =", DEVICE)
    print(asdict(config))

    summaries = []
    histories = []

    for seed in config.seeds:
        print(f"\n===== peak validation seed={seed} =====", flush=True)
        set_seed(seed)
        flags = build_flags(config, seed, "birm")
        dp = build_dp(config, seed)
        model = OfficialResNetLoRA(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            bayesian=True,
        ).to(DEVICE)
        optimizer = build_optimizer(model.parameters(), config)

        best_acc = -1.0
        best_step = -1
        best_minacc = 0.0
        best_majacc = 0.0
        start = time.time()

        for step in range(config.steps):
            loss = train_step(model, optimizer, flags, config, dp)
            test_acc, test_minacc, test_majacc = evaluate_model(model, dp)
            if test_acc > best_acc:
                best_acc = test_acc
                best_step = step
                best_minacc = test_minacc
                best_majacc = test_majacc
            histories.append(
                {
                    "seed": seed,
                    "step": step,
                    "objective": loss,
                    "test_acc": test_acc,
                    "test_minacc": test_minacc,
                    "test_majacc": test_majacc,
                    "best_test_acc_so_far": best_acc,
                    "best_step_so_far": best_step,
                    "elapsed_sec": time.time() - start,
                }
            )
        summaries.append(
            {
                "seed": seed,
                "best_test_acc": best_acc,
                "best_step": best_step,
                "best_test_minacc": best_minacc,
                "best_test_majacc": best_majacc,
                "final_test_acc": histories[-1]["test_acc"],
                "runtime_sec": time.time() - start,
            }
        )
        print(
            f"seed={seed} summary: best={100.0 * best_acc:.2f}% at step={best_step}, "
            f"final={100.0 * histories[-1]['test_acc']:.2f}%",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries).sort_values("best_test_acc", ascending=False)
    history_df = pd.DataFrame(histories)
    summary_path = ARTIFACT_DIR / f"{config.output_prefix}_run_summary.csv"
    history_path = ARTIFACT_DIR / f"{config.output_prefix}_step_history.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)

    thresholds = [0.55, 0.60, 0.65, 0.70]
    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print(
        f"\nPeak distribution: mean={100.0 * summary_df['best_test_acc'].mean():.2f}% "
        f"std={100.0 * summary_df['best_test_acc'].std(ddof=0):.2f}% "
        f"min={100.0 * summary_df['best_test_acc'].min():.2f}% "
        f"max={100.0 * summary_df['best_test_acc'].max():.2f}%"
    )
    for thr in thresholds:
        rate = (summary_df["best_test_acc"] >= thr).mean()
        print(f"peak >= {100.0 * thr:.0f}% : {100.0 * rate:.1f}% ({int((summary_df['best_test_acc'] >= thr).sum())}/{len(summary_df)})")
    print("\nPer-seed summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
