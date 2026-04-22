import copy
import time

import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler

from official_cifarmnist_comparison import (
    ARTIFACT_DIR,
    DEVICE,
    EBD,
    ExperimentConfig,
    OfficialResNetLoRA,
    build_dp,
    build_flags,
    build_optimizer,
    eval_acc_class,
    lora_hybrid_env_objective,
    mean_nll_class,
    set_seed,
    weight_norm_squared,
)
from utils import return_model


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


def run_one(base_config, variant_name, seed):
    set_seed(seed)
    flags = build_flags(base_config, seed, "birm")
    flags, _ = return_model(flags)
    dp = build_dp(base_config, seed)
    ebd = EBD(flags).to(DEVICE)
    model = OfficialResNetLoRA(
        rank=base_config.lora_rank,
        alpha=base_config.lora_alpha,
        bayesian=True,
    ).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), base_config)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(base_config.steps / 2)),
        gamma=base_config.step_gamma,
    )

    start = time.time()
    history = []
    best_acc = -1.0
    best_step = -1
    best_state = None

    for step in range(base_config.steps):
        model.train()
        train_x, train_y, train_g, _train_c = dp.fetch_train()
        features = model.features(train_x)

        if step < base_config.penalty_anneal_iters:
            logits = model.head(features, sample=False)
            train_nll = mean_nll_class(logits, train_y)
            variance_penalty = torch.tensor(0.0, device=DEVICE)
            grad_penalty = torch.tensor(0.0, device=DEVICE)
            train_penalty = torch.tensor(0.0, device=DEVICE)
            worst_env_loss = train_nll
            current_penalty_weight = 0.0
            stage = "erm_warmup"
        else:
            train_nll, train_penalty, variance_penalty, grad_penalty, worst_env_loss = lora_hybrid_env_objective(
                model, ebd, flags, features, train_y, train_g, base_config
            )
            current_penalty_weight = base_config.penalty_weight
            stage = "hybrid_penalty"

        kl_loss = model.kl_divergence()
        loss = train_nll.clone()
        loss = loss + flags.l2_regularizer_weight * weight_norm_squared(model)
        loss = loss + base_config.lora_kl_weight * kl_loss
        loss = loss + current_penalty_weight * train_penalty
        if current_penalty_weight > 1.0:
            loss = loss / (1.0 + current_penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % base_config.print_every == 0 or step == base_config.steps - 1:
            test_acc, test_minacc, test_majacc = evaluate_model(model, dp)
            if test_acc > best_acc:
                best_acc = test_acc
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {
                "variant": variant_name,
                "seed": seed,
                "step": step,
                "objective": float(loss.detach().cpu().item()),
                "train_nll": float(train_nll.detach().cpu().item()),
                "penalty": float(train_penalty.detach().cpu().item()),
                "variance_penalty": float(variance_penalty.detach().cpu().item()),
                "grad_penalty": float(grad_penalty.detach().cpu().item()),
                "worst_env_loss": float(worst_env_loss.detach().cpu().item()),
                "kl_loss": float(kl_loss.detach().cpu().item()),
                "penalty_weight": current_penalty_weight,
                "test_acc": test_acc,
                "test_minacc": test_minacc,
                "test_majacc": test_majacc,
                "best_test_acc_so_far": best_acc,
                "best_step_so_far": best_step,
                "elapsed_sec": time.time() - start,
                "stage": stage,
            }
            history.append(row)
            print(
                f"[{variant_name} seed={seed}] step={step:03d} stage={stage} "
                f"test={test_acc:.4f} min={test_minacc:.4f} maj={test_majacc:.4f} "
                f"nll={row['train_nll']:.4f} var={row['variance_penalty']:.6g} "
                f"grad={row['grad_penalty']:.6g} best={best_acc:.4f}@{best_step}",
                flush=True,
            )

    ckpt_path = ARTIFACT_DIR / f"official_cifarmnist_hybrid_official_schedule_{variant_name}_seed_{seed}_best.pt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
    summary = {
        "variant": variant_name,
        "seed": seed,
        "lr": base_config.lr,
        "penalty_weight": base_config.penalty_weight,
        "penalty_anneal_iters": base_config.penalty_anneal_iters,
        "hybrid_variance_weight": base_config.hybrid_variance_weight,
        "hybrid_grad_penalty_weight": base_config.hybrid_grad_penalty_weight,
        "best_test_acc": best_acc,
        "best_step": best_step,
        "final_test_acc": history[-1]["test_acc"],
        "runtime_sec": time.time() - start,
        "checkpoint_path": str(ckpt_path),
    }
    return summary, pd.DataFrame(history)


def main():
    template = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=180,
        print_every=10,
        penalty_weight=3000.0,
        penalty_anneal_iters=40,
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
        hybrid_variance_weight=0.3,
        hybrid_grad_penalty_weight=0.3,
        route_b_warmup_steps=400,
        output_prefix="official_cifarmnist_hybrid_official_schedule",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )

    variants = {
        "v03_g03_lr001_p3000_w40": {
            "lr": 0.01,
            "penalty_weight": 3000.0,
            "penalty_anneal_iters": 40,
            "hybrid_variance_weight": 0.3,
            "hybrid_grad_penalty_weight": 0.3,
        },
        "v1_g1_lr001_p3000_w40": {
            "lr": 0.01,
            "penalty_weight": 3000.0,
            "penalty_anneal_iters": 40,
            "hybrid_variance_weight": 1.0,
            "hybrid_grad_penalty_weight": 1.0,
        },
        "v03_g03_lr003_p5000_w20": {
            "lr": 0.03,
            "penalty_weight": 5000.0,
            "penalty_anneal_iters": 20,
            "hybrid_variance_weight": 0.3,
            "hybrid_grad_penalty_weight": 0.3,
        },
    }

    print("device =", DEVICE)
    summaries = []
    histories = []
    for variant_name, overrides in variants.items():
        config = copy.copy(template)
        for key, value in overrides.items():
            setattr(config, key, value)
        for seed in config.seeds:
            print(f"\n===== {variant_name} | seed={seed} =====", flush=True)
            summary, history = run_one(config, variant_name, seed)
            summaries.append(summary)
            histories.append(history)
            print(
                f"summary {variant_name} seed={seed}: "
                f"best={100.0 * summary['best_test_acc']:.2f}% "
                f"final={100.0 * summary['final_test_acc']:.2f}% "
                f"step={summary['best_step']}",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True)
    grouped = (
        summary_df.groupby("variant", sort=False)
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

    summary_path = ARTIFACT_DIR / "official_cifarmnist_hybrid_official_schedule_run_summary.csv"
    history_path = ARTIFACT_DIR / "official_cifarmnist_hybrid_official_schedule_step_history.csv"
    grouped_path = ARTIFACT_DIR / "official_cifarmnist_hybrid_official_schedule_grouped.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    grouped.to_csv(grouped_path, index=False)

    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print("saved:", grouped_path)
    print("\nGrouped:")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
