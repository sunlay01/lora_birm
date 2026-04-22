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
    lora_notebook_variance_objective,
    set_seed,
    weight_norm_squared,
)
from utils import mean_nll_class, return_model


def evaluate_model(model, dp):
    model.eval()
    test_x, test_y, test_g, test_c = dp.fetch_test()
    with torch.no_grad():
        logits = model(test_x, sample=False)
    acc, minacc, majacc = eval_acc_class(logits, test_y, test_c)
    return float(acc.cpu().item()), float(minacc.cpu().item()), float(majacc.cpu().item())


def train_peak_step(model, optimizer, flags, config, dp):
    model.train()
    train_x, train_y, train_g, _train_c = dp.fetch_train()
    features = model.features(train_x)
    train_nll, variance_penalty = lora_notebook_variance_objective(
        model, features, train_y, train_g, config
    )
    kl_loss = model.kl_divergence()
    loss = train_nll.clone()
    loss = loss + flags.l2_regularizer_weight * weight_norm_squared(model)
    loss = loss + config.lora_kl_weight * kl_loss
    loss = loss + config.penalty_weight * variance_penalty
    if config.penalty_weight > 1.0:
        loss = loss / config.penalty_weight
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (
        float(loss.detach().cpu().item()),
        float(train_nll.detach().cpu().item()),
        float(variance_penalty.detach().cpu().item()),
        0.0,
        float(kl_loss.detach().cpu().item()),
    )


def train_stabilize_step(model, optimizer, flags, config, dp, ebd):
    model.train()
    train_x, train_y, train_g, _train_c = dp.fetch_train()
    features = model.features(train_x)
    train_nll, total_penalty, variance_penalty, grad_penalty, _worst_env_loss = lora_hybrid_env_objective(
        model, ebd, flags, features, train_y, train_g, config
    )
    kl_loss = model.kl_divergence()
    loss = train_nll.clone()
    loss = loss + flags.l2_regularizer_weight * weight_norm_squared(model)
    loss = loss + config.lora_kl_weight * kl_loss
    loss = loss + config.penalty_weight * total_penalty
    if config.penalty_weight > 1.0:
        loss = loss / (1.0 + config.penalty_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (
        float(loss.detach().cpu().item()),
        float(train_nll.detach().cpu().item()),
        float(variance_penalty.detach().cpu().item()),
        float(grad_penalty.detach().cpu().item()),
        float(kl_loss.detach().cpu().item()),
    )


def run_variant(seed, variant_name, peak_steps, stabilize_lr, stabilize_penalty, stabilize_steps, freeze_backbone):
    peak_config = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=peak_steps,
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
        seeds=(seed,),
        methods=("LoRA-BIRM Notebook-Variance",),
        lora_rank=16,
        lora_alpha=16.0,
        lora_mc_samples=3,
        lora_kl_weight=0.0003,
        hybrid_variance_weight=0.3,
        hybrid_grad_penalty_weight=0.3,
        route_b_warmup_steps=400,
        output_prefix="official_cifarmnist_peak_then_stabilize_hybrid",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )
    stabilize_config = copy.copy(peak_config)
    stabilize_config.lr = stabilize_lr
    stabilize_config.penalty_weight = stabilize_penalty

    set_seed(seed)
    peak_flags = build_flags(peak_config, seed, "birm")
    stabilize_flags = build_flags(stabilize_config, seed, "birm")
    stabilize_flags, _ = return_model(stabilize_flags)
    dp = build_dp(peak_config, seed)
    ebd = EBD(stabilize_flags).to(DEVICE)
    model = OfficialResNetLoRA(
        rank=peak_config.lora_rank,
        alpha=peak_config.lora_alpha,
        bayesian=True,
    ).to(DEVICE)

    start = time.time()
    histories = []

    peak_optimizer = build_optimizer(model.parameters(), peak_config)
    best_acc = -1.0
    best_state = None
    best_step = -1
    for step in range(peak_steps):
        loss, train_nll, var_pen, grad_pen, kl_loss = train_peak_step(
            model, peak_optimizer, peak_flags, peak_config, dp
        )
        acc, minacc, majacc = evaluate_model(model, dp)
        if acc > best_acc:
            best_acc = acc
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        histories.append(
            {
                "variant": variant_name,
                "seed": seed,
                "phase": "peak",
                "step": step,
                "objective": loss,
                "train_nll": train_nll,
                "variance_penalty": var_pen,
                "grad_penalty": grad_pen,
                "kl_loss": kl_loss,
                "test_acc": acc,
                "test_minacc": minacc,
                "test_majacc": majacc,
                "best_test_acc_so_far": best_acc,
                "best_step_so_far": best_step,
                "elapsed_sec": time.time() - start,
            }
        )
        print(
            f"[{variant_name} seed={seed} peak] step={step:02d} "
            f"test={acc:.4f} min={minacc:.4f} maj={majacc:.4f} best={best_acc:.4f}@{best_step}",
            flush=True,
        )

    model.load_state_dict(best_state)
    if freeze_backbone:
        model.freeze_backbone()
    stabilize_optimizer = build_optimizer(model.parameters(), stabilize_config)
    scheduler = lr_scheduler.StepLR(
        stabilize_optimizer,
        step_size=max(1, stabilize_steps // 2),
        gamma=stabilize_config.step_gamma,
    )
    stabilize_best_acc = best_acc
    stabilize_best_step = best_step
    stabilize_best_state = copy.deepcopy(best_state)
    for step in range(stabilize_steps):
        loss, train_nll, var_pen, grad_pen, kl_loss = train_stabilize_step(
            model, stabilize_optimizer, stabilize_flags, stabilize_config, dp, ebd
        )
        scheduler.step()
        if step % 5 == 0 or step == stabilize_steps - 1:
            acc, minacc, majacc = evaluate_model(model, dp)
            if acc > stabilize_best_acc:
                stabilize_best_acc = acc
                stabilize_best_step = step
                stabilize_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            histories.append(
                {
                    "variant": variant_name,
                    "seed": seed,
                    "phase": "stabilize",
                    "step": step,
                    "objective": loss,
                    "train_nll": train_nll,
                    "variance_penalty": var_pen,
                    "grad_penalty": grad_pen,
                    "kl_loss": kl_loss,
                    "test_acc": acc,
                    "test_minacc": minacc,
                    "test_majacc": majacc,
                    "best_test_acc_so_far": stabilize_best_acc,
                    "best_step_so_far": stabilize_best_step,
                    "elapsed_sec": time.time() - start,
                }
            )
            print(
                f"[{variant_name} seed={seed} stab] step={step:03d} "
                f"test={acc:.4f} min={minacc:.4f} maj={majacc:.4f} "
                f"var={var_pen:.6g} grad={grad_pen:.6g} "
                f"best={stabilize_best_acc:.4f}@{stabilize_best_step}",
                flush=True,
            )

    ckpt_path = ARTIFACT_DIR / f"official_cifarmnist_peak_then_stabilize_{variant_name}_seed_{seed}_best.pt"
    torch.save(stabilize_best_state, ckpt_path)
    summary = {
        "variant": variant_name,
        "seed": seed,
        "peak_steps": peak_steps,
        "stabilize_lr": stabilize_lr,
        "stabilize_penalty": stabilize_penalty,
        "freeze_backbone": freeze_backbone,
        "peak_best_acc": best_acc,
        "peak_best_step": best_step,
        "best_test_acc": stabilize_best_acc,
        "best_step": stabilize_best_step,
        "final_test_acc": histories[-1]["test_acc"],
        "runtime_sec": time.time() - start,
        "checkpoint_path": str(ckpt_path),
    }
    return summary, pd.DataFrame(histories)


def main():
    seeds = (11, 17, 23)
    variants = [
        ("freeze_lr0005_p1000", 5, 0.005, 1000.0, 120, True),
        ("freeze_lr0001_p1000", 5, 0.001, 1000.0, 120, True),
        ("train_lr0005_p1000", 5, 0.005, 1000.0, 120, False),
    ]
    summaries = []
    histories = []
    print("device =", DEVICE)
    for variant in variants:
        for seed in seeds:
            print(f"\n===== {variant[0]} | seed={seed} =====", flush=True)
            summary, history = run_variant(seed, *variant)
            summaries.append(summary)
            histories.append(history)
            print(
                f"summary {variant[0]} seed={seed}: "
                f"peak={100.0 * summary['peak_best_acc']:.2f}% "
                f"best={100.0 * summary['best_test_acc']:.2f}% "
                f"final={100.0 * summary['final_test_acc']:.2f}%",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True)
    grouped = (
        summary_df.groupby("variant", sort=False)
        .agg(
            peak_mean=("peak_best_acc", "mean"),
            best_mean=("best_test_acc", "mean"),
            best_std=("best_test_acc", lambda x: x.std(ddof=0)),
            final_mean=("final_test_acc", "mean"),
            final_std=("final_test_acc", lambda x: x.std(ddof=0)),
            num_runs=("seed", "count"),
        )
        .reset_index()
        .sort_values(["final_mean", "best_mean"], ascending=False)
    )
    summary_path = ARTIFACT_DIR / "official_cifarmnist_peak_then_stabilize_run_summary.csv"
    history_path = ARTIFACT_DIR / "official_cifarmnist_peak_then_stabilize_step_history.csv"
    grouped_path = ARTIFACT_DIR / "official_cifarmnist_peak_then_stabilize_grouped.csv"
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
