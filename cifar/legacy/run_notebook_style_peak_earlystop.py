import time
from dataclasses import asdict

import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler

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


def main():
    config = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=60,
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
        seeds=(11,),
        methods=("LoRA-BIRM Notebook-Variance",),
        lora_rank=16,
        lora_alpha=16.0,
        lora_mc_samples=3,
        lora_kl_weight=0.0003,
        route_b_warmup_steps=400,
        output_prefix="official_cifarmnist_notebook_peak_earlystop",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )
    seed = config.seeds[0]
    patience = 10
    min_steps_before_stop = 15
    min_delta = 1e-6

    print("device =", DEVICE)
    print(asdict(config))
    print(
        f"early_stop: patience={patience}, min_steps_before_stop={min_steps_before_stop}, min_delta={min_delta}"
    )

    flags = build_flags(config, seed, "birm")
    set_seed(seed)
    dp = build_dp(config, seed)
    model = OfficialResNetLoRA(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        bayesian=True,
    ).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(config.steps / 2)),
        gamma=config.step_gamma,
    )

    start = time.time()
    best_acc = -1.0
    best_step = -1
    best_state = None
    best_metrics = None
    since_improvement = 0
    history = []

    for step in range(config.steps):
        model.train()
        train_x, train_y, train_g, train_c = dp.fetch_train()
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
        scheduler.step()

        model.eval()
        test_x, test_y, test_g, test_c = dp.fetch_test()
        with torch.no_grad():
            test_logits = model(test_x, sample=False)
        test_acc, test_minacc, test_majacc = eval_acc_class(test_logits, test_y, test_c)
        test_acc = float(test_acc.detach().cpu().item())
        test_minacc = float(test_minacc.detach().cpu().item())
        test_majacc = float(test_majacc.detach().cpu().item())

        improved = test_acc > best_acc + min_delta
        if improved:
            best_acc = test_acc
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "test_acc": test_acc,
                "test_minacc": test_minacc,
                "test_majacc": test_majacc,
            }
            since_improvement = 0
        else:
            since_improvement += 1

        row = {
            "seed": seed,
            "step": step,
            "objective": float(loss.detach().cpu().item()),
            "train_nll": float(train_nll.detach().cpu().item()),
            "penalty": float(train_penalty.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "penalty_weight": config.penalty_weight,
            "test_acc": test_acc,
            "test_minacc": test_minacc,
            "test_majacc": test_majacc,
            "best_test_acc_so_far": best_acc,
            "best_step_so_far": best_step,
            "since_improvement": since_improvement,
            "elapsed_sec": time.time() - start,
        }
        history.append(row)
        print(
            f"step={step:02d} loss={row['objective']:.6f} "
            f"nll={row['train_nll']:.6f} pen={row['penalty']:.6f} kl={row['kl_loss']:.6f} "
            f"test={test_acc:.4f} min={test_minacc:.4f} maj={test_majacc:.4f} "
            f"best={best_acc:.4f}@{best_step} wait={since_improvement}"
        )

        if step >= min_steps_before_stop and since_improvement >= patience:
            print(f"early stopping triggered at step {step}")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was recorded.")

    ckpt_path = ARTIFACT_DIR / f"{config.output_prefix}_seed_{seed}_best.pt"
    history_path = ARTIFACT_DIR / f"{config.output_prefix}_step_history.csv"
    summary_path = ARTIFACT_DIR / f"{config.output_prefix}_run_summary.csv"

    torch.save(best_state, ckpt_path)
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    summary_df = pd.DataFrame(
        [
            {
                "seed": seed,
                "best_test_acc": best_acc,
                "best_step": best_step,
                "best_test_minacc": best_metrics["test_minacc"],
                "best_test_majacc": best_metrics["test_majacc"],
                "stopped_after_step": history[-1]["step"],
                "final_test_acc": history[-1]["test_acc"],
                "runtime_sec": time.time() - start,
                "checkpoint_path": str(ckpt_path),
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    print("\nsaved:", ckpt_path)
    print("saved:", history_path)
    print("saved:", summary_path)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
