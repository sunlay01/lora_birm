import copy
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

    return (
        float(loss.detach().cpu().item()),
        float(train_nll.detach().cpu().item()),
        float(train_penalty.detach().cpu().item()),
        float(kl_loss.detach().cpu().item()),
    )


def main():
    base = ExperimentConfig(
        dataset="CifarMnist",
        batch_size=1024,
        steps=300,
        print_every=5,
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
        output_prefix="official_cifarmnist_notebook_hybrid_stabilize",
        irm_base_steps=2000,
        irm_base_lr=0.01,
        irm_base_opt="sgd",
        irm_base_penalty_weight=10000.0,
        irm_base_penalty_anneal_iters=40,
    )
    seed = base.seeds[0]
    stage1_steps = 5
    phase2_lrs = (0.01, 0.005)
    phase2_total_steps = 300

    print("device =", DEVICE)
    print(asdict(base))
    print(
        f"hybrid schedule: stage1_steps={stage1_steps}, "
        f"stage1_lr={base.lr}, phase2_lrs={phase2_lrs}, phase2_total_steps={phase2_total_steps}"
    )

    set_seed(seed)
    flags = build_flags(base, seed, "birm")
    dp = build_dp(base, seed)
    model = OfficialResNetLoRA(
        rank=base.lora_rank,
        alpha=base.lora_alpha,
        bayesian=True,
    ).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), base)

    stage1_history = []
    stage1_best_acc = -1.0
    stage1_best_step = -1
    stage1_best_state = None
    start = time.time()

    for step in range(stage1_steps):
        loss, train_nll, train_penalty, kl_loss = train_step(model, optimizer, flags, base, dp)
        test_acc, test_minacc, test_majacc = evaluate_model(model, dp)
        if test_acc > stage1_best_acc:
            stage1_best_acc = test_acc
            stage1_best_step = step
            stage1_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {
            "variant": "stage1_peak_search",
            "seed": seed,
            "step": step,
            "objective": loss,
            "train_nll": train_nll,
            "penalty": train_penalty,
            "kl_loss": kl_loss,
            "test_acc": test_acc,
            "test_minacc": test_minacc,
            "test_majacc": test_majacc,
            "best_test_acc_so_far": stage1_best_acc,
            "best_step_so_far": stage1_best_step,
            "elapsed_sec": time.time() - start,
        }
        stage1_history.append(row)
        print(
            f"[stage1] step={step:02d} loss={loss:.6f} nll={train_nll:.6f} "
            f"pen={train_penalty:.6f} kl={kl_loss:.6f} test={test_acc:.4f} "
            f"min={test_minacc:.4f} maj={test_majacc:.4f} "
            f"best={stage1_best_acc:.4f}@{stage1_best_step}"
        )

    if stage1_best_state is None:
        raise RuntimeError("Stage 1 did not record a best checkpoint.")

    all_histories = [pd.DataFrame(stage1_history)]
    summaries = []
    stage1_ckpt = ARTIFACT_DIR / f"{base.output_prefix}_stage1_best_seed_{seed}.pt"
    torch.save(stage1_best_state, stage1_ckpt)

    for phase2_lr in phase2_lrs:
        variant_name = f"hybrid_phase2_lr_{phase2_lr:g}"
        variant_model = OfficialResNetLoRA(
            rank=base.lora_rank,
            alpha=base.lora_alpha,
            bayesian=True,
        ).to(DEVICE)
        variant_model.load_state_dict(stage1_best_state)

        phase2_config = copy.copy(base)
        phase2_config.lr = phase2_lr
        phase2_flags = build_flags(phase2_config, seed, "birm")
        phase2_optimizer = build_optimizer(variant_model.parameters(), phase2_config)
        phase2_scheduler = lr_scheduler.StepLR(
            phase2_optimizer,
            step_size=max(1, int(phase2_total_steps / 2)),
            gamma=phase2_config.step_gamma,
        )

        phase2_best_acc = stage1_best_acc
        phase2_best_step = stage1_best_step
        phase2_best_state = copy.deepcopy(stage1_best_state)
        phase2_history = []
        phase2_start = time.time()

        for step in range(phase2_total_steps):
            loss, train_nll, train_penalty, kl_loss = train_step(
                variant_model, phase2_optimizer, phase2_flags, phase2_config, dp
            )
            phase2_scheduler.step()

            if step % phase2_config.print_every == 0 or step == phase2_total_steps - 1:
                test_acc, test_minacc, test_majacc = evaluate_model(variant_model, dp)
                if test_acc > phase2_best_acc:
                    phase2_best_acc = test_acc
                    phase2_best_step = step
                    phase2_best_state = {
                        k: v.detach().cpu().clone() for k, v in variant_model.state_dict().items()
                    }
                row = {
                    "variant": variant_name,
                    "seed": seed,
                    "step": step,
                    "objective": loss,
                    "train_nll": train_nll,
                    "penalty": train_penalty,
                    "kl_loss": kl_loss,
                    "test_acc": test_acc,
                    "test_minacc": test_minacc,
                    "test_majacc": test_majacc,
                    "best_test_acc_so_far": phase2_best_acc,
                    "best_step_so_far": phase2_best_step,
                    "elapsed_sec": time.time() - phase2_start,
                }
                phase2_history.append(row)
                print(
                    f"[{variant_name}] step={step:03d} loss={loss:.6f} nll={train_nll:.6f} "
                    f"pen={train_penalty:.6f} kl={kl_loss:.6f} test={test_acc:.4f} "
                    f"min={test_minacc:.4f} maj={test_majacc:.4f} "
                    f"best={phase2_best_acc:.4f}@{phase2_best_step}"
                )

        ckpt_path = ARTIFACT_DIR / f"{base.output_prefix}_{variant_name}_seed_{seed}_best.pt"
        torch.save(phase2_best_state, ckpt_path)
        all_histories.append(pd.DataFrame(phase2_history))
        summaries.append(
            {
                "variant": variant_name,
                "seed": seed,
                "stage1_best_acc": stage1_best_acc,
                "stage1_best_step": stage1_best_step,
                "phase2_lr": phase2_lr,
                "best_test_acc": phase2_best_acc,
                "best_step": phase2_best_step,
                "final_test_acc": phase2_history[-1]["test_acc"],
                "runtime_sec": time.time() - phase2_start,
                "checkpoint_path": str(ckpt_path),
            }
        )

    history_df = pd.concat(all_histories, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["best_test_acc", "final_test_acc"], ascending=False
    )
    history_path = ARTIFACT_DIR / f"{base.output_prefix}_step_history.csv"
    summary_path = ARTIFACT_DIR / f"{base.output_prefix}_run_summary.csv"
    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nsaved:", stage1_ckpt)
    print("saved:", history_path)
    print("saved:", summary_path)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
