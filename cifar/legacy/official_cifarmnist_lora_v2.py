import copy
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
REPO = THIS_DIR.parent / "birm_official"
LEGACY_REPO = Path("/root/Bayesian-Invariant-Risk-Minmization")
if not REPO.exists() and LEGACY_REPO.exists():
    REPO = LEGACY_REPO

ARTIFACT_DIR = ROOT / "official_cifarmnist_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
os.chdir(REPO)

sys.path.append(str(REPO))
sys.path.append(str(REPO / "dataset_scripts"))

import model as official_model  # noqa: E402
from model import EBD, resnet18_sepfc_us  # noqa: E402
from utils import CIFAR_LYPD, eval_acc_class, mean_nll_class, return_model  # noqa: E402


if not hasattr(official_model, "load_state_dict_from_url"):
    official_model.load_state_dict_from_url = torch.hub.load_state_dict_from_url


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class V2Config:
    dataset: str = "CifarMnist"
    batch_size: int = 1024
    steps: int = 500
    print_every: int = 50
    warmup_steps: int = 250
    penalty_anneal_iters: int = 250
    l2_regularizer_weight: float = 0.001
    lr: float = 0.01
    step_gamma: float = 1.0
    opt: str = "sgd"
    envs_num: int = 2
    image_scale: int = 64
    hidden_dim: int = 16
    grayscale_model: int = 0
    seeds: tuple = (11,)
    penalty_weights: tuple = (1000.0,)
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_mc_samples: int = 10
    train_base_backbone: bool = True
    pretrained_backbone: bool = False
    lora_layers: tuple = ("layer4",)
    experiment_name: str = "layer4_sgd_lr0.01_w250_s500"
    output_prefix: str = "official_cifarmnist_lora_v4_pretrained_seed11"


CONFIG = V2Config()


class Flags:
    pass


def prepare_official_cache():
    data_dir = REPO / "data_dir" / "cifar_mnist"
    data_dir.mkdir(parents=True, exist_ok=True)

    src_cifar_dir = ROOT / "data" / "cifar-10-batches-py"
    dst_cifar_dir = data_dir / "cifar-10-batches-py"
    if src_cifar_dir.exists() and not dst_cifar_dir.exists():
        shutil.copytree(src_cifar_dir, dst_cifar_dir)

    src_cifar_tar = ROOT / "data" / "cifar-10-python.tar.gz"
    dst_cifar_tar = data_dir / "cifar-10-python.tar.gz"
    if src_cifar_tar.exists() and src_cifar_tar.stat().st_size > 0 and not dst_cifar_tar.exists():
        shutil.copy2(src_cifar_tar, dst_cifar_tar)

    src_mnist = ROOT / "data" / "MNIST"
    dst_mnist = data_dir / "MNIST"
    if src_mnist.exists() and not dst_mnist.exists():
        shutil.copytree(src_mnist, dst_mnist)


def build_flags(config: V2Config, seed: int):
    flags = Flags()
    flags.envs_num = config.envs_num
    flags.batch_size = config.batch_size
    flags.seed = seed
    flags.dataset = config.dataset
    flags.opt = config.opt
    flags.l2_regularizer_weight = config.l2_regularizer_weight
    flags.print_every = config.print_every
    flags.data_num = 2000
    flags.lr = config.lr
    flags.env_type = "linear"
    flags.irm_type = "birm"
    flags.n_restarts = 1
    flags.image_scale = config.image_scale
    flags.hidden_dim = config.hidden_dim
    flags.step_gamma = config.step_gamma
    flags.penalty_anneal_iters = config.penalty_anneal_iters
    flags.penalty_weight = 0.0
    flags.steps = config.steps
    flags.grayscale_model = config.grayscale_model
    return return_model(flags)[0]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dp(config: V2Config, seed: int):
    flags = build_flags(config, seed)
    prepare_official_cache()
    return CIFAR_LYPD(flags)


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float, train_base: bool):
        super().__init__()
        self.conv = conv
        self.conv.weight.requires_grad_(train_base)
        if self.conv.bias is not None:
            self.conv.bias.requires_grad_(train_base)
        self.down = nn.Conv2d(
            conv.in_channels,
            rank,
            kernel_size=1,
            stride=conv.stride,
            padding=0,
            bias=False,
        )
        self.up = nn.Conv2d(rank, conv.out_channels, kernel_size=1, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_normal_(self.down.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.conv(x) + self.up(self.down(x)) * self.scaling


def inject_lora_layers(backbone, layers, rank: int, alpha: float, train_base: bool):
    for layer_name in layers:
        layer = getattr(backbone, layer_name)
        for block in layer:
            block.conv1 = LoRAConv2d(block.conv1, rank=rank, alpha=alpha, train_base=train_base)
            block.conv2 = LoRAConv2d(block.conv2, rank=rank, alpha=alpha, train_base=train_base)


class Layer4LoRAResNet(nn.Module):
    def __init__(self, config: V2Config):
        super().__init__()
        self.backbone = resnet18_sepfc_us(pretrained=config.pretrained_backbone, num_classes=1)
        if not config.train_base_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.backbone.class_classifier.parameters():
                p.requires_grad_(True)
        inject_lora_layers(
            self.backbone,
            layers=config.lora_layers,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            train_base=config.train_base_backbone,
        )

    def forward(self, x):
        return self.backbone(x)


def weight_norm_squared(model):
    total = torch.tensor(0.0, device=DEVICE)
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.norm().pow(2)
    return total


def build_optimizer(params, config: V2Config):
    params = [p for p in params if p.requires_grad]
    if config.opt == "adam":
        return optim.Adam(params, lr=config.lr)
    if config.opt == "sgd":
        return optim.SGD(params, lr=config.lr, momentum=0.9)
    raise ValueError(config.opt)


def split_train_groups(train_g):
    e1 = (train_g == 0).view(-1).nonzero().view(-1)
    e2 = (train_g == 1).view(-1).nonzero().view(-1)
    e1 = e1[torch.randperm(len(e1))]
    e2 = e2[torch.randperm(len(e2))]
    return torch.cat([e1[::2], e2[::2]]), torch.cat([e1[1::2], e2[1::2]])


def official_birm_penalty(logits, train_y, train_g, ebd, flags, config: V2Config):
    s1, s2 = split_train_groups(train_g)
    penalty = torch.tensor(0.0, device=DEVICE)
    for _ in range(config.lora_mc_samples):
        ebd.re_init_with_noise(flags.prior_sd_coef / flags.data_num)
        logits1 = ebd(train_g[s1]).view(-1, 1) * logits[s1]
        logits2 = ebd(train_g[s2]).view(-1, 1) * logits[s2]
        nll1 = mean_nll_class(logits1, train_y[s1])
        nll2 = mean_nll_class(logits2, train_y[s2])
        grad1 = torch.autograd.grad(nll1 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        grad2 = torch.autograd.grad(nll2 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        penalty = penalty + torch.mean(grad1 * grad2) / config.lora_mc_samples
    return penalty


def evaluate(model, dp):
    model.eval()
    test_x, test_y, test_g, test_c = dp.fetch_test()
    with torch.no_grad():
        logits = model(test_x)
    acc, minacc, majacc = eval_acc_class(logits, test_y, test_c)
    return (
        float(acc.detach().cpu().item()),
        float(minacc.detach().cpu().item()),
        float(majacc.detach().cpu().item()),
    )


def penalty_weight_at(step: int, max_weight: float, config: V2Config):
    if max_weight <= 0.0 or step < config.warmup_steps:
        return 0.0
    ramp_steps = max(1, config.steps - config.warmup_steps)
    return float(max_weight) * min(1.0, (step - config.warmup_steps + 1) / ramp_steps)


def run_variant(config: V2Config, seed: int, max_penalty_weight: float):
    set_seed(seed)
    flags = build_flags(config, seed)
    dp = build_dp(config, seed)
    model = Layer4LoRAResNet(config).to(DEVICE)
    ebd = EBD(flags).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, int(config.steps / 2)), gamma=config.step_gamma)
    method = f"{config.experiment_name} p={max_penalty_weight:g}"
    start = time.time()
    history = []
    best_acc = -1.0
    best_step = 0
    best_state = None
    log_lines = [
        "step train_loss train_nll train_penalty penalty_weight test_acc test_minacc test_majacc stage"
    ]

    for step in range(config.steps):
        model.train()
        train_x, train_y, train_g, train_c = dp.fetch_train()
        logits = model(train_x)
        train_nll = mean_nll_class(logits, train_y)
        current_penalty_weight = penalty_weight_at(step, max_penalty_weight, config)
        if current_penalty_weight > 0.0:
            train_penalty = official_birm_penalty(logits, train_y, train_g, ebd, flags, config)
            stage = "birm_ramp"
        else:
            train_penalty = torch.tensor(0.0, device=DEVICE)
            stage = "erm_warmup" if max_penalty_weight > 0.0 else "erm_only"

        loss = train_nll + config.l2_regularizer_weight * weight_norm_squared(model)
        loss = loss + current_penalty_weight * train_penalty
        if current_penalty_weight > 1.0:
            loss = loss / (1.0 + current_penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % config.print_every == 0 or step == config.steps - 1:
            test_acc, test_minacc, test_majacc = evaluate(model, dp)
            best_updated = test_acc > best_acc
            if best_updated:
                best_acc = test_acc
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            row = {
                "method": method,
                "seed": seed,
                "max_penalty_weight": max_penalty_weight,
                "step": step,
                "objective": float(loss.detach().cpu().item()),
                "train_nll": float(train_nll.detach().cpu().item()),
                "penalty": float(train_penalty.detach().cpu().item()),
                "penalty_weight": float(current_penalty_weight),
                "test_acc": test_acc,
                "test_minacc": test_minacc,
                "test_majacc": test_majacc,
                "best_test_acc_so_far": best_acc,
                "best_updated": best_updated,
                "elapsed_sec": time.time() - start,
                "stage": stage,
            }
            history.append(row)
            log_lines.append(
                f"{step} {row['objective']:.8g} {row['train_nll']:.8g} "
                f"{row['penalty']:.8g} {row['penalty_weight']:.8g} "
                f"{test_acc:.8g} {test_minacc:.8g} {test_majacc:.8g} {stage}"
            )

    if best_state is not None:
        ckpt_path = ARTIFACT_DIR / f"{config.output_prefix}_p{max_penalty_weight:g}_seed_{seed}_best.pt"
        torch.save(best_state, ckpt_path)

    summary = {
        "method": method,
        "seed": seed,
        "max_penalty_weight": max_penalty_weight,
        "best_test_acc": best_acc,
        "best_step": best_step,
        "runtime_sec": time.time() - start,
        "final_test_acc": history[-1]["test_acc"],
        "final_loss": history[-1]["objective"],
    }
    log_path = ARTIFACT_DIR / f"{config.output_prefix}_p{max_penalty_weight:g}_seed_{seed}.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    return summary, pd.DataFrame(history)


def build_final_table(summary_df):
    rows = []
    for method, group in summary_df.groupby("method", sort=False):
        rows.append({
            "method": method,
            "best_ood_acc_mean_pct": 100.0 * group["best_test_acc"].mean(),
            "selected_best_ood_acc_pct": 100.0 * group["best_test_acc"].max(),
            "num_runs": len(group),
            "runtime_total_sec": group["runtime_sec"].sum(),
            "best_step_mean": group["best_step"].mean(),
        })
    return pd.DataFrame(rows)


def run_all(config: V2Config):
    print("device =", DEVICE)
    print(asdict(config))
    summaries = []
    histories = []
    variants = [
        {
            "experiment_name": "pretrained_layer4_frozen_sgd_lr0.01_w250_s500",
            "steps": 500,
            "warmup_steps": 250,
            "penalty_anneal_iters": 250,
            "opt": "sgd",
            "lr": 0.01,
            "penalty_weights": (1000.0,),
            "lora_layers": ("layer4",),
            "pretrained_backbone": True,
            "train_base_backbone": False,
        },
        {
            "experiment_name": "pretrained_layer4_trainable_sgd_lr0.01_w250_s500",
            "steps": 500,
            "warmup_steps": 250,
            "penalty_anneal_iters": 250,
            "opt": "sgd",
            "lr": 0.01,
            "penalty_weights": (1000.0,),
            "lora_layers": ("layer4",),
            "pretrained_backbone": True,
            "train_base_backbone": True,
        },
    ]
    for seed in config.seeds:
        for variant in variants:
            variant_config = copy.copy(config)
            for key, value in variant.items():
                setattr(variant_config, key, value)
            variant_config.output_prefix = f"{config.output_prefix}_{variant_config.experiment_name}"
            for penalty_weight in variant_config.penalty_weights:
                print(
                    f"\n===== {variant_config.experiment_name} | seed={seed} | p={penalty_weight:g} =====",
                    flush=True,
                )
                summary, history = run_variant(variant_config, seed, penalty_weight)
                summaries.append(summary)
                histories.append(history)
                print(
                    f"best={100.0 * summary['best_test_acc']:.2f}% "
                    f"step={summary['best_step']} "
                    f"runtime={summary['runtime_sec']:.1f}s",
                    flush=True,
                )
    return pd.DataFrame(summaries), pd.concat(histories, ignore_index=True)


if __name__ == "__main__":
    summary_df, history_df = run_all(CONFIG)
    final_df = build_final_table(summary_df)
    summary_path = ARTIFACT_DIR / f"{CONFIG.output_prefix}_run_summary.csv"
    history_path = ARTIFACT_DIR / f"{CONFIG.output_prefix}_step_history.csv"
    final_path = ARTIFACT_DIR / f"{CONFIG.output_prefix}_final_table.csv"
    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    final_df.to_csv(final_path, index=False)
    print("\nsaved:", summary_path)
    print("saved:", history_path)
    print("saved:", final_path)
    print("\nFinal table:")
    print(final_df.sort_values("best_ood_acc_mean_pct", ascending=False).to_string(index=False))
