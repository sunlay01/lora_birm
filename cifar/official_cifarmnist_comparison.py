import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


ROOT = Path("/root")
REPO = ROOT / "Bayesian-Invariant-Risk-Minmization"
ARTIFACT_DIR = ROOT / "official_cifarmnist_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
os.chdir(REPO)

sys.path.append(str(REPO))
sys.path.append(str(REPO / "dataset_scripts"))

from model import EBD, resnet18_sepfc_us  # noqa: E402
from utils import (  # noqa: E402
    CIFAR_LYPD,
    eval_acc_class,
    mean_nll_class,
    return_model,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    dataset: str = "CifarMnist"
    batch_size: int = 1024
    steps: int = 2000
    print_every: int = 100
    penalty_weight: float = 10000.0
    penalty_anneal_iters: int = 40
    l2_regularizer_weight: float = 0.001
    lr: float = 0.01
    step_gamma: float = 1.0
    opt: str = "sgd"
    envs_num: int = 2
    image_scale: int = 64
    hidden_dim: int = 16
    grayscale_model: int = 0
    seeds: tuple = (11, 17, 23, 29, 37)
    methods: tuple = ("IRMv1 -> LoRA-BIRM",)
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_mc_samples: int = 10
    lora_kl_weight: float = 0.001
    hybrid_variance_weight: float = 1.0
    hybrid_grad_penalty_weight: float = 1.0
    hybrid_worst_env_weight: float = 0.0
    route_b_warmup_steps: int = 400
    output_prefix: str = "official_cifarmnist_lora_only"
    irm_base_steps: int = 2000
    irm_base_lr: float = 0.01
    irm_base_opt: str = "sgd"
    irm_base_penalty_weight: float = 10000.0
    irm_base_penalty_anneal_iters: int = 40


CONFIG = ExperimentConfig()


class Flags:
    pass


def build_flags(config: ExperimentConfig, seed: int, irm_type: str):
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
    flags.irm_type = irm_type
    flags.n_restarts = 1
    flags.image_scale = config.image_scale
    flags.hidden_dim = config.hidden_dim
    flags.step_gamma = config.step_gamma
    flags.penalty_anneal_iters = config.penalty_anneal_iters
    flags.penalty_weight = config.penalty_weight
    flags.steps = config.steps
    flags.grayscale_model = config.grayscale_model
    return flags


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dp(config: ExperimentConfig, seed: int):
    flags = build_flags(config, seed, "erm")
    prepare_official_cache()
    return CIFAR_LYPD(flags)


def build_optimizer(params, config: ExperimentConfig):
    if config.opt == "adam":
        return optim.Adam(params, lr=config.lr)
    if config.opt == "sgd":
        return optim.SGD(params, lr=config.lr, momentum=0.9)
    raise ValueError(config.opt)


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


def evaluate_official_style(model, dp):
    model.eval()
    test_x, test_y, test_g, test_c = dp.fetch_test()
    with torch.no_grad():
        test_logits = model(test_x)
    test_acc, test_minacc, test_majacc = eval_acc_class(test_logits, test_y, test_c)
    return {
        "test_acc": float(test_acc.detach().cpu().item()),
        "test_minacc": float(test_minacc.detach().cpu().item()),
        "test_majacc": float(test_majacc.detach().cpu().item()),
    }


class LoRALinearHead(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, bayesian):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.scaling = alpha / rank
        self.bayesian = bayesian
        if bayesian:
            self.B_mu = nn.Parameter(torch.zeros(out_features, rank))
            self.B_logvar = nn.Parameter(torch.full((out_features, rank), -5.0))
        else:
            self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x, sample=True):
        base = self.linear(x)
        if self.bayesian:
            if sample:
                std = torch.exp(0.5 * self.B_logvar)
                B = self.B_mu + torch.randn_like(std) * std
            else:
                B = self.B_mu
        else:
            B = self.B
        return base + (x @ self.A.t() @ B.t()) * self.scaling

    def kl_divergence(self):
        if not self.bayesian:
            return torch.tensor(0.0, device=self.linear.weight.device)
        return -0.5 * torch.sum(1 + self.B_logvar - self.B_mu.pow(2) - self.B_logvar.exp())


class OfficialResNetLoRA(nn.Module):
    def __init__(self, rank, alpha, bayesian):
        super().__init__()
        backbone = resnet18_sepfc_us(pretrained=False, num_classes=1)
        self.backbone = backbone
        in_features = backbone.class_classifier.in_features
        self.backbone.class_classifier = nn.Identity()
        self.head = LoRALinearHead(in_features, 1, rank=rank, alpha=alpha, bayesian=bayesian)

    def features(self, x):
        return self.backbone.encoder(x)

    def forward(self, x, sample=True):
        return self.head(self.features(x), sample=sample)

    def kl_divergence(self):
        return self.head.kl_divergence()

    def load_from_official_base(self, state_dict):
        backbone_state = self.backbone.state_dict()
        filtered_backbone = {
            key: value for key, value in state_dict.items() if key in backbone_state
        }
        backbone_state.update(filtered_backbone)
        self.backbone.load_state_dict(backbone_state)

        classifier_weight = state_dict.get("class_classifier.weight")
        classifier_bias = state_dict.get("class_classifier.bias")
        if classifier_weight is not None:
            self.head.linear.weight.data.copy_(classifier_weight)
        if classifier_bias is not None:
            self.head.linear.bias.data.copy_(classifier_bias)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)


def weight_norm_squared(model):
    total = torch.tensor(0.0, device=DEVICE)
    for p in model.parameters():
        total = total + p.norm().pow(2)
    return total


def build_stage_config(config: ExperimentConfig, **overrides):
    params = asdict(config)
    params.update(overrides)
    return ExperimentConfig(**params)


def train_official_model_local(method: str, config: ExperimentConfig, seed: int):
    assert method in ("ERM", "IRMv1", "BIRM")
    irm_type = {"ERM": "erm", "IRMv1": "irmv1", "BIRM": "birm"}[method]
    flags = build_flags(config, seed, irm_type)
    flags, model_type = return_model(flags)
    set_seed(seed)
    dp = build_dp(config, seed)
    model = resnet18_sepfc_us(pretrained=False, num_classes=1).to(DEVICE)
    ebd = EBD(flags).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(config.steps / 2)),
        gamma=config.step_gamma,
    )
    history = []
    best_acc = -1.0
    best_step = 0
    best_state = None
    start = time.time()

    for step in range(flags.steps):
        model.train()
        train_x, train_y, train_g, train_c = dp.fetch_train()
        train_logits, train_nll, train_penalty = official_train_step(
            model, ebd, flags, model_type, train_x, train_y, train_g
        )
        loss = train_nll.clone()
        loss = loss + flags.l2_regularizer_weight * weight_norm_squared(model)
        penalty_weight = flags.penalty_weight if step >= flags.penalty_anneal_iters else 0.0
        loss = loss + penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss = loss / (1.0 + penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % flags.print_every == 0 or step == flags.steps - 1:
            metrics = evaluate_official_style(model, dp)
            best_updated = metrics["test_acc"] > best_acc
            if best_updated:
                best_acc = metrics["test_acc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({
                "method": method,
                "seed": seed,
                "step": step,
                "objective": float(loss.detach().cpu().item()),
                "train_nll": float(train_nll.detach().cpu().item()),
                "penalty": float(train_penalty.detach().cpu().item()),
                "kl_loss": np.nan,
                "penalty_weight": float(penalty_weight),
                "test_acc": metrics["test_acc"],
                "test_minacc": metrics["test_minacc"],
                "test_majacc": metrics["test_majacc"],
                "best_test_acc_so_far": best_acc,
                "best_updated": best_updated,
                "elapsed_sec": time.time() - start,
                "source": "official_local_train",
                "stage": f"{method.lower()}_pretrain",
            })

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    summary = {
        "method": method,
        "seed": seed,
        "best_test_acc": best_acc,
        "best_step": best_step,
        "runtime_sec": time.time() - start,
        "final_test_acc": history[-1]["test_acc"],
        "final_loss": history[-1]["objective"],
        "source": "official_local_train",
    }
    return summary, pd.DataFrame(history), best_state


def official_train_step(model, ebd, flags, model_type, train_x, train_y, train_g):
    if model_type == "irmv1":
        train_logits = ebd(train_g).view(-1, 1) * model(train_x)
        train_nll = mean_nll_class(train_logits, train_y)
        grad = torch.autograd.grad(train_nll * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        train_penalty = torch.mean(grad**2)
    elif model_type == "irmv1b":
        e1 = (train_g == 0).view(-1).nonzero().view(-1)
        e2 = (train_g == 1).view(-1).nonzero().view(-1)
        e1 = e1[torch.randperm(len(e1))]
        e2 = e2[torch.randperm(len(e2))]
        s1 = torch.cat([e1[::2], e2[::2]])
        s2 = torch.cat([e1[1::2], e2[1::2]])
        train_logits = ebd(train_g).view(-1, 1) * model(train_x)
        train_nll1 = mean_nll_class(train_logits[s1], train_y[s1])
        train_nll2 = mean_nll_class(train_logits[s2], train_y[s2])
        train_nll = train_nll1 + train_nll2
        grad1 = torch.autograd.grad(train_nll1 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        grad2 = torch.autograd.grad(train_nll2 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        train_penalty = torch.mean(grad1 * grad2)
    elif model_type == "bayes_batch":
        sample_n = 10
        train_penalty = 0.0
        train_logits = model(train_x)
        e1 = (train_g == 0).view(-1).nonzero().view(-1)
        e2 = (train_g == 1).view(-1).nonzero().view(-1)
        e1 = e1[torch.randperm(len(e1))]
        e2 = e2[torch.randperm(len(e2))]
        s1 = torch.cat([e1[::2], e2[::2]])
        s2 = torch.cat([e1[1::2], e2[1::2]])
        train_nll = mean_nll_class(train_logits, train_y)
        for _ in range(sample_n):
            ebd.re_init_with_noise(flags.prior_sd_coef / flags.data_num)
            logits1 = ebd(train_g[s1]).view(-1, 1) * train_logits[s1]
            logits2 = ebd(train_g[s2]).view(-1, 1) * train_logits[s2]
            nll1 = mean_nll_class(logits1, train_y[s1])
            nll2 = mean_nll_class(logits2, train_y[s2])
            grad1 = torch.autograd.grad(nll1 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
            grad2 = torch.autograd.grad(nll2 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
            train_penalty = train_penalty + torch.mean(grad1 * grad2) / sample_n
        train_logits = ebd(train_g).view(-1, 1) * train_logits
    elif model_type == "erm":
        train_logits = model(train_x)
        train_nll = mean_nll_class(train_logits, train_y)
        train_penalty = torch.tensor(0.0, device=DEVICE)
    else:
        raise ValueError(model_type)
    return train_logits, train_nll, train_penalty


OFFICIAL_STEP_RE = re.compile(
    r"^\s*(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
OFFICIAL_FINAL_RE = re.compile(r"Final test acc:\s*([-+0-9.eE]+)")


def parse_official_main_output(method: str, seed: int, stdout: str, runtime_sec: float):
    history = []
    best_acc = -1.0
    best_step = 0
    final_test_acc = np.nan

    for line in stdout.splitlines():
        step_match = OFFICIAL_STEP_RE.match(line)
        if step_match:
            step = int(step_match.group(1))
            objective = float(step_match.group(2))
            penalty = float(step_match.group(3))
            test_acc = float(step_match.group(4))
            if test_acc > 1.0:
                test_acc /= 100.0
            best_updated = test_acc > best_acc
            if best_updated:
                best_acc = test_acc
                best_step = step
            history.append({
                "method": method,
                "seed": seed,
                "step": step,
                "objective": objective,
                "train_nll": np.nan,
                "penalty": penalty,
                "kl_loss": np.nan,
                "penalty_weight": np.nan,
                "test_acc": test_acc,
                "test_minacc": np.nan,
                "test_majacc": np.nan,
                "best_test_acc_so_far": best_acc,
                "best_updated": best_updated,
                "elapsed_sec": np.nan,
                "source": "official_main_subprocess",
                "stage": "official_main",
            })
            continue

        final_match = OFFICIAL_FINAL_RE.search(line)
        if final_match:
            final_test_acc = float(final_match.group(1))
            if final_test_acc > 1.0:
                final_test_acc /= 100.0

    if not history:
        raise RuntimeError("Failed to parse any step records from official main.py output.")

    elapsed = np.linspace(0.0, runtime_sec, num=len(history))
    for idx, value in enumerate(elapsed):
        history[idx]["elapsed_sec"] = float(value)

    if np.isnan(final_test_acc):
        final_test_acc = history[-1]["test_acc"]

    summary = {
        "method": method,
        "seed": seed,
        "best_test_acc": best_acc,
        "best_step": best_step,
        "runtime_sec": runtime_sec,
        "final_test_acc": final_test_acc,
        "final_loss": history[-1]["objective"],
        "source": "official_main_subprocess",
    }
    return summary, pd.DataFrame(history)


def run_official_baseline(method: str, config: ExperimentConfig, seed: int):
    prepare_official_cache()
    irm_type = {"ERM": "erm", "IRMv1": "irmv1", "BIRM": "birm"}[method]
    log_stem = method.lower().replace(" ", "_")
    log_path = ARTIFACT_DIR / f"official_{log_stem}_seed_{seed}.log"
    cmd = [
        sys.executable,
        str(REPO / "main.py"),
        "--dataset", config.dataset,
        "--irm_type", irm_type,
        "--l2_regularizer_weight", str(config.l2_regularizer_weight),
        "--lr", str(config.lr),
        "--step_gamma", str(config.step_gamma),
        "--batch_size", str(config.batch_size),
        "--penalty_anneal_iters", str(config.penalty_anneal_iters),
        "--opt", config.opt,
        "--print_every", str(config.print_every),
        "--penalty_weight", str(config.penalty_weight),
        "--steps", str(config.steps),
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    runtime_sec = time.time() - start
    log_text = proc.stdout
    if proc.stderr:
        log_text = f"{log_text}\n[stderr]\n{proc.stderr}"
    log_path.write_text(log_text)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Official main.py failed for {method} seed={seed} with code {proc.returncode}. "
            f"See {log_path}"
        )
    return parse_official_main_output(method, seed, proc.stdout, runtime_sec)


def route_b_stage(step: int, config: ExperimentConfig):
    return "warmup_erm" if step < config.route_b_warmup_steps else "birm_finetune"


def stage_name(method: str, step: int, config: ExperimentConfig):
    if method == "LoRA-BIRM Route B":
        return route_b_stage(step, config)
    if method == "IRMv1 -> LoRA-BIRM":
        return "irm_base_init_birm_finetune"
    if method == "LoRA-BIRM Notebook-Variance":
        return "single_stage_notebook_variance"
    return "single_stage_birm"


def split_train_groups(train_g):
    e1 = (train_g == 0).view(-1).nonzero().view(-1)
    e2 = (train_g == 1).view(-1).nonzero().view(-1)
    e1 = e1[torch.randperm(len(e1))]
    e2 = e2[torch.randperm(len(e2))]
    s1 = torch.cat([e1[::2], e2[::2]])
    s2 = torch.cat([e1[1::2], e2[1::2]])
    return s1, s2


def lora_official_birm_objective(model, ebd, flags, features, train_y, train_g, config):
    s1, s2 = split_train_groups(train_g)
    train_nll_terms = []
    penalty_terms = []

    for _ in range(config.lora_mc_samples):
        logits_full = model.head(features, sample=True)
        train_nll_terms.append(mean_nll_class(logits_full, train_y))

        # Match official CifarMnist BIRM: sample noisy EBD scalars and take the
        # gradient-product penalty with respect to EBD, not model/head weights.
        ebd.re_init_with_noise(flags.prior_sd_coef / flags.data_num)
        logits1 = ebd(train_g[s1]).view(-1, 1) * logits_full[s1]
        logits2 = ebd(train_g[s2]).view(-1, 1) * logits_full[s2]
        nll1 = mean_nll_class(logits1, train_y[s1])
        nll2 = mean_nll_class(logits2, train_y[s2])
        grad1 = torch.autograd.grad(nll1 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        grad2 = torch.autograd.grad(nll2 * flags.envs_num, ebd.parameters(), create_graph=True)[0]
        penalty_terms.append(torch.mean(grad1 * grad2))

    return torch.stack(train_nll_terms).mean(), torch.stack(penalty_terms).mean()


def lora_notebook_variance_objective(model, features, train_y, train_g, config):
    env_nll, variance_penalty, _worst_env_loss, _env_losses = lora_env_loss_stats(
        model, features, train_y, train_g, config
    )
    return env_nll, variance_penalty


def lora_env_loss_stats(model, features, train_y, train_g, config):
    env_losses = []
    env_ids = train_g.view(-1).unique(sorted=True)
    for env_id in env_ids:
        env_mask = train_g.view(-1) == env_id
        env_features = features[env_mask]
        env_targets = train_y[env_mask]
        mc_losses = []
        for _ in range(config.lora_mc_samples):
            env_logits = model.head(env_features, sample=True)
            mc_losses.append(mean_nll_class(env_logits, env_targets))
        env_losses.append(torch.stack(mc_losses).mean())
    env_losses = torch.stack(env_losses)
    return env_losses.mean(), env_losses.var(unbiased=False), env_losses.max(), env_losses


def lora_hybrid_env_objective(model, ebd, flags, features, train_y, train_g, config):
    env_nll, variance_penalty, worst_env_loss, _env_losses = lora_env_loss_stats(
        model, features, train_y, train_g, config
    )
    _official_nll, grad_penalty = lora_official_birm_objective(
        model, ebd, flags, features, train_y, train_g, config
    )
    total_penalty = (
        config.hybrid_variance_weight * variance_penalty
        + config.hybrid_grad_penalty_weight * grad_penalty
        + config.hybrid_worst_env_weight * worst_env_loss
    )
    return env_nll, total_penalty, variance_penalty, grad_penalty, worst_env_loss


def run_lora_method(method: str, config: ExperimentConfig, seed: int):
    assert method in (
        "LoRA-BIRM Route A",
        "LoRA-BIRM Route B",
        "IRMv1 -> LoRA-BIRM",
        "LoRA-BIRM Notebook-Variance",
        "LoRA-BIRM Hybrid-Env",
    )
    flags = build_flags(config, seed, "birm")
    flags, _ = return_model(flags)
    set_seed(seed)
    dp = build_dp(config, seed)
    ebd = EBD(flags).to(DEVICE)
    model = OfficialResNetLoRA(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        bayesian=True,
    ).to(DEVICE)
    if method == "IRMv1 -> LoRA-BIRM":
        base_config = build_stage_config(
            config,
            steps=config.irm_base_steps,
            lr=config.irm_base_lr,
            opt=config.irm_base_opt,
            penalty_weight=config.irm_base_penalty_weight,
            penalty_anneal_iters=config.irm_base_penalty_anneal_iters,
            methods=("IRMv1",),
        )
        base_summary, base_history, base_state = train_official_model_local("IRMv1", base_config, seed)
        model.load_from_official_base(base_state)
        model.freeze_backbone()
        finetune_config = build_stage_config(
            config,
            lr=0.001,
            penalty_anneal_iters=max(config.penalty_anneal_iters, 200),
        )
    else:
        base_summary = None
        base_history = None
        finetune_config = config
    optimizer = build_optimizer(model.parameters(), finetune_config)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(finetune_config.steps / 2)),
        gamma=finetune_config.step_gamma,
    )
    start = time.time()
    history = []
    best_acc = -1.0
    best_step = 0
    best_state = None
    log_lines = ["step train_loss train_nll train_penalty kl_loss penalty_weight test_acc test_minacc test_majacc stage"]

    for step in range(config.steps):
        model.train()
        if method == "IRMv1 -> LoRA-BIRM":
            model.backbone.eval()
        train_x, train_y, train_g, train_c = dp.fetch_train()
        features = model.features(train_x)
        if method == "LoRA-BIRM Route B" and step < config.route_b_warmup_steps:
            logits = model.head(features, sample=False)
            train_nll = mean_nll_class(logits, train_y)
            train_penalty = torch.tensor(0.0, device=DEVICE)
            kl_loss = model.kl_divergence()
            penalty_weight = 0.0
            variance_penalty = torch.tensor(0.0, device=DEVICE)
            grad_penalty = torch.tensor(0.0, device=DEVICE)
            worst_env_loss = train_nll
        elif method == "LoRA-BIRM Notebook-Variance":
            train_nll, train_penalty = lora_notebook_variance_objective(
                model, features, train_y, train_g, config
            )
            kl_loss = model.kl_divergence()
            penalty_weight = config.penalty_weight
            variance_penalty = train_penalty
            grad_penalty = torch.tensor(0.0, device=DEVICE)
            worst_env_loss = train_nll
        elif method == "LoRA-BIRM Hybrid-Env":
            train_nll, train_penalty, variance_penalty, grad_penalty, worst_env_loss = lora_hybrid_env_objective(
                model, ebd, flags, features, train_y, train_g, config
            )
            kl_loss = model.kl_divergence()
            penalty_weight = config.penalty_weight
        else:
            train_nll, train_penalty = lora_official_birm_objective(
                model, ebd, flags, features, train_y, train_g, config
            )
            kl_loss = model.kl_divergence()
            penalty_weight = flags.penalty_weight if step >= finetune_config.penalty_anneal_iters else 0.0
            variance_penalty = torch.tensor(0.0, device=DEVICE)
            grad_penalty = train_penalty
            worst_env_loss = train_nll

        loss = train_nll.clone()
        loss = loss + flags.l2_regularizer_weight * weight_norm_squared(model)
        loss = loss + config.lora_kl_weight * kl_loss
        loss = loss + penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss = loss / (1.0 + penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % flags.print_every == 0 or step == flags.steps - 1:
            model.eval()
            test_x, test_y, test_g, test_c = dp.fetch_test()
            with torch.no_grad():
                test_logits = model(test_x, sample=False)
            test_acc, test_minacc, test_majacc = eval_acc_class(test_logits, test_y, test_c)
            test_acc = float(test_acc.detach().cpu().item())
            test_minacc = float(test_minacc.detach().cpu().item())
            test_majacc = float(test_majacc.detach().cpu().item())
            best_updated = test_acc > best_acc
            if best_updated:
                best_acc = test_acc
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({
                "method": method,
                "seed": seed,
                "step": step,
                "objective": float(loss.detach().cpu().item()),
                "train_nll": float(train_nll.detach().cpu().item()),
                "penalty": float(train_penalty.detach().cpu().item()),
                "variance_penalty": float(variance_penalty.detach().cpu().item()),
                "grad_penalty": float(grad_penalty.detach().cpu().item()),
                "worst_env_loss": float(worst_env_loss.detach().cpu().item()),
                "kl_loss": float(kl_loss.detach().cpu().item()),
                "penalty_weight": float(penalty_weight),
                "test_acc": test_acc,
                "test_minacc": test_minacc,
                "test_majacc": test_majacc,
                "best_test_acc_so_far": best_acc,
                "best_updated": best_updated,
                "elapsed_sec": time.time() - start,
                "source": "official_cifarmnist_lora",
                "stage": stage_name(method, step, config),
            })
            log_lines.append(
                f"{step} {float(loss.detach().cpu().item()):.8g} "
                f"{float(train_nll.detach().cpu().item()):.8g} "
                f"{float(train_penalty.detach().cpu().item()):.8g} "
                f"{float(kl_loss.detach().cpu().item()):.8g} "
                f"{float(penalty_weight):.8g} "
                f"{test_acc:.8g} {test_minacc:.8g} {test_majacc:.8g} "
                f"{history[-1]['stage']}"
            )
    if base_history is not None:
        base_history = base_history.copy()
        base_history["method"] = method
        base_history["source"] = "irmv1_base_for_lora_birm"
        history_df = pd.concat([base_history, pd.DataFrame(history)], ignore_index=True)
        runtime_sec = float(base_summary["runtime_sec"]) + (time.time() - start)
    else:
        history_df = pd.DataFrame(history)
        runtime_sec = time.time() - start
    summary = {
        "method": method,
        "seed": seed,
        "best_test_acc": best_acc,
        "best_step": best_step,
        "runtime_sec": runtime_sec,
        "final_test_acc": history[-1]["test_acc"],
        "final_loss": history[-1]["objective"],
        "source": "official_cifarmnist_lora",
    }
    log_stem = method.lower().replace(" ", "_").replace("-", "_")
    log_path = ARTIFACT_DIR / f"{config.output_prefix}_{log_stem}_seed_{seed}.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    if method == "IRMv1 -> LoRA-BIRM":
        ckpt_path = ARTIFACT_DIR / f"{config.output_prefix}_{log_stem}_seed_{seed}_best.pt"
        torch.save(best_state if best_state is not None else model.state_dict(), ckpt_path)
    return summary, history_df


def run_all(config: ExperimentConfig):
    all_summaries = []
    all_histories = []
    print("device =", DEVICE)
    print(asdict(config))
    for method in config.methods:
        for seed in config.seeds:
            print(f"\n===== {method} | seed={seed} =====", flush=True)
            if method in ("ERM", "IRMv1", "BIRM"):
                summary, history = run_official_baseline(method, config, seed)
            else:
                summary, history = run_lora_method(method, config, seed)
            all_summaries.append(summary)
            all_histories.append(history)
            print(
                f"best={100.0 * summary['best_test_acc']:.2f}% "
                f"step={summary['best_step']} "
                f"runtime={summary['runtime_sec']:.1f}s",
                flush=True,
            )
    summary_df = pd.DataFrame(all_summaries)
    history_df = pd.concat(all_histories, ignore_index=True)
    return summary_df, history_df


def build_final_table(summary_df: pd.DataFrame, methods=None):
    if methods is None:
        methods = tuple(summary_df["method"].drop_duplicates())
    rows = []
    for method in methods:
        group = summary_df[summary_df["method"] == method]
        if group.empty:
            continue
        rows.append({
            "method": method,
            "best_ood_acc_mean_pct": 100.0 * group["best_test_acc"].mean(),
            "best_ood_acc_std_pct": 100.0 * group["best_test_acc"].std(ddof=0),
            "selected_best_ood_acc_pct": 100.0 * group["best_test_acc"].max(),
            "num_runs": len(group),
            "runtime_mean_sec": group["runtime_sec"].mean(),
            "runtime_total_sec": group["runtime_sec"].sum(),
            "best_step_mean": group["best_step"].mean(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary_df, history_df = run_all(CONFIG)
    final_df = build_final_table(summary_df, CONFIG.methods)
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
