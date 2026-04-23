import gc
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from modelscope import snapshot_download
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HF_ENDPOINT", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


DEFAULT_MAX_STEPS = 400
SEED = get_env_int("QWEN_SEED", 42)
MAX_STEPS = get_env_int("QWEN_MAX_STEPS", DEFAULT_MAX_STEPS)
if MAX_STEPS <= 0:
    raise ValueError(f"QWEN_MAX_STEPS must be positive, got {MAX_STEPS}")
DEFAULT_RUN_NAME = f"qwen_lora_birm_stable_{MAX_STEPS}_lr2e-5_pen3_ema98"
EVAL_INTERVAL = 25
BATCH = 8
VAL_BATCH = 16
MAX_LENGTH = 128
HEAD_LR = 8e-4
LORA_LR = 2e-5
WEIGHT_DECAY = 0.01
ETA_MIN = 1e-6
EMA_DECAY = 0.98
KL_WEIGHT = 1e-4
MAX_SNAPSHOT_PENALTY = 3.0
PENALTY_WARMUP_RATIO = 2.0 / 3.0
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LAST_N_LAYERS = 6

RUN_NAME = os.getenv(
    "QWEN_RUN_NAME",
    DEFAULT_RUN_NAME if SEED == 42 else f"{DEFAULT_RUN_NAME}_seed{SEED}",
)
OUT_ROOT = Path(os.getenv("QWEN_OUT_ROOT", "/root/qwen_lora_birm_tuning"))
OUT_DIR = OUT_ROOT / RUN_NAME
MODEL_CACHE_DIR = Path(os.getenv("QWEN_MODEL_CACHE_DIR", "/root/autodl-tmp/modelscope_cache"))
DATA_PARQUET = Path(os.getenv("QWEN_DATA_PARQUET", "/root/train-00000-of-00001.parquet"))
STEP_HISTORY_CSV = OUT_DIR / "step_history.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
BEST_HEAD_PATH = OUT_DIR / "best_head.pt"
BEST_LORA_DIR = OUT_DIR / "best_lora_adapter"
CONFIG_JSON = OUT_DIR / "config.json"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_csv(path, row):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def prepare_data():
    print("[1/5] Loading parquet and building reversed shortcut environment...", flush=True)
    df = pd.read_parquet(DATA_PARQUET)
    if "comment_text" in df.columns:
        df.rename(columns={"comment_text": "text"}, inplace=True)
    if "target" in df.columns:
        df.rename(columns={"target": "toxicity"}, inplace=True)

    identity_cols = ["male", "female", "LGBTQ", "christian", "muslim", "black", "white"]
    present_identity_cols = [c for c in identity_cols if c in df.columns]
    df["toxicity"] = (df["toxicity"] >= 0.5).astype(int)
    df["identity_sum"] = df[present_identity_cols].sum(axis=1)

    group_toxic_with_id = df[(df["toxicity"] == 1) & (df["identity_sum"] > 0)]
    group_toxic_no_id = df[(df["toxicity"] == 1) & (df["identity_sum"] == 0)]
    group_safe_with_id = df[(df["toxicity"] == 0) & (df["identity_sum"] > 0)]
    group_safe_no_id = df[(df["toxicity"] == 0) & (df["identity_sum"] == 0)]

    n_toxic_train = 10000
    train_toxic_with_id = group_toxic_with_id.sample(n=int(n_toxic_train * 0.9), random_state=42)
    train_toxic_no_id = group_toxic_no_id.sample(n=int(n_toxic_train * 0.1), random_state=42)

    n_safe_train = 10000
    train_safe_with_id = group_safe_with_id.sample(n=int(n_safe_train * 0.1), random_state=42)
    train_safe_no_id = group_safe_no_id.sample(n=int(n_safe_train * 0.9), random_state=42)

    train_df = pd.concat(
        [train_toxic_with_id, train_toxic_no_id, train_safe_with_id, train_safe_no_id]
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    remain_toxic_no_id = group_toxic_no_id.drop(train_toxic_no_id.index)
    remain_safe_with_id = group_safe_with_id.drop(train_safe_with_id.index)
    max_test_size = min(len(remain_toxic_no_id), len(remain_safe_with_id))
    test_toxic_no_id = remain_toxic_no_id.sample(n=max_test_size, random_state=42)
    test_safe_with_id = remain_safe_with_id.sample(n=max_test_size, random_state=42)
    val_df = pd.concat([test_toxic_no_id, test_safe_with_id]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    env1_indices = train_df[train_df["identity_sum"] > 0].index.tolist()
    env2_indices = train_df[train_df["identity_sum"] == 0].index.tolist()
    print(
        f"Environment ready: E1={len(env1_indices)} E2={len(env2_indices)} "
        f"reversed_test={len(val_df)}",
        flush=True,
    )
    return train_df, val_df, env1_indices, env2_indices


class BIRMLightweightHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mu_shared = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.logvar_shared = nn.Parameter(torch.full((2, hidden_size), -4.0))
        self.bias_shared = nn.Parameter(torch.zeros(2))
        self.mu_e1 = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.bias_e1 = nn.Parameter(torch.zeros(2))
        self.mu_e2 = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.bias_e2 = nn.Parameter(torch.zeros(2))

    def _sample_weight(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-8.0, max=0.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, features, env_id):
        features = self.norm(features)
        logits_shared = F.linear(
            features, self._sample_weight(self.mu_shared, self.logvar_shared), self.bias_shared
        )
        if env_id == 1:
            logits_env = F.linear(features, self.mu_e1, self.bias_e1)
        else:
            logits_env = F.linear(features, self.mu_e2, self.bias_e2)
        return logits_shared, logits_env

    def predict_logits(self, features):
        return F.linear(self.norm(features), self.mu_shared, self.bias_shared)

    def kl_divergence(self):
        logvar = torch.clamp(self.logvar_shared, min=-8.0, max=0.0)
        return -0.5 * torch.sum(1.0 + logvar - self.mu_shared.pow(2) - logvar.exp())


def get_trainable_state(module):
    return {
        name: param.detach().float().cpu().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(module, state, device):
    params = dict(module.named_parameters())
    for name, value in state.items():
        if name in params:
            params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def update_ema_state(ema_state, module, decay):
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        value = param.detach().float().cpu()
        if name not in ema_state:
            ema_state[name] = value.clone()
        else:
            ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)


def make_collate(tokenizer):
    def collate(batch):
        texts = [item["text"] if isinstance(item, dict) else item for item in batch]
        labels = [item["toxicity"] if isinstance(item, dict) else 0 for item in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    return collate


def extract_last_token_hidden(outputs, attention_mask, labels_len):
    return outputs.hidden_states[-1][
        torch.arange(labels_len, device=attention_mask.device),
        attention_mask.sum(dim=1) - 1,
        :,
    ].to(torch.float32)


def evaluate(model, head, val_loader, device):
    model.eval()
    head.eval()
    correct, total, pred_ones = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            hidden = extract_last_token_hidden(out, batch["attention_mask"], len(batch["labels"]))
            logits = head.predict_logits(hidden)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            pred_ones += (preds == 1).sum().item()
            total += len(batch["labels"])
            del out, hidden, logits
    return correct / max(total, 1), pred_ones, total


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": RUN_NAME,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "batch": BATCH,
        "val_batch": VAL_BATCH,
        "max_length": MAX_LENGTH,
        "head_lr": HEAD_LR,
        "lora_lr": LORA_LR,
        "weight_decay": WEIGHT_DECAY,
        "eta_min": ETA_MIN,
        "ema_decay": EMA_DECAY,
        "kl_weight": KL_WEIGHT,
        "max_snapshot_penalty": MAX_SNAPSHOT_PENALTY,
        "penalty_warmup_ratio": PENALTY_WARMUP_RATIO,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "last_n_layers": LAST_N_LAYERS,
        "data_parquet": str(DATA_PARQUET),
        "out_dir": str(OUT_DIR),
        "model_cache_dir": str(MODEL_CACHE_DIR),
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")

    torch.cuda.empty_cache()
    gc.collect()
    train_df, val_df, env1_indices, env2_indices = prepare_data()

    print("[2/5] Loading Qwen2.5-3B-Instruct...", flush=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download("qwen/Qwen2.5-3B-Instruct", cache_dir=str(MODEL_CACHE_DIR))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = make_collate(tokenizer)
    train_records = train_df.to_dict("records")
    val_records = val_df.to_dict("records")
    loader_e1 = DataLoader(
        Subset(train_records, env1_indices),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    loader_e2 = DataLoader(
        Subset(train_records, env2_indices),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_records, batch_size=VAL_BATCH, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable()

    print("[3/5] Attaching LoRA to last layers q_proj/v_proj...", flush=True)
    target_modules = []
    for layer_idx in range(base_model.config.num_hidden_layers - LAST_N_LAYERS, base_model.config.num_hidden_layers):
        target_modules.extend(
            [
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
            ]
        )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
    )
    model = get_peft_model(base_model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    print("[4/5] Starting LoRA-BIRM stable run...", flush=True)
    hidden_size = model.config.hidden_size
    head = BIRMLightweightHead(hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": LORA_LR,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_STEPS, eta_min=ETA_MIN
    )

    penalty_warmup_steps = max(1, int(MAX_STEPS * PENALTY_WARMUP_RATIO))
    ema_head_state = get_trainable_state(head)
    ema_model_state = get_trainable_state(model)
    best_acc = 0.0
    best_step = -1
    best_tag = "none"
    best_pred_ones = 0
    best_total = 0
    best_head_state = None
    best_model_state = None
    train_correct, train_total, running_loss = 0, 0, 0.0
    start_time = time.time()
    step = 0

    model.train()
    head.train()
    optimizer.zero_grad()
    while step < MAX_STEPS:
        for b1, b2 in zip(loader_e1, loader_e2):
            if step >= MAX_STEPS:
                break
            b1 = {k: v.to(device) for k, v in b1.items()}
            b2 = {k: v.to(device) for k, v in b2.items()}
            penalty_weight = min(
                MAX_SNAPSHOT_PENALTY,
                MAX_SNAPSHOT_PENALTY * step / penalty_warmup_steps,
            )

            out_e1 = model(
                input_ids=b1["input_ids"],
                attention_mask=b1["attention_mask"],
                output_hidden_states=True,
            )
            h_e1 = extract_last_token_hidden(out_e1, b1["attention_mask"], len(b1["labels"]))
            out_e2 = model(
                input_ids=b2["input_ids"],
                attention_mask=b2["attention_mask"],
                output_hidden_states=True,
            )
            h_e2 = extract_last_token_hidden(out_e2, b2["attention_mask"], len(b2["labels"]))

            logits_s_e1, logits_env_e1 = head(h_e1, 1)
            logits_s_e2, logits_env_e2 = head(h_e2, 2)
            loss_s_e1 = F.cross_entropy(logits_s_e1, b1["labels"])
            loss_env_e1 = F.cross_entropy(logits_env_e1, b1["labels"])
            loss_s_e2 = F.cross_entropy(logits_s_e2, b2["labels"])
            loss_env_e2 = F.cross_entropy(logits_env_e2, b2["labels"])
            env_losses = torch.stack([loss_s_e1, loss_s_e2])
            variance_penalty = env_losses.var(unbiased=False)
            alignment_penalty = 0.5 * (
                torch.abs(loss_s_e1 - loss_env_e1) + torch.abs(loss_s_e2 - loss_env_e2)
            )
            kl_loss = head.kl_divergence() / float(hidden_size)
            env_nll = 0.25 * (loss_s_e1 + loss_s_e2 + loss_env_e1 + loss_env_e2)
            penalty = variance_penalty + alignment_penalty
            final_loss = env_nll + penalty_weight * penalty + KL_WEIGHT * kl_loss

            preds_e1 = torch.argmax(logits_s_e1, dim=1)
            preds_e2 = torch.argmax(logits_s_e2, dim=1)
            train_correct += (preds_e1 == b1["labels"]).sum().item()
            train_correct += (preds_e2 == b2["labels"]).sum().item()
            train_total += len(b1["labels"]) + len(b2["labels"])
            running_loss += final_loss.item()

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_ema_state(ema_head_state, head, EMA_DECAY)
            update_ema_state(ema_model_state, model, EMA_DECAY)

            del out_e1, out_e2, h_e1, h_e2, final_loss
            torch.cuda.empty_cache()

            if step > 0 and step % EVAL_INTERVAL == 0:
                current_head_state = get_trainable_state(head)
                current_model_state = get_trainable_state(model)
                current_acc, current_pred_ones, current_total = evaluate(model, head, val_loader, device)

                load_trainable_state(head, ema_head_state, device)
                load_trainable_state(model, ema_model_state, device)
                ema_acc, ema_pred_ones, ema_total = evaluate(model, head, val_loader, device)
                load_trainable_state(head, current_head_state, device)
                load_trainable_state(model, current_model_state, device)

                if ema_acc >= current_acc:
                    selected_acc, selected_pred_ones, selected_total, tag = (
                        ema_acc,
                        ema_pred_ones,
                        ema_total,
                        "ema",
                    )
                    selected_head_state = {k: v.clone() for k, v in ema_head_state.items()}
                    selected_model_state = {k: v.clone() for k, v in ema_model_state.items()}
                else:
                    selected_acc, selected_pred_ones, selected_total, tag = (
                        current_acc,
                        current_pred_ones,
                        current_total,
                        "current",
                    )
                    selected_head_state = {k: v.clone() for k, v in current_head_state.items()}
                    selected_model_state = {k: v.clone() for k, v in current_model_state.items()}

                is_best = selected_acc > best_acc
                if is_best:
                    best_acc = selected_acc
                    best_step = step
                    best_tag = tag
                    best_pred_ones = selected_pred_ones
                    best_total = selected_total
                    best_head_state = selected_head_state
                    best_model_state = selected_model_state
                    torch.save(best_head_state, BEST_HEAD_PATH)
                    load_trainable_state(head, best_head_state, device)
                    load_trainable_state(model, best_model_state, device)
                    model.save_pretrained(BEST_LORA_DIR)
                    load_trainable_state(head, current_head_state, device)
                    load_trainable_state(model, current_model_state, device)

                train_acc = train_correct / max(train_total, 1)
                avg_loss = running_loss / EVAL_INTERVAL
                elapsed_min = (time.time() - start_time) / 60.0
                row = {
                    "step": step,
                    "elapsed_min": elapsed_min,
                    "penalty_weight": penalty_weight,
                    "train_acc": train_acc,
                    "avg_loss": avg_loss,
                    "current_acc": current_acc,
                    "current_pred_ones": current_pred_ones,
                    "ema_acc": ema_acc,
                    "ema_pred_ones": ema_pred_ones,
                    "selected_acc": selected_acc,
                    "selected_pred_ones": selected_pred_ones,
                    "selected_tag": tag,
                    "is_best": is_best,
                    "best_acc": best_acc,
                    "best_step": best_step,
                    "best_tag": best_tag,
                    "total": selected_total,
                }
                append_csv(STEP_HISTORY_CSV, row)
                print(
                    "RESULT "
                    f"step={step} selected={selected_acc * 100:.2f}% tag={tag} "
                    f"current={current_acc * 100:.2f}% ema={ema_acc * 100:.2f}% "
                    f"best={best_acc * 100:.2f}%@{best_step}/{best_tag} "
                    f"pred_ones={selected_pred_ones}/{selected_total} "
                    f"train={train_acc * 100:.2f}% loss={avg_loss:.4f} "
                    f"penalty={penalty_weight:.2f} elapsed={elapsed_min:.2f}m",
                    flush=True,
                )
                train_correct, train_total, running_loss = 0, 0, 0.0
                model.train()
                head.train()

            step += 1

    print("[5/5] Final evaluation with best snapshot...", flush=True)
    if best_head_state is not None:
        load_trainable_state(head, best_head_state, device)
        load_trainable_state(model, best_model_state, device)
    final_acc, final_pred_ones, final_total = evaluate(model, head, val_loader, device)
    runtime_min = (time.time() - start_time) / 60.0
    summary = {
        "method": "LoRA-BIRM Snapshot Stable",
        "runtime_min": runtime_min,
        "final_best_reloaded_acc": final_acc,
        "final_pred_ones": final_pred_ones,
        "final_total": final_total,
        "best_acc": best_acc,
        "best_step": best_step,
        "best_tag": best_tag,
        "best_pred_ones": best_pred_ones,
        "best_total": best_total,
        **config,
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print(
        f"FINAL best_reloaded={final_acc * 100:.2f}% "
        f"best_recorded={best_acc * 100:.2f}%@{best_step}/{best_tag} "
        f"pred_ones={final_pred_ones}/{final_total} runtime={runtime_min:.2f}m",
        flush=True,
    )
    print(f"Saved step history: {STEP_HISTORY_CSV}", flush=True)
    print(f"Saved summary: {SUMMARY_CSV}", flush=True)
    print(f"Saved best LoRA adapter: {BEST_LORA_DIR}", flush=True)
    print(f"Saved best head: {BEST_HEAD_PATH}", flush=True)


if __name__ == "__main__":
    main()
