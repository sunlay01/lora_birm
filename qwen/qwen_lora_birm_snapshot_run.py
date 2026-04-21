import os
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 0. 物理隔绝与显存清理
# ==========================================
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HF_ENDPOINT', None)
torch.cuda.empty_cache()
gc.collect()

# ==========================================
# 1. 离线读取与极端虚假捷径注入 (对抗环境)
# ==========================================
print("📂 [1/5] 正在读取数据并构建【极端虚假捷径环境】...")
file_path = "train-00000-of-00001.parquet" 
df = pd.read_parquet(file_path)

if 'comment_text' in df.columns: df.rename(columns={'comment_text': 'text'}, inplace=True)
if 'target' in df.columns: df.rename(columns={'target': 'toxicity'}, inplace=True)

identity_cols = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'black', 'white']
df['toxicity'] = (df['toxicity'] >= 0.5).astype(int)
df['identity_sum'] = df[[c for c in identity_cols if c in df.columns]].sum(axis=1)

group_toxic_with_id = df[(df['toxicity'] == 1) & (df['identity_sum'] > 0)] 
group_toxic_no_id   = df[(df['toxicity'] == 1) & (df['identity_sum'] == 0)]
group_safe_with_id  = df[(df['toxicity'] == 0) & (df['identity_sum'] > 0)] 
group_safe_no_id    = df[(df['toxicity'] == 0) & (df['identity_sum'] == 0)]

# --- 🎯 构建训练集 (制造 90% 的虚假捷径) ---
n_toxic_train = 10000
train_toxic_with_id = group_toxic_with_id.sample(n=int(n_toxic_train * 0.9), random_state=42)
train_toxic_no_id   = group_toxic_no_id.sample(n=int(n_toxic_train * 0.1), random_state=42)

n_safe_train = 10000
train_safe_with_id = group_safe_with_id.sample(n=int(n_safe_train * 0.1), random_state=42)
train_safe_no_id   = group_safe_no_id.sample(n=int(n_safe_train * 0.9), random_state=42)

train_df = pd.concat([train_toxic_with_id, train_toxic_no_id, train_safe_with_id, train_safe_no_id]).sample(frac=1).reset_index(drop=True)

# --- 🎯 构建测试集 (纯反转，打破捷径) ---
remain_toxic_no_id   = group_toxic_no_id.drop(train_toxic_no_id.index)
remain_safe_with_id  = group_safe_with_id.drop(train_safe_with_id.index)

max_test_size = min(len(remain_toxic_no_id), len(remain_safe_with_id)) 
test_toxic_no_id = remain_toxic_no_id.sample(n=max_test_size, random_state=42)
test_safe_with_id = remain_safe_with_id.sample(n=max_test_size, random_state=42)

val_df = pd.concat([test_toxic_no_id, test_safe_with_id]).sample(frac=1).reset_index(drop=True)

env1_indices = train_df[train_df['identity_sum'] > 0].index.tolist()
env2_indices = train_df[train_df['identity_sum'] == 0].index.tolist()

print(f"🔥 环境构建完毕！训练集 E1: {len(env1_indices)} | E2: {len(env2_indices)} | 纯反转测试集: {len(val_df)}")

# ==========================================
# 2. 极速加载模型与降存 Tokenizer (Batch=4)
# ==========================================
print("🧠 [2/5] 正在极速加载 Qwen 底座模型...")
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

model_dir = snapshot_download('qwen/Qwen2.5-3B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

def custom_collate_fn(batch):
    texts = [item['text'] if isinstance(item, dict) else item for item in batch]
    labels = [item['toxicity'] if isinstance(item, dict) else 0 for item in batch]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels, dtype=torch.long)
    return inputs

BATCH = 8
loader_e1 = DataLoader(Subset(train_df.to_dict('records'), env1_indices), batch_size=BATCH, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
loader_e2 = DataLoader(Subset(train_df.to_dict('records'), env2_indices), batch_size=BATCH, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_df.to_dict('records'), batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, output_hidden_states=True)
base_model.config.use_cache = False  
base_model.enable_input_require_grads() 
base_model.gradient_checkpointing_enable()

# ==========================================
# 3. 定义分类头与核心算法 (LoRA-BIRM Snapshot)
# ==========================================
def compute_irmv1_penalty(logits, y):
    scale = torch.tensor(1., requires_grad=True).to(device)
    loss = F.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

class StandardHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size) # 降压神药
        self.classifier = nn.Linear(hidden_size, 2) # 自带 Bias
        
    def forward(self, x): 
        x = self.norm(x)
        return self.classifier(x)

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
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(0.5 * logvar)

    def forward(self, features, env_id):
        features = self.norm(features)
        w_shared = self._sample_weight(self.mu_shared, self.logvar_shared)
        logits_shared = F.linear(features, w_shared, self.bias_shared)
        
        if env_id == 1:
            logits_env = F.linear(features, self.mu_e1, self.bias_e1)
        else:
            logits_env = F.linear(features, self.mu_e2, self.bias_e2)
            
        return logits_shared, logits_env

    def predict_logits(self, features):
        features = self.norm(features)
        return F.linear(features, self.mu_shared, self.bias_shared)

    def kl_divergence(self):
        logvar = torch.clamp(self.logvar_shared, min=-8.0, max=0.0)
        return -0.5 * torch.sum(1.0 + logvar - self.mu_shared.pow(2) - logvar.exp())

def get_trainable_state(module):
    return {
        name: param.detach().float().cpu().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }

def load_trainable_state(module, state):
    params = dict(module.named_parameters())
    for name, value in state.items():
        if name in params:
            params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))

def update_ema_state(ema_state, module, decay=0.98):
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        value = param.detach().float().cpu()
        if name not in ema_state:
            ema_state[name] = value.clone()
        else:
            ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)

def evaluate_classifier(model, head, method_name, max_batches=None):
    model.eval()
    head.eval()
    correct, total, pred_ones = 0, 0, 0
    with torch.no_grad():
        for i, val_b in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            val_b = {k: v.to(device) for k, v in val_b.items()}
            out_val = model(input_ids=val_b['input_ids'], attention_mask=val_b['attention_mask'], output_hidden_states=True)
            h_val = out_val.hidden_states[-1][torch.arange(len(val_b['labels'])), val_b['attention_mask'].sum(dim=1)-1, :].to(torch.float32)
            val_logits = head(h_val) if method_name == 'irmv1' else head.predict_logits(h_val)
            preds = torch.argmax(val_logits, dim=1)
            correct += (preds == val_b['labels']).sum().item()
            pred_ones += (preds == 1).sum().item()
            total += len(val_b['labels'])
            del out_val, h_val, val_logits
    return correct / max(total, 1), pred_ones, total

# ==========================================
# 4. LoRA-BIRM Snapshot 训练循环
# ==========================================
def run_experiment(method_name, model, max_steps=1000): 
    print(f"\n{'='*70}\n🚀 开始执行实验: {method_name.upper()}\n{'='*70}")
    
    if method_name in ['irmv1', 'birm']:
        for param in model.parameters(): param.requires_grad = False
            
    hidden_size = model.config.hidden_size
    if method_name == 'irmv1':
        head = StandardHead(hidden_size).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.01)
    else:
        head = BIRMLightweightHead(hidden_size).to(device)
        trainable_params = [{'params': head.parameters(), 'lr': 8e-4, 'weight_decay': 0.01}]
        if method_name == 'lora-birm':
            trainable_params.append({'params': [p for p in model.parameters() if p.requires_grad], 'lr': 8e-5, 'weight_decay': 0.01})
        optimizer = torch.optim.AdamW(trainable_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    eval_interval = 50
    ema_decay = 0.92
    kl_weight = 1e-4
    max_snapshot_penalty = 6.0
    penalty_warmup_steps = (2 * max_steps) // 3

    start_time = time.time()
    best_val_acc = 0.0
    best_snapshot_tag = "none"
    step_count = 0
    best_head_weights = None
    best_model_weights = None 
    ema_head_state = get_trainable_state(head)
    ema_model_state = get_trainable_state(model) if method_name == 'lora-birm' else None
    
    train_correct_in_period = 0
    train_samples_in_period = 0
    running_loss_sum = 0.0 
    
    model.train(); head.train(); optimizer.zero_grad() 
    
    while step_count < max_steps:
        for b1, b2 in zip(loader_e1, loader_e2):
            if step_count >= max_steps: break
                
            b1 = {k: v.to(device) for k, v in b1.items()}
            b2 = {k: v.to(device) for k, v in b2.items()}
            
            if method_name == 'lora-birm':
                current_penalty_weight = min(max_snapshot_penalty, max_snapshot_penalty * step_count / max(penalty_warmup_steps, 1))
            else:
                current_penalty_weight = min(max_snapshot_penalty, max_snapshot_penalty * step_count / max(penalty_warmup_steps, 1))

            out_e1 = model(input_ids=b1['input_ids'], attention_mask=b1['attention_mask'], output_hidden_states=True)
            h_e1 = out_e1.hidden_states[-1][torch.arange(len(b1['labels'])), b1['attention_mask'].sum(dim=1)-1, :].to(torch.float32)
            
            out_e2 = model(input_ids=b2['input_ids'], attention_mask=b2['attention_mask'], output_hidden_states=True)
            h_e2 = out_e2.hidden_states[-1][torch.arange(len(b2['labels'])), b2['attention_mask'].sum(dim=1)-1, :].to(torch.float32)

            if method_name in ['erm', 'irmv1']:
                logits_e1, logits_e2 = head(h_e1), head(h_e2)
                loss_e1 = F.cross_entropy(logits_e1, b1['labels'])
                loss_e2 = F.cross_entropy(logits_e2, b2['labels'])
                if method_name == 'irmv1':
                    penalty = compute_irmv1_penalty(logits_e1, b1['labels']) + compute_irmv1_penalty(logits_e2, b2['labels'])
                else:
                    penalty = torch.tensor(0.0, device=device)
                final_loss = loss_e1 + loss_e2 + current_penalty_weight * penalty
                preds_e1, preds_e2 = torch.argmax(logits_e1, dim=1), torch.argmax(logits_e2, dim=1)
                
            elif method_name in ['birm', 'lora-birm']:
                logits_s_e1, logits_env_e1 = head(h_e1, 1)
                logits_s_e2, logits_env_e2 = head(h_e2, 2)
                
                loss_s_e1 = F.cross_entropy(logits_s_e1, b1['labels'])
                loss_env_e1 = F.cross_entropy(logits_env_e1, b1['labels'])
                loss_s_e2 = F.cross_entropy(logits_s_e2, b2['labels'])
                loss_env_e2 = F.cross_entropy(logits_env_e2, b2['labels'])
                
                if method_name == 'lora-birm':
                    env_losses = torch.stack([loss_s_e1, loss_s_e2])
                    variance_penalty = env_losses.var(unbiased=False)
                    alignment_penalty = 0.5 * (torch.abs(loss_s_e1 - loss_env_e1) + torch.abs(loss_s_e2 - loss_env_e2))
                    kl_loss = head.kl_divergence() / float(hidden_size)
                    env_nll = 0.25 * (loss_s_e1 + loss_s_e2 + loss_env_e1 + loss_env_e2)
                    penalty = variance_penalty + alignment_penalty
                    final_loss = env_nll + current_penalty_weight * penalty + kl_weight * kl_loss
                else:
                    env_losses = torch.stack([loss_s_e1, loss_s_e2])
                    variance_penalty = env_losses.var(unbiased=False)
                    alignment_penalty = 0.5 * (torch.abs(loss_s_e1 - loss_env_e1) + torch.abs(loss_s_e2 - loss_env_e2))
                    kl_loss = head.kl_divergence() / float(hidden_size)
                    base_loss = 0.25 * (loss_s_e1 + loss_s_e2 + loss_env_e1 + loss_env_e2)
                    penalty = variance_penalty + alignment_penalty
                    final_loss = base_loss + current_penalty_weight * penalty + kl_weight * kl_loss
                
                preds_e1, preds_e2 = torch.argmax(logits_s_e1, dim=1), torch.argmax(logits_s_e2, dim=1)

            train_correct_in_period += (preds_e1 == b1['labels']).sum().item() + (preds_e2 == b2['labels']).sum().item()
            train_samples_in_period += len(b1['labels']) + len(b2['labels'])

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            if method_name == 'lora-birm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update_ema_state(ema_head_state, head, decay=ema_decay)
            if method_name == 'lora-birm':
                update_ema_state(ema_model_state, model, decay=ema_decay)
                
            running_loss_sum += final_loss.item()
            del out_e1, out_e2, h_e1, h_e2, final_loss
            torch.cuda.empty_cache()

            # ---------- LoRA-BIRM Snapshot: 当前模型 + EMA 双候选快照 ----------
            if step_count > 0 and step_count % eval_interval == 0:
                current_head_state = get_trainable_state(head)
                current_model_state = get_trainable_state(model) if method_name == 'lora-birm' else None

                current_acc, current_pred_ones, current_total = evaluate_classifier(model, head, method_name, max_batches=None)
                val_accuracy, pred_ones, val_total, snapshot_tag = current_acc, current_pred_ones, current_total, "current"

                if method_name == 'lora-birm' and ema_head_state is not None and ema_model_state is not None:
                    load_trainable_state(head, ema_head_state)
                    load_trainable_state(model, ema_model_state)
                    ema_acc, ema_pred_ones, ema_total = evaluate_classifier(model, head, method_name, max_batches=None)
                    load_trainable_state(head, current_head_state)
                    load_trainable_state(model, current_model_state)
                    if ema_acc >= current_acc:
                        val_accuracy, pred_ones, val_total, snapshot_tag = ema_acc, ema_pred_ones, ema_total, "ema"

                train_accuracy = train_correct_in_period / train_samples_in_period
                avg_train_loss = running_loss_sum / eval_interval
                
                print(f"[{method_name.upper()}] Step {step_count}/{max_steps}")
                if method_name == 'lora-birm':
                    stage_str = f"LoRA-BIRM Snapshot | penalty={current_penalty_weight:.2f} | candidate={snapshot_tag}"
                else:
                    stage_str = "BIRM/IRM baseline"
                print(f"  ├─ 阶段: {stage_str} | 🏋️ 训练 Acc: {train_accuracy*100:.2f}% | 平均 Loss: {avg_train_loss:.4f}")
                print(f"  └─ 🎯 测试 Acc: {val_accuracy*100:.2f}% | 诊断预测有毒数: {pred_ones}/{val_total}")

                if val_accuracy > best_val_acc: 
                    best_val_acc = val_accuracy
                    best_snapshot_tag = snapshot_tag
                    if snapshot_tag == "ema" and method_name == 'lora-birm':
                        best_head_weights = {k: v.clone() for k, v in ema_head_state.items()}
                        best_model_weights = {k: v.clone() for k, v in ema_model_state.items()}
                    else:
                        best_head_weights = {k: v.clone() for k, v in current_head_state.items()}
                        best_model_weights = {k: v.clone() for k, v in current_model_state.items()} if current_model_state is not None else None
                    print(f"  🌟 [突破点] 泛化能力创新高！已保存 {snapshot_tag.upper()} 最佳快照！")
                
                train_correct_in_period, train_samples_in_period = 0, 0
                running_loss_sum = 0.0 
                model.train(); head.train()
                torch.cuda.empty_cache()
            
            step_count += 1

    # ---------- 期末统考：重载最佳快照，全量测试 ----------
    print(f"\n⏳ {method_name.upper()} 训练结束。正在重载最佳权重执行【期末全量统考】...")
    
    if best_head_weights is not None:
        load_trainable_state(head, best_head_weights)
    if method_name == 'lora-birm' and best_model_weights is not None:
        load_trainable_state(model, best_model_weights)
        
    final_accuracy, final_pred_ones, final_total = evaluate_classifier(model, head, method_name, max_batches=None)
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"🏆 {method_name.upper()} 期末统考完毕！最佳快照={best_snapshot_tag} | 全量测试准确率: {final_accuracy*100:.2f}% | 预测有毒数: {final_pred_ones}/{final_total} | 耗时: {total_time:.2f} 分钟\n")
    
    del head, optimizer
    torch.cuda.empty_cache(); gc.collect()
    return total_time, final_accuracy


# ==========================================
# 5. 依次执行三大算法 (LoRA-BIRM Snapshot 优先 + 底座重置版)
# ==========================================
MAX_STEPS = 600 
results = {}

print("\n" + "*"*75)
print("⭐ 第一场：优先测试 LoRA-BIRM Snapshot (低秩贝叶斯 + EMA 快照)")
print("*"*75)

n_layers = base_model.config.num_hidden_layers
target_modules = []
for layer_idx in range(n_layers - 6, n_layers):
    target_modules += [
        f"model.layers.{layer_idx}.self_attn.q_proj",
        f"model.layers.{layer_idx}.self_attn.v_proj",
    ]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
)
lora_model = get_peft_model(base_model, peft_config)
lora_model.enable_input_require_grads()
lora_model.print_trainable_parameters()

results['LoRA-BIRM Snapshot (Ours)'] = run_experiment('lora-birm', lora_model, MAX_STEPS)

print("\n🔄 [状态重置] 正在卸载 LoRA 适配器，恢复纯净底座以运行基线模型...")
base_model = lora_model.unload()

print("\n" + "*"*70)
print("⭐ 第二场：运行基线竞争对手 BIRM (完全冻结底座)")
print("*"*70)
results['BIRM'] = run_experiment('birm', base_model, MAX_STEPS)

print("\n" + "*"*70)
print("⭐ 第三场：运行最基础的竞争对手 ERM (完全线性，无IRM惩罚)")
print("*"*70)
results['ERM'] = run_experiment('erm', base_model, MAX_STEPS)

print("\n" + "*"*70)
print("⭐ 第四场：运行最基础的竞争对手 IRMv1 (完全线性)")
print("*"*70)
results['IRMv1'] = run_experiment('irmv1', base_model, MAX_STEPS)


# ==========================================
# 6. 打印最终毕业论文数据汇总表
# ==========================================
print("\n" + "="*80)
print("🏆 毕业论文消融实验结果汇总 (对抗环境 + LoRA-BIRM Snapshot + 全量评测) 🏆")
print("="*80)
print(f"{'模型架构':<28} | {'运行耗时 (分钟)':<15} | {'最终反转测试集准确率':<20}")
print("-" * 80)
print(f"★ {'LoRA-BIRM Snapshot (Ours)':<26} | {results['LoRA-BIRM Snapshot (Ours)'][0]:<20.2f} | {results['LoRA-BIRM Snapshot (Ours)'][1]*100:.2f}%")
print(f"  {'BIRM':<26} | {results['BIRM'][0]:<20.2f} | {results['BIRM'][1]*100:.2f}%")
print(f"  {'ERM':<26} | {results['ERM'][0]:<20.2f} | {results['ERM'][1]*100:.2f}%")
print(f"  {'IRMv1':<26} | {results['IRMv1'][0]:<20.2f} | {results['IRMv1'][1]*100:.2f}%")
print("="*80)
print("✅ 实验完成：建议同时记录最佳快照步数、全量反转测试准确率与运行时间。")
