import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from collections import OrderedDict
from collections import defaultdict, deque

from module.wordRep import WordRep
from bestModel.jointModel import JointModel as jointModel
from dataloader import MyDataSet, collate_fn

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)             # CPU
    torch.cuda.manual_seed(seed)        # 当前 GPU
    torch.cuda.manual_seed_all(seed)    # 所有 GPU
    # 对 cuDNN 做一些额外控制以保证确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ================= 参数配置 =================
class Args:
    # 🔴 请确保这些路径与你的环境一致
    model_name_or_path = "/root/autodl-tmp/model/bert-base-uncased" 
    max_seq_length = 128
    hidden_dim_ffw = 256
    dropout_rate = 0.3
    batch_size = 16
    num_epochs = 15
    lr = 3e-5
    save_dir = "/root/autodl-tmp/best_code/checkpoints/MixSNIPS"
    intent_label_file = "/root/autodl-tmp/datasets/MixSNIPS_clean/intent_label.txt"
    slot_label_file   = "/root/autodl-tmp/datasets/MixSNIPS_clean/slot_label.txt"
    warmup_ratio = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_intent_num = 3

args = Args()

# 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)
device = torch.device(args.device)
print(f"🚀 Using device: {device}")

# ================= 工具函数 =================
def load_labels_from_file(file_path):
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    return labels

def intent_accuracy(preds, golds):
    match = (preds == golds).all(dim=1)
    return match.float().mean().item()

def slot_metrics(all_slot_preds, all_slot_golds):
    """计算 Token/Span 级别的 P, R, F1"""
    global_tp = 0
    global_fp = 0
    global_fn = 0

    for pred_triples, gold_triples in zip(all_slot_preds, all_slot_golds):
        pred_set = set(tuple(t) for t in pred_triples)
        gold_set = set(tuple(t) for t in gold_triples)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        global_tp += tp
        global_fp += fp
        global_fn += fn

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def decode_span_logits(start_logits, end_logits, words_lengths, id2label):
    
    start_preds = torch.argmax(start_logits, dim=-1) # [B, L]
    end_preds = torch.argmax(end_logits, dim=-1)     # [B, L]
    
    batch_triples = []
    batch_size, seq_len = start_preds.shape
    word_attention_mask = (words_lengths > 0).long()
    for b in range(batch_size):
        triples = []
        current_starts = {} 
        
        # 注意：这里的 seq_len 是 word 级别的长度，mask 也是
        for i in range(seq_len):
            if word_attention_mask[b, i] == 0: continue 
            
            s_label = start_preds[b, i].item()
            e_label = end_preds[b, i].item()
            
            if s_label != 0: 
                current_starts[s_label] = i 
            
            if e_label != 0:
                if e_label in current_starts:
                    start_idx = current_starts[e_label]
                    if e_label in id2label:
                        triples.append((start_idx, i, id2label[e_label]))
                    del current_starts[e_label] 
        
        batch_triples.append(triples)
    return batch_triples


def semantic_accuracy(all_intent_preds, all_intent_golds, all_slot_preds, all_slot_golds):
    total = len(all_intent_preds)
    joint_correct = 0
    for i in range(total):
        intent_correct = (all_intent_preds[i] == all_intent_golds[i]).all().item()
        slot_pred_set = set(tuple(t) for t in all_slot_preds[i])
        slot_gold_set = set(tuple(t) for t in all_slot_golds[i])
        slot_correct = (slot_pred_set == slot_gold_set)
        
        if intent_correct and slot_correct:
            joint_correct += 1
    return joint_correct / total if total > 0 else 0.0

def type_margin_loss(logits, gold_ids, mask, o_id=0, margin=1.0):
    """
    logits: [B, L, C]
    gold_ids: [B, L]  (start_ids 或 end_ids)
    mask: [B, L]      (1 表示有效位置)
    o_id: O 类的 id（你现在是 0）
    margin: 希望 gold_logit 比 max_wrong_logit 至少大多少
    """
    B, L, C = logits.shape
    mask = mask.float()

    # 只对实体位置（gold != O）做惩罚
    ent_mask = mask * (gold_ids != o_id).float()
    denom = ent_mask.sum().clamp_min(1.0)

    # gold logit: [B, L]
    gold_logit = logits.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)

    # max wrong logit: [B, L]
    # 把 gold 类屏蔽掉，然后取最大
    wrong_logits = logits.clone()
    wrong_logits.scatter_(-1, gold_ids.unsqueeze(-1), -1e9)
    max_wrong = wrong_logits.max(dim=-1).values

    # hinge: max(0, margin - (gold - wrong))
    loss = F.relu(margin - (gold_logit - max_wrong))

    return (loss * ent_mask).sum() / denom

# ================= 验证函数 =================
def evaluate(model, dev_loader, id2label, device):
    model.eval()
    
    all_intent_preds, all_intent_golds = [], []
    all_slot_preds, all_slot_golds = [], []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids, attention_mask, words_lengths, word_attention_mask, \
            intent_labels, start_ids, end_ids, count_labels, sentences = \
                [x.to(device) if torch.is_tensor(x) else x for x in batch]
            
            soft_intent_logits, hard_intent_logits, start_logits, end_logits, count_logits = model(
                input_ids, attention_mask, word_attention_mask, words_lengths
            )

            intent_probs = torch.sigmoid(hard_intent_logits)
            B, num_intents = intent_probs.shape
            count_probs  = torch.softmax(count_logits, dim=-1)
            k_hat = torch.argmax(count_probs, dim=-1) + 1  # [B] 

            intent_pred = torch.zeros(
                B,
                num_intents,
                device=intent_probs.device,
                dtype=torch.long
            )

            for b in range(B):
                k = k_hat[b].item()
                topk = torch.topk(intent_probs[b], k)
                intent_pred[b, topk.indices] = 1

            all_intent_preds.append(intent_pred.cpu())
            all_intent_golds.append(intent_labels.cpu())
            
            # 槽位预测
            batch_slot_preds = decode_span_logits(start_logits, end_logits, words_lengths, id2label)

            all_slot_preds.extend(batch_slot_preds)
            
            # 真实槽位还原
            gold_start_logits = torch.nn.functional.one_hot(start_ids, num_classes=len(id2label)).float()
            gold_end_logits = torch.nn.functional.one_hot(end_ids, num_classes=len(id2label)).float()
            batch_slot_golds = decode_span_logits(gold_start_logits, gold_end_logits, words_lengths, id2label)
            all_slot_golds.extend(batch_slot_golds)

    all_intent_preds = torch.cat(all_intent_preds, dim=0)
    all_intent_golds = torch.cat(all_intent_golds, dim=0)

    # 1. Intent Accuracy
    intent_acc = intent_accuracy(all_intent_preds, all_intent_golds)
    
    # 2. Slot F1 (Token/Span Level)
    slot_p, slot_r, slot_f1 = slot_metrics(all_slot_preds, all_slot_golds)
    
    # 3. Slot Accuracy (Sentence Level) - 这就是你之前掉分的地方
    slot_correct_count = 0
    total_samples = len(all_slot_preds)
    for preds, golds in zip(all_slot_preds, all_slot_golds):
        if set(tuple(t) for t in preds) == set(tuple(t) for t in golds):
            slot_correct_count += 1
    slot_acc = slot_correct_count / total_samples if total_samples > 0 else 0.0

    slot_intent = (slot_f1 + intent_acc) / 2
    # 4. Semantic Accuracy
    semantic_acc = semantic_accuracy(all_intent_preds, all_intent_golds, all_slot_preds, all_slot_golds)
    
    return {
        "intent_acc": intent_acc,
        "slot_f1": slot_f1,
        "slot_acc": slot_acc,
        "slot_p": slot_p,
        "slot_r": slot_r,
        "slot_intent": slot_intent,
        "semantic_acc": semantic_acc
    }

# ================= 训练主循环 =================
def train_model(args):
    # 1. 标签加载
    slot_label_list = ["O"] + load_labels_from_file(args.slot_label_file)
    intent_label_list = load_labels_from_file(args.intent_label_file)
    
    id2label = {i: label for i, label in enumerate(slot_label_list)}
    num_slot_labels = len(slot_label_list)
    num_intent_labels = len(intent_label_list)
    
    print(f"标签加载完毕: Slot {num_slot_labels} 个 (含O), Intent {num_intent_labels} 个")

    # 2. 数据加载
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
    # train_set = MyDataSet(args, "/root/autodl-tmp/datasets/MixATIS_clean/train_explained_new.jsonl",
    #                       intent_label_list, slot_label_list, tokenizer)
    # dev_set = MyDataSet(args, "/root/autodl-tmp/datasets/MixATIS_clean/dev_explaind.jsonl",
    #                     intent_label_list, slot_label_list, tokenizer)
    # test_set = MyDataSet(args, "/root/autodl-tmp/datasets/MixATIS_clean/test_explaind.jsonl",
    #                      intent_label_list, slot_label_list, tokenizer)
    train_set = MyDataSet(args, "/root/autodl-tmp/datasets/MixSNIPS_clean/train_explaind.jsonl", 
                            intent_label_list, slot_label_list, tokenizer)
    dev_set = MyDataSet(args, "/root/autodl-tmp/datasets/MixSNIPS_clean/dev_explaind.jsonl",
                          intent_label_list, slot_label_list, tokenizer)
    test_set = MyDataSet(args, "/root/autodl-tmp/datasets/MixSNIPS_clean/test_explaind.jsonl",
                           intent_label_list, slot_label_list, tokenizer)

    collate_func = lambda x: collate_fn(x, pad_id=0, max_seq_len=args.max_seq_length)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_func, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_func, num_workers=4, pin_memory=True)
    # 3. 模型初始化
    model = jointModel(args, num_intent_labels, num_slot_labels).to(device)
    
    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(len(train_loader)*args.num_epochs*args.warmup_ratio),
                                                num_training_steps=len(train_loader)*args.num_epochs)

    # 5. Loss 函数
    intent_criterion = nn.BCEWithLogitsLoss()
    count_criterion = nn.CrossEntropyLoss()
    
    slot_weights = torch.ones(num_slot_labels).to(device)
    slot_weights[0] = 0.9
    
    slot_criterion = nn.CrossEntropyLoss(
        weight=slot_weights,
        reduction='none',
        label_smoothing=0
    )

    dev_history = []  
    ckpt_dir = args.save_dir
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in loop:
            input_ids, attention_mask, words_lengths, word_attention_mask, \
            intent_tags, start_ids, end_ids, count_labels, sentences = \
                [x.to(device) if torch.is_tensor(x) else x for x in batch]

            optimizer.zero_grad()
            
            soft_intent_logits, hard_intent_logits, start_logits, end_logits, count_logits = model(
                input_ids, attention_mask, word_attention_mask, words_lengths,
                start_positions=start_ids
            )
            # --- Loss 计算 (核心修正部分) ---
            # 1. 意图 Loss
            loss_intent_soft = intent_criterion(soft_intent_logits, intent_tags)
            loss_intent_hard = intent_criterion(hard_intent_logits, intent_tags)
           
            # 1. teacher（稳定）
            soft_probs = F.softmax(soft_intent_logits.detach() / 2.0, dim=-1)
            # 2. student
            hard_log_probs = F.log_softmax(hard_intent_logits / 2.0, dim=-1)
            constraint_loss = F.kl_div(hard_log_probs, soft_probs, reduction='batchmean')


            prob = torch.sigmoid(hard_intent_logits) 
            pred_count = prob.sum(dim=1) 
            target_count = count_labels.float() 

            loss_cardinality = F.mse_loss(pred_count, target_count)
            loss_intent_count = count_criterion(count_logits, count_labels)

            # MixATIS
            # if epoch < 5:
            #     loss_intent = 0.1 * loss_intent_soft + 0.9 * loss_intent_hard + 0.3 * loss_cardinality + 0.3 * loss_intent_count
            # else:
            #     loss_intent = 1.0 * loss_intent_hard + 0.3 * loss_cardinality + 0.3 * loss_intent_count

            #MixSNIPS（cardinality不需要 因为MixSNIPS意图数量少只有七个 且容易区分）
            if epoch <= 2:
                loss_intent = 0.25 * loss_intent_soft + 1.0 * loss_intent_hard + 0.35 * loss_intent_count + 0.01 * loss_cardinality 
            else:
                loss_intent = 1.2 * loss_intent_hard + 0.1 * loss_intent_soft + 0.3 * loss_intent_count + 0.01 * loss_cardinality

            # 2. 槽位 Loss
            active_loss = words_lengths.view(-1) > 0
            active_start_logits = start_logits.view(-1, num_slot_labels)[active_loss]
            active_end_logits = end_logits.view(-1, num_slot_labels)[active_loss]
            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]
            
            loss_start = slot_criterion(active_start_logits, active_start_labels)
            loss_end = slot_criterion(active_end_logits, active_end_labels)

            # MixATIS & MixSNIPS
            # if epoch <= 2:
            #     loss_start_end = 0.82 * loss_start.mean() + 0.18 * loss_end.mean()
            # else:
            #     loss_start_end = 0.78 * loss_start.mean() + 0.12 * loss_end.mean()
            if epoch <= 2:
                loss_start_end = 0.85 * loss_start.mean() + 0.15 * loss_end.mean()
            else:
                loss_start_end = 0.8 * loss_start.mean() + 0.2 * loss_end.mean()
    
            L = start_logits.size(1)
            word_mask = (words_lengths[:, :L] > 0).float()                 # [B, L]
            pos = torch.arange(L, device=start_logits.device).float().unsqueeze(0)  # [1, L]
            p_s = torch.softmax(start_logits, dim=-1)                      # [B, L, C]
            p_e = torch.softmax(end_logits,   dim=-1)
            s = p_s.max(dim=-1)[0] * word_mask                             # [B, L]
            e = p_e.max(dim=-1)[0] * word_mask
            s = s.detach()
            s = s / (s.sum(dim=1, keepdim=True) + 1e-9)
            e = e / (e.sum(dim=1, keepdim=True) + 1e-9)
            exp_start = (s * pos).sum(dim=1)
            exp_end   = (e * pos).sum(dim=1)
            loss_span = torch.relu(exp_start - exp_end).mean()

            # ===== 新增：类型错加大惩罚（只对 gold!=O 生效）=====
            lambda_type = 0.7   # 先用 0.5~2.0 试
            margin = 1.0        # 0.5~2.0 试
            loss_type = type_margin_loss(start_logits, start_ids, word_mask, o_id=0, margin=margin) \
                    + type_margin_loss(end_logits, end_ids, word_mask, o_id=0, margin=margin)

            # MixATIS 1+0.02
            # loss_slot = loss_start_end + 0.02 * loss_span
            # MixSNIPS 修改  lambda_type * loss_type
            if epoch<5:
                loss_slot = loss_start_end + 0.02 * loss_span
            else:
                loss_slot = loss_start_end + 0.02 * loss_span +  lambda_type * loss_type
            
            
            # MixATIS 1+1.5
            # if epoch < 5:
            #     loss = 1.0 * loss_intent + 1.5 * loss_slot + 0.03 * constraint_loss
            # else:
            #     loss = 1.0 * loss_intent + 1.5 * loss_slot 
            #MixSNIPS
            if epoch < 5:
                loss = 1.0 * loss_intent + 1.5 * loss_slot 
            else:
                loss = 1.0 * loss_intent + 1.0 * loss_slot + 0.15 * constraint_loss
    
     
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        # --- 验证 ---
        metrics_dev = evaluate(model, dev_loader, id2label, device)
        dev_semantic = metrics_dev['semantic_acc']
        print(f"dev Epoch {epoch} | Intent: {metrics_dev['intent_acc']:.4f} | Slot Acc: {metrics_dev['slot_acc']:.4f} | Slot F1: {metrics_dev['slot_f1']:.4f} | slot_intent: {metrics_dev['slot_intent']:.4f} | Semantic: {metrics_dev['semantic_acc']:.4f}")
        metrics_test = evaluate(model, test_loader, id2label, device)
        print(f"test Epoch {epoch} | Intent: {metrics_test['intent_acc']:.4f} | Slot Acc: {metrics_test['slot_acc']:.4f} | Slot F1: {metrics_test['slot_f1']:.4f} | slot_intent: {metrics_test['slot_intent']:.4f} | Semantic: {metrics_test['semantic_acc']:.4f}")
        if epoch > 3:
            dev_history.append({
                "epoch": epoch,
                "slot_intent": metrics_dev["slot_intent"],
                "semantic": dev_semantic
            })

        # 保存当前 epoch checkpoint（用于 averaging）
        if epoch > 3:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "dev_semantic": dev_semantic
            }, ckpt_path)
    # ================= Averaging based on dev =================
    print("\n🔁 Start checkpoint averaging based on dev performance")

    # 选 dev slot_intent top-3
    topk = sorted(dev_history, key=lambda x: x["slot_intent"], reverse=True)[:3]
    avg_epochs = [item["epoch"] for item in topk]

    print("Averaging epochs:", avg_epochs)

    ckpt_paths = [
        os.path.join(ckpt_dir, f"ckpt_epoch_{e}.pt") for e in avg_epochs
    ]

    avg_state_dict = average_checkpoints(ckpt_paths, device=device)

    # 加载 averaged 参数
    model.load_state_dict(avg_state_dict)
    model.eval()

    final_metrics = evaluate(model, test_loader, id2label, device)
    print("\n🎯 Final Test Results (Averaged Model)")
    print(f"Intent:   {final_metrics['intent_acc']:.4f}")
    print(f"Slot Acc: {final_metrics['slot_acc']:.4f}")
    print(f"Slot F1:  {final_metrics['slot_f1']:.4f}")
    print(f"slot_intent: {final_metrics['slot_intent']:.4f}")
    print(f"Semantic: {final_metrics['semantic_acc']:.4f}")

def average_checkpoints(ckpt_paths, device="cpu"):
    avg_state = OrderedDict()
    n = len(ckpt_paths)

    for i, path in enumerate(ckpt_paths):
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt["model_state"]

        for k, v in state_dict.items():
            if i == 0:
                avg_state[k] = v.clone().float()
            else:
                avg_state[k] += v.float()

    for k in avg_state:
        avg_state[k] /= n

    return avg_state


if __name__ == "__main__":
    set_seed(8887)
    train_model(args)