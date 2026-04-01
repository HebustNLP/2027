# # -*- coding: utf-8 -*-
# import argparse
# import glob
# import logging
# import os
# import json
# import csv
# from datetime import datetime

# import torch
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler

# from callback.optimizater.adamw import AdamW
# from callback.lr_scheduler import get_linear_schedule_with_warmup
# from callback.progressbar import ProgressBar
# from tools.common import seed_everything, json_to_text
# from tools.common import init_logger, logger

# from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
# from models.bert_for_ner import BertSpanForNer
# from models.albert_for_ner import AlbertSpanForNer
# from processors.utils_ner import CNerTokenizer, bert_extract_item
# from processors.ner_span import convert_examples_to_features
# from processors.ner_span import ner_processors as processors
# from processors.ner_span import collate_fn
# from metrics.ner_metrics import SpanEntityScore

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, AlbertConfig)), ())

# MODEL_CLASSES = {
#     # bert ernie bert_wwm bert_wwwm_ext
#     "bert": (BertConfig, BertSpanForNer, CNerTokenizer),
#     "albert": (AlbertConfig, AlbertSpanForNer, CNerTokenizer),
# }

# # =========================
# # ✅ I/O helpers
# # =========================
# def _ensure_dir(path: str):
#     if path is None or path == "":
#         return
#     os.makedirs(path, exist_ok=True)

# def _append_jsonl(path: str, obj: dict):
#     """保留：如果你后续还想写 jsonl 可以继续用（当前改造后不再使用）"""
#     _ensure_dir(os.path.dirname(path))
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# def _to_json_str(obj):
#     try:
#         return json.dumps(obj, ensure_ascii=False)
#     except Exception:
#         return str(obj)

# def _append_csv_row(path: str, row: dict, fieldnames: list):
#     _ensure_dir(os.path.dirname(path))
#     file_exists = os.path.exists(path)
#     with open(path, "a", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow({k: row.get(k, "") for k in fieldnames})

# def _data_file_path(args, data_type: str) -> str:
#     """
#     统一映射数据文件名：
#     - train -> train.json
#     - dev   -> dev.json
#     - test  -> test.json
#     """
#     fname = {"train": "train.json", "dev": "dev.json", "test": "test.json"}.get(data_type, None)
#     if fname is None:
#         raise ValueError(f"Unknown data_type: {data_type}")
#     return os.path.join(args.data_dir, fname)

# def _load_text_list_from_jsonl(path: str):
#     """
#     读取 jsonl，抽取每行的 text。
#     返回 texts(list[str or None]) 和 raw_records(list[dict])。
#     """
#     texts = []
#     raw_records = []
#     if not os.path.exists(path):
#         logger.warning("Text source file not found: %s", path)
#         return texts, raw_records

#     with open(path, "r", encoding="utf-8") as f:
#         for line_idx, line in enumerate(f):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except Exception:
#                 logger.warning("Bad json line at %s:%d", path, line_idx)
#                 obj = {}
#             raw_records.append(obj)
#             texts.append(obj.get("text"))
#     return texts, raw_records

# def _get_cached_texts(args, data_type: str):
#     """
#     ✅ 缓存 dev/test 的 text，避免每次 evaluate 都反复读文件
#     """
#     if not hasattr(args, "_cached_texts"):
#         args._cached_texts = {}
#     if data_type in args._cached_texts:
#         return args._cached_texts[data_type]

#     path = _data_file_path(args, data_type)
#     texts, raw_records = _load_text_list_from_jsonl(path)
#     args._cached_texts[data_type] = (texts, raw_records, path)
#     return args._cached_texts[data_type]

# def _spans_to_readable(spans, id2label, text: str = None):
#     """
#     spans: list[(label_id, start, end)]
#     -> list[{"label","start","end","entity_text"}]
#     """
#     out = []
#     if not spans:
#         return out
#     for x in spans:
#         if not isinstance(x, (list, tuple)) or len(x) < 3:
#             continue
#         lid, s, e = x[0], x[1], x[2]
#         lab = id2label.get(lid, str(lid))
#         s_i, e_i = int(s), int(e)

#         ent = None
#         if isinstance(text, str) and 0 <= s_i <= e_i < len(text):
#             ent = text[s_i : e_i + 1]

#         out.append({"label": lab, "start": s_i, "end": e_i, "entity_text": ent})
#     return out

# def _get_f1_from_results(results: dict):
#     for k in ("f1", "F1", "f1_score", "f1-score"):
#         if k in results:
#             try:
#                 return float(results[k])
#             except Exception:
#                 return None
#     return None

# def _save_best_model_to_output_dir(args, model, tokenizer, global_step: int, test_f1: float):
#     """
#     ✅ 只保留一份：直接覆盖保存到 args.output_dir
#     这样 do_eval/do_predict 默认加载的就是最优模型
#     """
#     _ensure_dir(args.output_dir)
#     model_to_save = model.module if hasattr(model, "module") else model
#     model_to_save.save_pretrained(args.output_dir)
#     tokenizer.save_vocabulary(args.output_dir)
#     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
#     logger.info("✅ Saved BEST(test) model to %s | step=%s | test_f1=%.6f",
#                 args.output_dir, str(global_step), float(test_f1))

# # =========================
# # train / eval / predict
# # =========================
# def train(args, train_dataset, model, tokenizer):
#     """Train the model"""
#     args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
#     train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

#     train_dataloader = DataLoader(
#         train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
#     )

#     if args.max_steps > 0:
#         t_total = args.max_steps
#         args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
#     else:
#         t_total = len(train_dataloader) // args.gradient_accumulation_steps * int(args.num_train_epochs)

#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": args.weight_decay,
#         },
#         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
#     )

#     # Check if saved optimizer or scheduler states exist
#     if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
#         os.path.join(args.model_name_or_path, "scheduler.pt")
#     ):
#         optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

#     if args.fp16:
#         try:
#             from apex import amp
#         except ImportError:
#             raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#         model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

#     # multi-gpu training
#     if args.n_gpu > 1:
#         model = torch.nn.DataParallel(model)

#     # Distributed training
#     if args.local_rank != -1:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
#             find_unused_parameters=True,
#         )

#     logger.info("***** Running training *****")
#     logger.info("  Num examples = %d", len(train_dataset))
#     logger.info("  Num Epochs = %d", int(args.num_train_epochs))
#     logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
#     # logger.info(
#     #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
#     #     args.train_batch_size
#     #     * args.gradient_accumulation_steps
#     #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
#     # )
#     logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
#     logger.info("  Total optimization steps = %d", t_total)

#     # ✅ best-on-test tracking
#     if not hasattr(args, "best_test_f1"):
#         args.best_test_f1 = -1.0
#     if not hasattr(args, "best_test_step"):
#         args.best_test_step = -1
#     if not hasattr(args, "best_saved"):
#         args.best_saved = False

#     global_step = 0
#     steps_trained_in_current_epoch = 0

#     if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
#         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
#         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
#         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
#         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
#         logger.info("  Continuing training from epoch %d", epochs_trained)
#         logger.info("  Continuing training from global step %d", global_step)
#         logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

#     tr_loss = 0.0
#     model.zero_grad()
#     seed_everything(args.seed)

#     for epoch_idx in range(int(args.num_train_epochs)):
#         pbar = ProgressBar(n_total=len(train_dataloader), desc=f"Training(Epoch{epoch_idx+1})")
#         for step, batch in enumerate(train_dataloader):
#             if steps_trained_in_current_epoch > 0:
#                 steps_trained_in_current_epoch -= 1
#                 continue

#             model.train()
#             batch = tuple(t.to(args.device) for t in batch)

#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask": batch[1],
#                 "start_positions": batch[3],
#                 "end_positions": batch[4],
#             }
#             if args.model_type != "distilbert":
#                 inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

#             inputs["expl_input_ids"] = batch[6]
#             inputs["expl_attention_mask"] = batch[7]
#             if args.model_type != "distilbert":
#                 inputs["expl_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)

#             outputs = model(**inputs)
#             loss = outputs[0]
#             if global_step < 3:  # 只看最开始几步
#                 start_logits, end_logits = outputs[1], outputs[2]
#                 start_pred = start_logits.argmax(-1).view(-1)
#                 end_pred = end_logits.argmax(-1).view(-1)
#                 logger.info("[DEBUG train] gs=%d start_nonzero=%d/%d end_nonzero=%d/%d",
#                             global_step,
#                             (start_pred != 0).sum().item(), start_pred.numel(),
#                             (end_pred != 0).sum().item(), end_pred.numel())

#             if args.n_gpu > 1:
#                 loss = loss.mean()
#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps

#             if args.fp16:
#                 from apex import amp
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#             else:
#                 loss.backward()

#             pbar(step, {"loss": loss.item()})
#             tr_loss += loss.item()

#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 if args.fp16:
#                     from apex import amp
#                     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
#                 else:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

#                 scheduler.step()
#                 optimizer.step()
#                 model.zero_grad()
#                 global_step += 1

#                 # ===== logging：每次都评估 dev + test；并且只在 test F1 变好时保存模型 =====
#                 if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
#                     print(" ")
#                     if args.local_rank == -1:
#                         _ = evaluate(args, model, tokenizer, prefix="dev", data_type="dev", global_step=global_step)
#                         test_res = evaluate(args, model, tokenizer, prefix="test", data_type="test", global_step=global_step)

#                         test_f1 = _get_f1_from_results(test_res)
#                         if test_f1 is not None and test_f1 > float(getattr(args, "best_test_f1", -1.0)):
#                             args.best_test_f1 = float(test_f1)
#                             args.best_test_step = int(global_step)
#                             args.best_saved = True
#                             _save_best_model_to_output_dir(args, model, tokenizer, global_step=global_step, test_f1=test_f1)

#                 # ✅ 已按你的需求：不再按 save_steps 保存 checkpoint
#                 # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
#                 #     ...

#         print(" ")
#         if "cuda" in str(args.device):
#             torch.cuda.empty_cache()

#     return global_step, tr_loss / max(global_step, 1)

# def evaluate(args, model, tokenizer, prefix="", data_type="dev", global_step=None):
#     """
#     ✅ 通用评估：dev / test 都可以
#     ✅ 你的需求（已改为 CSV）：
#       - dev/test 明细文件里写入原始 text（从 args.data_dir/{dev|test}.json 读）
#       - 明细里包含 true/pred 的 span + entity_text（从 text 截取）
#       - 分数每次评估追加写入 eval_scores.csv
#     """
#     metric = SpanEntityScore(args.id2label)
#     _ensure_dir(args.output_dir)

#     log_dir = os.path.join(args.output_dir, "eval_logs")
#     _ensure_dir(log_dir)

#     scores_file = os.path.join(log_dir, "eval_scores.csv")
#     details_file = os.path.join(log_dir, f"{data_type}_details.csv")  # dev_details.csv / test_details.csv

#     # CSV 字段
#     score_fields = [
#         "time", "global_step", "prefix", "data_type",
#         "loss", "precision", "recall", "f1", "acc",
#         "results_json", "entity_info_json",
#         "text_source", "details_file"
#     ]
#     details_fields = [
#         "time", "global_step", "prefix", "data_type",
#         "example_id", "text", "true_json", "pred_json"
#     ]

#     # ✅ 读原始 text（稳定）
#     texts, raw_records, text_source_path = _get_cached_texts(args, data_type)

#     # ✅ 评估用 features 列表（不是 TensorDataset）
#     eval_features = load_and_cache_examples(
#         args, args.task_name, tokenizer,
#         data_type=data_type,
#         return_features=True
#     )

#     # 长度不一致也能跑，但会提示
#     if len(texts) != 0 and len(texts) != len(eval_features):
#         logger.warning(
#             "[WARN] text lines (%d) != eval_features (%d) for %s. text_source=%s",
#             len(texts), len(eval_features), data_type, text_source_path
#         )

#     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

#     logger.info("***** Running evaluation (%s) %s *****", data_type, prefix)
#     logger.info("  Num examples = %d", len(eval_features))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     logger.info("  Text source = %s", text_source_path)

#     eval_loss = 0.0
#     nb_eval_steps = 0
#     pbar = ProgressBar(n_total=len(eval_features), desc=f"Evaluating-{data_type}")

#     for step, f in enumerate(eval_features):
#         input_lens = f.input_len
#         input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
#         input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
#         segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
#         start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
#         end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
#         expl_lens = getattr(f, "expl_input_len", 0)
#         expl_input_ids = torch.tensor([f.expl_input_ids[:expl_lens]], dtype=torch.long).to(args.device)
#         expl_input_mask = torch.tensor([f.expl_input_mask[:expl_lens]], dtype=torch.long).to(args.device)
#         expl_segment_ids = torch.tensor([f.expl_segment_ids[:expl_lens]], dtype=torch.long).to(args.device)


#         subjects_true = getattr(f, "subjects", [])

#         # ✅ 拿原始 text（按行对齐）
#         text = None
#         if step < len(texts):
#             text = texts[step]

#         model.eval()
#         with torch.no_grad():
#             inputs = {
#                 "input_ids": input_ids,
#                 "attention_mask": input_mask,
#                 "start_positions": start_ids,
#                 "end_positions": end_ids,
#             }
#             if args.model_type != "distilbert":
#                 inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)

#             inputs["expl_input_ids"] = expl_input_ids
#             inputs["expl_attention_mask"] = expl_input_mask
#             if args.model_type != "distilbert":
#                 inputs["expl_token_type_ids"] = (expl_segment_ids if args.model_type in ["bert", "xlnet"] else None)

#             outputs = model(**inputs)
#             tmp_eval_loss, start_logits, end_logits = outputs[:3]
#             # ===== DEBUG：检查是否塌缩成全 O =====
#             if step < 3:  # 只看前3条，避免刷屏
#                 start_pred = start_logits.argmax(-1).view(-1)  # [L]
#                 end_pred   = end_logits.argmax(-1).view(-1)

#                 uniq_s, cnt_s = torch.unique(start_pred, return_counts=True)
#                 uniq_e, cnt_e = torch.unique(end_pred, return_counts=True)

#                 logger.info("[DEBUG %s] step=%d start_nonzero=%d/%d uniq=%s cnt=%s",
#                             data_type, step,
#                             (start_pred != 0).sum().item(), start_pred.numel(),
#                             uniq_s.detach().cpu().tolist(), cnt_s.detach().cpu().tolist())

#                 logger.info("[DEBUG %s] step=%d end_nonzero=%d/%d uniq=%s cnt=%s",
#                             data_type, step,
#                             (end_pred != 0).sum().item(), end_pred.numel(),
#                             uniq_e.detach().cpu().tolist(), cnt_e.detach().cpu().tolist())
#             subjects_pred = bert_extract_item(start_logits, end_logits)
#             if step < 3:
#                 logger.info("[DEBUG %s] step=%d len(subjects_true)=%d len(subjects_pred)=%d",
#                             data_type, step, len(subjects_true), len(subjects_pred))         
#             metric.update(true_subject=subjects_true, pred_subject=subjects_pred)

#             # ====== ✅ 明细输出：每条样本写一行（CSV） ======
#             now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             rec = {
#                 "time": now_str,
#                 "global_step": global_step,
#                 "prefix": prefix,
#                 "data_type": data_type,
#                 "example_id": step,
#                 "text": text,
#                 "true_json": _to_json_str(_spans_to_readable(subjects_true, args.id2label, text=text)),
#                 "pred_json": _to_json_str(_spans_to_readable(subjects_pred, args.id2label, text=text)),
#             }
#             _append_csv_row(details_file, rec, details_fields)

#             if args.n_gpu > 1:
#                 tmp_eval_loss = tmp_eval_loss.mean()
#             eval_loss += tmp_eval_loss.item()

#         nb_eval_steps += 1
#         pbar(step)

#     print(" ")
#     eval_loss = eval_loss / max(nb_eval_steps, 1)
#     eval_info, entity_info = metric.result()

#     results = {f"{key}": value for key, value in eval_info.items()}
#     results["loss"] = eval_loss

#     logger.info("***** Eval results (%s) %s *****", data_type, prefix)
#     info = "-".join([f" {key}: {value:.4f} " for key, value in results.items() if isinstance(value, (int, float))])
#     logger.info(info if info else str(results))

#     logger.info("***** Entity results (%s) %s *****", data_type, prefix)
#     for key in sorted(entity_info.keys()):
#         logger.info("******* %s results ********" % key)
#         info = "-".join([f" {k}: {v:.4f} " for k, v in entity_info[key].items() if isinstance(v, (int, float))])
#         logger.info(info if info else str(entity_info[key]))

#     # ====== ✅ 每次评估分数写入 CSV ======
#     score_row = {
#         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "global_step": global_step,
#         "prefix": prefix,
#         "data_type": data_type,
#         "loss": results.get("loss", ""),
#         "precision": results.get("precision", ""),
#         "recall": results.get("recall", ""),
#         "f1": results.get("f1", ""),
#         "acc": results.get("acc", ""),
#         "results_json": _to_json_str(results),
#         "entity_info_json": _to_json_str(entity_info),
#         "text_source": text_source_path,
#         "details_file": details_file,
#     }
#     _append_csv_row(scores_file, score_row, score_fields)

#     return results

# def predict(args, model, tokenizer, prefix=""):
#     pred_output_dir = args.output_dir
#     if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
#         os.makedirs(pred_output_dir)

#     test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="test", return_features=False)
#     print(len(test_dataset))

#     test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
#     test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)

#     logger.info("***** Running prediction %s *****", prefix)
#     logger.info("  Num examples = %d", len(test_dataset))
#     logger.info("  Batch size = %d", 1)

#     results = []
#     pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
#     for step, batch in enumerate(test_dataloader):
#         model.eval()
#         batch = tuple(t.to(args.device) for t in batch)
#         with torch.no_grad():
#             inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None, "end_positions": None}
#             if args.model_type != "distilbert":
#                 inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
#             inputs["expl_input_ids"] = batch[6]
#             inputs["expl_attention_mask"] = batch[7]
#             if args.model_type != "distilbert":
#                 inputs["expl_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)
#             outputs = model(**inputs)

#         start_logits, end_logits = outputs[:2]
#         R = bert_extract_item(start_logits, end_logits)

#         if R:
#             label_entities = [[args.id2label[x[0]], x[1], x[2]] for x in R]
#         else:
#             label_entities = []

#         json_d = {"id": step, "entities": label_entities}
#         results.append(json_d)
#         pbar(step)

#     print(" ")
#     os.makedirs(os.path.join(pred_output_dir, prefix), exist_ok=True)
#     output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
#     output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")

#     with open(output_predic_file, "w", encoding="utf-8") as writer:
#         for record in results:
#             writer.write(json.dumps(record, ensure_ascii=False) + "\n")

#     test_text = []
#     with open(os.path.join(args.data_dir, "test.json"), "r", encoding="utf-8") as fr:
#         for line in fr:
#             test_text.append(json.loads(line))

#     test_submit = []
#     for x, y in zip(test_text, results):
#         json_d = {}
#         json_d["id"] = x.get("id", None)
#         json_d["label"] = {}
#         entities = y["entities"]
#         words = list(x["text"])
#         if len(entities) != 0:
#             for subject in entities:
#                 tag = subject[0]
#                 start = subject[1]
#                 end = subject[2]
#                 word = "".join(words[start : end + 1])
#                 if tag in json_d["label"]:
#                     if word in json_d["label"][tag]:
#                         json_d["label"][tag][word].append([start, end])
#                     else:
#                         json_d["label"][tag][word] = [[start, end]]
#                 else:
#                     json_d["label"][tag] = {}
#                     json_d["label"][tag][word] = [[start, end]]
#         test_submit.append(json_d)

#     json_to_text(output_submit_file, test_submit)

# def load_and_cache_examples(args, task, tokenizer, data_type="train", return_features=False):
#     """
#     ✅ return_features=True：返回 features 列表（用于 evaluate）
#     ✅ return_features=False：返回 TensorDataset（用于 train/predict）
#     """
#     if args.local_rank not in [-1, 0] and not evaluate:
#         torch.distributed.barrier()

#     processor = processors[task]()

#     cached_features_file = os.path.join(
#         args.data_dir,
#         "cached_span-{}_{}_text{}_expl{}_{}".format(
#             data_type,
#             list(filter(None, args.model_name_or_path.split("/"))).pop(),
#             str(args.train_max_seq_length if data_type == "train" else args.eval_max_seq_length),
#             str(args.train_expl_max_seq_length if data_type == "train" else args.eval_expl_max_seq_length),
#             str(task),
#         ),
#     )

#     if os.path.exists(cached_features_file) and not args.overwrite_cache:
#         logger.info("Loading features from cached file %s", cached_features_file)
#         features = torch.load(cached_features_file)
#     else:
#         logger.info("Creating features from dataset file at %s", args.data_dir)
#         label_list = processor.get_labels()

#         if data_type == "train":
#             examples = processor.get_train_examples(args.data_dir)
#         elif data_type == "dev":
#             examples = processor.get_dev_examples(args.data_dir)
#         elif data_type == "test":
#             examples = processor.get_test_examples(args.data_dir)
#         else:
#             raise ValueError(f"Unknown data_type: {data_type}")

#         features = convert_examples_to_features(
#             examples=examples,
#             tokenizer=tokenizer,
#             label_list=label_list,
#             max_seq_length=args.train_max_seq_length if data_type == "train" else args.eval_max_seq_length,
#             expl_max_seq_length=args.train_expl_max_seq_length if data_type == "train" else args.eval_expl_max_seq_length,
#             cls_token_at_end=bool(args.model_type in ["xlnet"]),
#             pad_on_left=bool(args.model_type in ["xlnet"]),
#             cls_token=tokenizer.cls_token,
#             cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
#             sep_token=tokenizer.sep_token,
#             pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
#             pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
#         )

#         if args.local_rank in [-1, 0]:
#             logger.info("Saving features into cached file %s", cached_features_file)
#             torch.save(features, cached_features_file)

#     if args.local_rank == 0 and not evaluate:
#         torch.distributed.barrier()

#     if return_features:
#         return features

#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#     all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
#     all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
#     all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
#     all_expl_input_ids = torch.tensor([f.expl_input_ids for f in features], dtype=torch.long)
#     all_expl_input_mask = torch.tensor([f.expl_input_mask for f in features], dtype=torch.long)
#     all_expl_segment_ids = torch.tensor([f.expl_segment_ids for f in features], dtype=torch.long)
#     all_expl_input_lens = torch.tensor([f.expl_input_len for f in features], dtype=torch.long)


#     dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens,
#                            all_expl_input_ids, all_expl_input_mask, all_expl_segment_ids, all_expl_input_lens)
#     return dataset

# def main():
#     parser = argparse.ArgumentParser()

#     # Required parameters
#     parser.add_argument("--task_name", default=None, type=str, required=True,
#                         help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
#     parser.add_argument("--data_dir", default=None, type=str, required=True,
#                         help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
#     parser.add_argument("--model_type", default=None, type=str, required=True,
#                         help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
#     parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
#                         help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
#     parser.add_argument("--output_dir", default=None, type=str, required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")

#     # Other parameters
#     parser.add_argument("--markup", default="bio", type=str, choices=["bios", "bio"])
#     parser.add_argument("--loss_type", default="lsr", type=str, choices=["lsr", "focal", "ce"])
#     parser.add_argument("--labels", default="", type=str,
#                         help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
#     parser.add_argument("--config_name", default="", type=str,
#                         help="Pretrained config name or path if not the same as model_name")
#     parser.add_argument("--tokenizer_name", default="", type=str,
#                         help="Pretrained tokenizer name or path if not the same as model_name")
#     parser.add_argument("--cache_dir", default="", type=str,
#                         help="Where do you want to store the pre-trained models downloaded from s3")

#     parser.add_argument("--train_max_seq_length", default=128, type=int,
#                         help="The maximum total input sequence length after tokenization (train).")
#     parser.add_argument("--eval_max_seq_length", default=512, type=int,
#                         help="The maximum total input sequence length after tokenization (eval).")

#     parser.add_argument("--train_expl_max_seq_length", default=128, type=int,
#                         help="The maximum explanation input length after tokenization (train).")
#     parser.add_argument("--eval_expl_max_seq_length", default=128, type=int,
#                         help="The maximum explanation input length after tokenization (eval).")

#     parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
#     parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
#     parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
#     parser.add_argument("--evaluate_during_training", action="store_true",
#                         help="Whether to run evaluation during training at each logging step.")
#     parser.add_argument("--do_lower_case", action="store_true",
#                         help="Set this flag if you are using an uncased model.")

#     parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
#                         help="Batch size per GPU/CPU for training.")
#     parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
#                         help="Batch size per GPU/CPU for evaluation.")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
#     parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
#     parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
#     parser.add_argument("--num_train_epochs", default=10.0, type=float,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--max_steps", default=-1, type=int,
#                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

#     parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
#     parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
#     parser.add_argument("--save_steps", type=int, default=50, help="(unused now) previously saved checkpoint every X steps.")
#     parser.add_argument("--eval_all_checkpoints", action="store_true",
#                         help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
#     parser.add_argument("--predict_checkpoints", type=int, default=0,
#                         help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")

#     parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
#     parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
#     parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached sets")
#     parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

#     parser.add_argument("--fp16", action="store_true",
#                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
#     parser.add_argument("--fp16_opt_level", type=str, default="O1",
#                         help="For fp16: Apex AMP optimization level selected in ['O0','O1','O2','O3'].")

#     parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
#     parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
#     parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
#     args = parser.parse_args()

#     # output_dir/<model_type>
#     if not os.path.exists(args.output_dir):
#         os.mkdir(args.output_dir)
#     args.output_dir = os.path.join(args.output_dir, str(args.model_type))
#     if not os.path.exists(args.output_dir):
#         os.mkdir(args.output_dir)

#     init_logger(log_file=args.output_dir + "/{}-{}.log".format(args.model_type, args.task_name))

#     if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
#         raise ValueError(
#             "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
#                 args.output_dir
#             )
#         )

#     if args.server_ip and args.server_port:
#         import ptvsd
#         print("Waiting for debugger attach")
#         ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
#         ptvsd.wait_for_attach()

#     # Setup CUDA, GPU & distributed training
#     if args.local_rank == -1 or args.no_cuda:
#         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#         args.n_gpu = torch.cuda.device_count()
#     else:
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         torch.distributed.init_process_group(backend="nccl")
#         args.n_gpu = 1
#     args.device = device

#     logger.warning(
#         "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#         args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
#     )

#     seed_everything(args.seed)

#     # Prepare NER task
#     args.task_name = args.task_name.lower()
#     if args.task_name not in processors:
#         raise ValueError("Task not found: %s" % args.task_name)

#     processor = processors[args.task_name]()
#     label_list = processor.get_labels()
#     args.id2label = {i: label for i, label in enumerate(label_list)}
#     args.label2id = {label: i for i, label in enumerate(label_list)}
#     num_labels = len(label_list)

#     # Load pretrained model and tokenizer
#     if args.local_rank not in [-1, 0]:
#         torch.distributed.barrier()

#     args.model_type = args.model_type.lower()
#     config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

#     config = config_class.from_pretrained(
#         args.config_name if args.config_name else args.model_name_or_path,
#         num_labels=num_labels,
#         loss_type=args.loss_type,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#         soft_label=True,
#     )
#     tokenizer = tokenizer_class.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         do_lower_case=args.do_lower_case,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )
#     model = model_class.from_pretrained(
#         args.model_name_or_path,
#         from_tf=bool(".ckpt" in args.model_name_or_path),
#         config=config,
#     )

#     if args.local_rank == 0:
#         torch.distributed.barrier()

#     model.to(args.device)
#     logger.info("Training/evaluation parameters %s", args)

#     # Training
#     if args.do_train:
#         train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="train", return_features=False)
#         global_step, tr_loss = train(args, train_dataset, model, tokenizer)
#         logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

#     # ✅ 训练结束：不要再无条件覆盖保存（避免把 best 覆盖成 last）
#     if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
#         if not getattr(args, "best_saved", False):
#             if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
#                 os.makedirs(args.output_dir)
#             logger.info("No best(test) saved during training; saving LAST model to %s", args.output_dir)
#             model_to_save = model.module if hasattr(model, "module") else model
#             model_to_save.save_pretrained(args.output_dir)
#             tokenizer.save_vocabulary(args.output_dir)
#             torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
#         else:
#             logger.info(
#                 "Best(test) model already saved: step=%s f1=%.6f at %s",
#                 str(getattr(args, "best_test_step", "")),
#                 float(getattr(args, "best_test_f1", -1.0)),
#                 args.output_dir,
#             )

#     # Evaluation (manual eval entry)
#     results = {}
#     if args.do_eval and args.local_rank in [-1, 0]:
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         checkpoints = [args.output_dir]
#         if args.eval_all_checkpoints:
#             checkpoints = list(
#                 os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
#             )
#             logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)

#         logger.info("Evaluate the following checkpoints: %s", checkpoints)
#         for checkpoint in checkpoints:
#             ckpt_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
#             prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
#             model = model_class.from_pretrained(checkpoint)
#             model.to(args.device)

#             result_dev = evaluate(args, model, tokenizer, prefix=prefix, data_type="dev", global_step=ckpt_step)
#             result_test = evaluate(args, model, tokenizer, prefix=prefix, data_type="test", global_step=ckpt_step)

#             if ckpt_step:
#                 result_dev = {"dev_{}_{}".format(ckpt_step, k): v for k, v in result_dev.items()}
#                 result_test = {"test_{}_{}".format(ckpt_step, k): v for k, v in result_test.items()}

#             results.update(result_dev)
#             results.update(result_test)

#         output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
#         with open(output_eval_file, "w", encoding="utf-8") as writer:
#             for key in sorted(results.keys()):
#                 writer.write("{} = {}\n".format(key, str(results[key])))

#     # Predict
#     if args.do_predict and args.local_rank in [-1, 0]:
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         checkpoints = [args.output_dir]
#         if args.predict_checkpoints > 0:
#             checkpoints = list(
#                 os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
#             )
#             logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
#             checkpoints = [x for x in checkpoints if x.split("-")[-1] == str(args.predict_checkpoints)]

#         logger.info("Predict the following checkpoints: %s", checkpoints)
#         for checkpoint in checkpoints:
#             prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
#             model = model_class.from_pretrained(checkpoint)
#             model.to(args.device)
#             predict(args, model, tokenizer, prefix=prefix)

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import json
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything, json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_ner import BertSpanForNer
from models.albert_for_ner import AlbertSpanForNer
from processors.utils_ner import CNerTokenizer, bert_extract_item
from processors.ner_span import convert_examples_to_features
from processors.ner_span import ner_processors as processors
from processors.ner_span import collate_fn
from metrics.ner_metrics import SpanEntityScore

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    # bert ernie bert_wwm bert_wwwm_ext
    "bert": (BertConfig, BertSpanForNer, CNerTokenizer),
    "albert": (AlbertConfig, AlbertSpanForNer, CNerTokenizer),
}

# =========================
# ✅ I/O helpers
# =========================
def _ensure_dir(path: str):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)

def _append_jsonl(path: str, obj: dict):
    """保留：如果你后续还想写 jsonl 可以继续用（当前改造后不再使用）"""
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _to_json_str(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def _append_csv_row(path: str, row: dict, fieldnames: list):
    _ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def _data_file_path(args, data_type: str) -> str:
    """
    统一映射数据文件名：
    - train -> train.json
    - dev   -> dev.json
    - test  -> test.json
    """
    fname = {"train": "train.json", "dev": "dev.json", "test": "test.json"}.get(data_type, None)
    if fname is None:
        raise ValueError(f"Unknown data_type: {data_type}")
    return os.path.join(args.data_dir, fname)

def _load_text_list_from_jsonl(path: str):
    """
    读取 jsonl，抽取每行的 text。
    返回 texts(list[str or None]) 和 raw_records(list[dict])。
    """
    texts = []
    raw_records = []
    if not os.path.exists(path):
        logger.warning("Text source file not found: %s", path)
        return texts, raw_records

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                logger.warning("Bad json line at %s:%d", path, line_idx)
                obj = {}
            raw_records.append(obj)
            texts.append(obj.get("text"))
    return texts, raw_records

def _get_cached_texts(args, data_type: str):
    """
    ✅ 缓存 dev/test 的 text，避免每次 evaluate 都反复读文件
    """
    if not hasattr(args, "_cached_texts"):
        args._cached_texts = {}
    if data_type in args._cached_texts:
        return args._cached_texts[data_type]

    path = _data_file_path(args, data_type)
    texts, raw_records = _load_text_list_from_jsonl(path)
    args._cached_texts[data_type] = (texts, raw_records, path)
    return args._cached_texts[data_type]

def _spans_to_readable(spans, id2label, text: str = None):
    """
    spans: list[(label_id, start, end)]
    -> list[{"label","start","end","entity_text"}]
    """
    out = []
    if not spans:
        return out
    for x in spans:
        if not isinstance(x, (list, tuple)) or len(x) < 3:
            continue
        lid, s, e = x[0], x[1], x[2]
        lab = id2label.get(lid, str(lid))
        s_i, e_i = int(s), int(e)

        ent = None
        if isinstance(text, str) and 0 <= s_i <= e_i < len(text):
            ent = text[s_i : e_i + 1]

        out.append({"label": lab, "start": s_i, "end": e_i, "entity_text": ent})
    return out

def _get_f1_from_results(results: dict):
    for k in ("f1", "F1", "f1_score", "f1-score"):
        if k in results:
            try:
                return float(results[k])
            except Exception:
                return None
    return None

def _save_best_model_to_output_dir(args, model, tokenizer, global_step: int, dev_f1: float):
    """
    ✅ 只保留一份：直接覆盖保存到 args.output_dir
    这样 do_eval/do_predict 默认加载的就是最优模型（按 dev F1 选）
    """
    _ensure_dir(args.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_vocabulary(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    logger.info("✅ Saved BEST(dev) model to %s | step=%s | dev_f1=%.6f",
                args.output_dir, str(global_step), float(dev_f1))

# =========================
# train / eval / predict
# =========================
def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * int(args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", int(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # ✅ best-on-dev tracking
    if not hasattr(args, "best_dev_f1"):
        args.best_dev_f1 = -1.0
    if not hasattr(args, "best_dev_step"):
        args.best_dev_step = -1
    if not hasattr(args, "best_saved"):
        args.best_saved = False

    global_step = 0
    steps_trained_in_current_epoch = 0

    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss = 0.0
    model.zero_grad()
    seed_everything(args.seed)

    for epoch_idx in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc=f"Training(Epoch{epoch_idx+1})")
        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

            inputs["expl_input_ids"] = batch[6]
            inputs["expl_attention_mask"] = batch[7]
            if args.model_type != "distilbert":
                inputs["expl_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)

            outputs = model(**inputs)
            loss = outputs[0]
            if global_step < 3:  # 只看最开始几步
                start_logits, end_logits = outputs[1], outputs[2]
                start_pred = start_logits.argmax(-1).view(-1)
                end_pred = end_logits.argmax(-1).view(-1)
                logger.info("[DEBUG train] gs=%d start_nonzero=%d/%d end_nonzero=%d/%d",
                            global_step,
                            (start_pred != 0).sum().item(), start_pred.numel(),
                            (end_pred != 0).sum().item(), end_pred.numel())

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            pbar(step, {"loss": loss.item()})
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    from apex import amp
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # ===== logging：只评估 dev；用 dev F1 保存 best（不再每次 dev 后都跑 test）=====
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    if args.local_rank == -1:
                        dev_res = evaluate(args, model, tokenizer, prefix="dev", data_type="dev", global_step=global_step)
                        dev_f1 = _get_f1_from_results(dev_res)

                        if dev_f1 is not None and dev_f1 > float(getattr(args, "best_dev_f1", -1.0)):
                            args.best_dev_f1 = float(dev_f1)
                            args.best_dev_step = int(global_step)
                            args.best_saved = True
                            _save_best_model_to_output_dir(args, model, tokenizer, global_step=global_step, dev_f1=dev_f1)

                # ✅ 已按你的需求：不再按 save_steps 保存 checkpoint
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     ...

        print(" ")
        if "cuda" in str(args.device):
            torch.cuda.empty_cache()

    return global_step, tr_loss / max(global_step, 1)

def evaluate(args, model, tokenizer, prefix="", data_type="dev", global_step=None):
    """
    ✅ 通用评估：dev / test 都可以
    ✅ 改造点：
      - dev/test 的分数分别写入 eval_scores_dev.csv / eval_scores_test.csv（不混在一起）
      - 明细分别写入 dev_details.csv / test_details.csv
      - 明细里包含原始 text + true/pred spans + entity_text
    """
    metric = SpanEntityScore(args.id2label)
    _ensure_dir(args.output_dir)

    log_dir = os.path.join(args.output_dir, "eval_logs")
    _ensure_dir(log_dir)

    # ✅ dev/test 分数分开写
    scores_file = os.path.join(log_dir, f"eval_scores_{data_type}.csv")
    details_file = os.path.join(log_dir, f"{data_type}_details.csv")

    # CSV 字段
    score_fields = [
        "time", "global_step", "prefix", "data_type",
        "loss", "precision", "recall", "f1", "acc",
        "results_json", "entity_info_json",
        "text_source", "details_file"
    ]
    details_fields = [
        "time", "global_step", "prefix", "data_type",
        "example_id", "text", "true_json", "pred_json"
    ]

    # ✅ 读原始 text（稳定）
    texts, raw_records, text_source_path = _get_cached_texts(args, data_type)

    # ✅ 评估用 features 列表（不是 TensorDataset）
    eval_features = load_and_cache_examples(
        args, args.task_name, tokenizer,
        data_type=data_type,
        return_features=True
    )

    # 长度不一致也能跑，但会提示
    if len(texts) != 0 and len(texts) != len(eval_features):
        logger.warning(
            "[WARN] text lines (%d) != eval_features (%d) for %s. text_source=%s",
            len(texts), len(eval_features), data_type, text_source_path
        )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    logger.info("***** Running evaluation (%s) %s *****", data_type, prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Text source = %s", text_source_path)

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc=f"Evaluating-{data_type}")

    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)

        expl_lens = getattr(f, "expl_input_len", 0)
        expl_input_ids = torch.tensor([f.expl_input_ids[:expl_lens]], dtype=torch.long).to(args.device)
        expl_input_mask = torch.tensor([f.expl_input_mask[:expl_lens]], dtype=torch.long).to(args.device)
        expl_segment_ids = torch.tensor([f.expl_segment_ids[:expl_lens]], dtype=torch.long).to(args.device)

        subjects_true = getattr(f, "subjects", [])

        # ✅ 拿原始 text（按行对齐）
        text = None
        if step < len(texts):
            text = texts[step]

        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "start_positions": start_ids,
                "end_positions": end_ids,
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)

            inputs["expl_input_ids"] = expl_input_ids
            inputs["expl_attention_mask"] = expl_input_mask
            if args.model_type != "distilbert":
                inputs["expl_token_type_ids"] = (expl_segment_ids if args.model_type in ["bert", "xlnet"] else None)

            outputs = model(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]

            # ===== DEBUG：检查是否塌缩成全 O =====
            if step < 3:
                start_pred = start_logits.argmax(-1).view(-1)
                end_pred = end_logits.argmax(-1).view(-1)

                uniq_s, cnt_s = torch.unique(start_pred, return_counts=True)
                uniq_e, cnt_e = torch.unique(end_pred, return_counts=True)

                logger.info("[DEBUG %s] step=%d start_nonzero=%d/%d uniq=%s cnt=%s",
                            data_type, step,
                            (start_pred != 0).sum().item(), start_pred.numel(),
                            uniq_s.detach().cpu().tolist(), cnt_s.detach().cpu().tolist())

                logger.info("[DEBUG %s] step=%d end_nonzero=%d/%d uniq=%s cnt=%s",
                            data_type, step,
                            (end_pred != 0).sum().item(), end_pred.numel(),
                            uniq_e.detach().cpu().tolist(), cnt_e.detach().cpu().tolist())

            subjects_pred = bert_extract_item(start_logits, end_logits)

            if step < 3:
                logger.info("[DEBUG %s] step=%d len(subjects_true)=%d len(subjects_pred)=%d",
                            data_type, step, len(subjects_true), len(subjects_pred))

            metric.update(true_subject=subjects_true, pred_subject=subjects_pred)

            # ====== ✅ 明细输出：每条样本写一行（CSV） ======
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rec = {
                "time": now_str,
                "global_step": global_step,
                "prefix": prefix,
                "data_type": data_type,
                "example_id": step,
                "text": text,
                "true_json": _to_json_str(_spans_to_readable(subjects_true, args.id2label, text=text)),
                "pred_json": _to_json_str(_spans_to_readable(subjects_pred, args.id2label, text=text)),
            }
            _append_csv_row(details_file, rec, details_fields)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        pbar(step)

    print(" ")
    eval_loss = eval_loss / max(nb_eval_steps, 1)
    eval_info, entity_info = metric.result()

    results = {f"{key}": value for key, value in eval_info.items()}
    results["loss"] = eval_loss

    logger.info("***** Eval results (%s) %s *****", data_type, prefix)
    info = "-".join([f" {key}: {value:.4f} " for key, value in results.items() if isinstance(value, (int, float))])
    logger.info(info if info else str(results))

    logger.info("***** Entity results (%s) %s *****", data_type, prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f" {k}: {v:.4f} " for k, v in entity_info[key].items() if isinstance(v, (int, float))])
        logger.info(info if info else str(entity_info[key]))

    # ====== ✅ 每次评估分数写入 CSV（dev/test 分开写） ======
    score_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "global_step": global_step,
        "prefix": prefix,
        "data_type": data_type,
        "loss": results.get("loss", ""),
        "precision": results.get("precision", ""),
        "recall": results.get("recall", ""),
        "f1": results.get("f1", ""),
        "acc": results.get("acc", ""),
        "results_json": _to_json_str(results),
        "entity_info_json": _to_json_str(entity_info),
        "text_source": text_source_path,
        "details_file": details_file,
    }
    _append_csv_row(scores_file, score_row, score_fields)

    return results

def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="test", return_features=False)
    print(len(test_dataset))

    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)

    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None, "end_positions": None}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            inputs["expl_input_ids"] = batch[6]
            inputs["expl_attention_mask"] = batch[7]
            if args.model_type != "distilbert":
                inputs["expl_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)

        start_logits, end_logits = outputs[:2]
        R = bert_extract_item(start_logits, end_logits)

        if R:
            label_entities = [[args.id2label[x[0]], x[1], x[2]] for x in R]
        else:
            label_entities = []

        json_d = {"id": step, "entities": label_entities}
        results.append(json_d)
        pbar(step)

    print(" ")
    os.makedirs(os.path.join(pred_output_dir, prefix), exist_ok=True)
    output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")

    with open(output_predic_file, "w", encoding="utf-8") as writer:
        for record in results:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    test_text = []
    with open(os.path.join(args.data_dir, "test.json"), "r", encoding="utf-8") as fr:
        for line in fr:
            test_text.append(json.loads(line))

    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d["id"] = x.get("id", None)
        json_d["label"] = {}
        entities = y["entities"]
        words = list(x["text"])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start : end + 1])
                if tag in json_d["label"]:
                    if word in json_d["label"][tag]:
                        json_d["label"][tag][word].append([start, end])
                    else:
                        json_d["label"][tag][word] = [[start, end]]
                else:
                    json_d["label"][tag] = {}
                    json_d["label"][tag][word] = [[start, end]]
        test_submit.append(json_d)

    json_to_text(output_submit_file, test_submit)

def load_and_cache_examples(args, task, tokenizer, data_type="train", return_features=False):
    """
    ✅ return_features=True：返回 features 列表（用于 evaluate）
    ✅ return_features=False：返回 TensorDataset（用于 train/predict）
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_span-{}_{}_text{}_expl{}_{}".format(
            data_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.train_max_seq_length if data_type == "train" else args.eval_max_seq_length),
            str(args.train_expl_max_seq_length if data_type == "train" else args.eval_expl_max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label_list=label_list,
            max_seq_length=args.train_max_seq_length if data_type == "train" else args.eval_max_seq_length,
            expl_max_seq_length=args.train_expl_max_seq_length if data_type == "train" else args.eval_expl_max_seq_length,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            pad_on_left=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if return_features:
        return features

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_expl_input_ids = torch.tensor([f.expl_input_ids for f in features], dtype=torch.long)
    all_expl_input_mask = torch.tensor([f.expl_input_mask for f in features], dtype=torch.long)
    all_expl_segment_ids = torch.tensor([f.expl_segment_ids for f in features], dtype=torch.long)
    all_expl_input_lens = torch.tensor([f.expl_input_len for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens,
        all_expl_input_ids, all_expl_input_mask, all_expl_segment_ids, all_expl_input_lens
    )
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--markup", default="bio", type=str, choices=["bios", "bio"])
    parser.add_argument("--loss_type", default="lsr", type=str, choices=["lsr", "focal", "ce"])
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization (train).")
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization (eval).")

    parser.add_argument("--train_expl_max_seq_length", default=256, type=int,
                        help="The maximum explanation input length after tokenization (train).")
    parser.add_argument("--eval_expl_max_seq_length", default=256, type=int,
                        help="The maximum explanation input length after tokenization (eval).")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="(unused now) previously saved checkpoint every X steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0','O1','O2','O3'].")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # output_dir/<model_type>
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, str(args.model_type))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    init_logger(log_file=args.output_dir + "/{}-{}.log".format(args.model_type, args.task_name))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
    )

    seed_everything(args.seed)

    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        loss_type=args.loss_type,
        cache_dir=args.cache_dir if args.cache_dir else None,
        soft_label=True,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_global_step = None
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="train", return_features=False)
        train_global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", train_global_step, tr_loss)

    # ✅ 训练结束：不要再无条件覆盖保存（避免把 best 覆盖成 last）
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not getattr(args, "best_saved", False):
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)
            logger.info("No best(dev) saved during training; saving LAST model to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_vocabulary(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        else:
            logger.info(
                "Best(dev) model already saved: step=%s f1=%.6f at %s",
                str(getattr(args, "best_dev_step", "")),
                float(getattr(args, "best_dev_f1", -1.0)),
                args.output_dir,
            )

        # ============================
        # ✅ FINAL: 所有 epoch 结束后，用 best(dev) 模型跑一次 test，并打印最终结果
        # ============================
        logger.info("===== Final evaluation on TEST using BEST(dev) checkpoint =====")
        best_ckpt_dir = args.output_dir  # best(dev) 已覆盖保存到这里；若没有 best_saved，则这里是 last

        best_tokenizer = tokenizer_class.from_pretrained(best_ckpt_dir, do_lower_case=args.do_lower_case)
        best_model = model_class.from_pretrained(best_ckpt_dir)
        best_model.to(args.device)

        best_step = str(getattr(args, "best_dev_step", "")) if getattr(args, "best_saved", False) else str(train_global_step)
        final_test_res = evaluate(
            args,
            best_model,
            best_tokenizer,
            prefix="best_dev_final",
            data_type="test",
            global_step=best_step
        )

        logger.info("===== FINAL TEST RESULT (best dev) ===== %s", _to_json_str(final_test_res))
        print("FINAL_TEST_RESULT(best_dev):", final_test_res)

    # Evaluation (manual eval entry)
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            ckpt_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result_dev = evaluate(args, model, tokenizer, prefix=prefix, data_type="dev", global_step=ckpt_step)
            result_test = evaluate(args, model, tokenizer, prefix=prefix, data_type="test", global_step=ckpt_step)

            if ckpt_step:
                result_dev = {"dev_{}_{}".format(ckpt_step, k): v for k, v in result_dev.items()}
                result_test = {"test_{}_{}".format(ckpt_step, k): v for k, v in result_test.items()}

            results.update(result_dev)
            results.update(result_test)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    # Predict
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
            checkpoints = [x for x in checkpoints if x.split("-")[-1] == str(args.predict_checkpoints)]

        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)

if __name__ == "__main__":
    main()
