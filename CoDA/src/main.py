import argparse
import sys
import logging
import pickle
from functools import partial
import time
from tqdm import tqdm
from collections import Counter
import random
import numpy as np
import time
import os, json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from sslot_losses import SupConLoss, LinearModel
from transformers import AdamW, T5Tokenizer
from t5 import MyT5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, accuracy_score

from data_utils import ABSADataset, task_data_list
from const import *
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores, extract_spans_para
import copy


def _load_complexity_cache(path: str):
    """
    Supports:
      1) JSON file: {raw_sentence: score}
      2) Directory: load & merge all files matching 'complexity*.json'
    """
    if not path:
        return {}

    def _load_one_json(fp: str):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # force float
            return {k: float(v) for k, v in obj.items()}
        except FileNotFoundError:
            logging.warning(f"[complexity] cache not found: {fp}")
            return {}
        except Exception as e:
            logging.warning(f"[complexity] failed to load cache: {fp} ({e})")
            return {}

    # ---- directory mode ----
    if os.path.isdir(path):
        merged = {}
        # auto-detect: complexity*.json (covers dev/dev2/test etc.)
        files = []
        for fn in os.listdir(path):
            if fn.lower().startswith("complexity") and fn.lower().endswith(".json"):
                files.append(os.path.join(path, fn))
        files.sort()

        if not files:
            logging.warning(f"[complexity] no complexity*.json found under dir: {path}")
            return {}

        logging.info(f"[complexity] loading {len(files)} caches from dir: {path}")
        for fp in files:
            cur = _load_one_json(fp)
            for k, v in cur.items():
                if k in merged and merged[k] != v:
                    # stable conflict strategy: keep max (more conservative)
                    logging.warning(
                        f"[complexity] conflict for same sentence in {os.path.basename(fp)}: "
                        f"old={merged[k]}, new={v}, use max"
                    )
                    merged[k] = max(merged[k], v)
                else:
                    merged[k] = v
        return merged

    # ---- file mode (backward compatible) ----
    return _load_one_json(path)
def _extract_stage1_tri_tags(stage1_order_text: str):
    """Extract a robust tri-order tag sequence (A/C/S) from a stage-1 decoded order string.

    We may have multiple views separated by [SSEP]. Instead of using only the first view,
    we take the *most frequent* (A/C/S) order across all views. This reduces noise when
    stage-1 is unstable.
    """
    if not stage1_order_text:
        return []

    views = stage1_order_text.split(' [SSEP] ')
    cand = []
    for v in views:
        tags = []
        for tok in v.split():
            if tok.startswith('[') and tok.endswith(']'):
                t = tok.strip('[]')
                if t in ('A','C','S','O'):
                    tags.append(t)
        tags = [t for t in tags if t in ('A','C','S')]
        if tags:
            cand.append(tuple(tags))

    if not cand:
        return []

    from collections import Counter
    best, _ = Counter(cand).most_common(1)[0]
    return list(best)

def _tri_consistency_score(template_order: str, tri_tags):
    if not tri_tags:
        return 0.0
    toks = [t.strip('[]') for t in template_order.split() if t.startswith('[')]
    pos = {t:i for i,t in enumerate(toks)}
    tags = [t for t in tri_tags if t in pos]
    if len(tags) < 2:
        return 0.0
    total = 0
    ok = 0
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            total += 1
            if pos[tags[i]] < pos[tags[j]]:
                ok += 1
    return ok / total if total else 0.0

def _span_score(template_order: str):
    """A simple structure regularizer for template selection.

    Old version rewarded *larger* tag distances, which often selects weird/unstable templates
    and increases false positives. We instead reward *compact/consistent* layouts by
    returning 1 - normalized_span (higher is better).
    """
    toks = [t.strip('[]') for t in template_order.split() if t.startswith('[')]
    pos = {t:i for i,t in enumerate(toks)}
    if not all(t in pos for t in ('A','O','C','S')):
        return 0.0
    span = abs(pos['A']-pos['O']) + abs(pos['O']-pos['C']) + abs(pos['C']-pos['S'])
    # max possible span is 9 when four tags are at extremes
    return 1.0 - (span / 9.0)

def _select_aux_templates(rest_orders, m, w_entropy, tri_tags=None, w_tri=0.0):
    if not rest_orders or m <= 0:
        return []
    n = len(rest_orders)
    scored = []
    for rank, t in enumerate(rest_orders):
        rank_norm = rank / max(1, n-1)
        entropy_pref = 1.0 - rank_norm
        span = _span_score(t)
        tri = _tri_consistency_score(t, tri_tags) if (tri_tags and w_tri > 0) else 0.0
        score = span + float(w_entropy) * entropy_pref + float(w_tri) * tri
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:m]]

def _vote_quads(list_of_quad_lists, threshold: int):
    if not list_of_quad_lists:
        return []
    c = Counter()
    for quads in list_of_quad_lists:
        for q in quads:
            c[tuple(q)] += 1
    return [list(q) for q, cnt in c.items() if cnt >= threshold]

def _quads_to_canonical_str(quads):
    out = []
    for ac, at, sp, ot in quads:
        out.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")
    return " [SSEP] ".join(out)


# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    #print(f"Random seed set as {seed}")


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_struct_token', action='store_true', help='Use sentence-level structure control token')
    parser.add_argument('--struct_with_l', action='store_true', help='Link STRUCT token with structure-level labels')
    parser.add_argument('--struct_lambda', default=0.05, type=float, help='Weight for structure-level contrastive loss (scheme-2)'

    # basic settings
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument(
        "--task",
        default='asqp',
        choices=["asqp", "acos", "memd"],
        type=str,
        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument(
        "--first_stage_views",
        default='exclude_O',
        choices=["exclude_A", "exclude_C", "exclude_O", "exclude_S", "asqp", "acos", "memd"],
        type=str,
        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument(
        "--dataset",
        default='rest15',
        type=str,
        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument(
        "--eval_data_split",
        default='test',
        choices=["test", "dev"],
        type=str,
    )
    parser.add_argument("--model_name",
                        default='t5-base',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir",
                        default='outputs/temp',
                        type=str,
                        help="Output directory")
    parser.add_argument("--load_ckpt_name",
                        default=None,
                        type=str,
                        help="load ckpt path")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument(
        "--do_inference",
        default=True,
        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    # ---- Auxiliary generation controlled by LLM complexity (inference-time) ----
    parser.add_argument("--enable_complexity_aux",
                        action="store_true",
                        help="Enable auxiliary generation for complex samples, then vote and merge with baseline outputs.")
    parser.add_argument("--complexity_cache_path",
                        type=str,
                        default="",
                        help="Path to JSON cache {raw_sentence: complexity_score in [0,1]}.")
    parser.add_argument("--complexity_threshold",
                        type=float,
                        default=1,
                        help="Run auxiliary generation only if complexity >= this threshold.")
    parser.add_argument("--aux_templates",
                        type=int,
                        default=2,
                        help="Number of auxiliary templates to generate per triggered sample (top-M by span+entropy).")
    parser.add_argument("--aux_vote_threshold",
                        type=int,
                        default=2,
                        help="Vote threshold for auxiliary quads: keep quads predicted by >=K auxiliary generations.")
    parser.add_argument("--aux_entropy_weight",
                        type=float,
                        default=0.25,
                        help="Weight for entropy-rank preference in auxiliary template selection.")
    parser.add_argument("--aux_use_stage1_order",
                        action="store_true",
                        help="Use stage-1 predicted triplet order as a weak preference when selecting auxiliary templates.")
    parser.add_argument("--aux_tri_weight",
                        type=float,
                        default=0.15,
                        help="Weight for tri-order consistency in auxiliary template selection (only if --aux_use_stage1_order).")

    # ---- Complexity-aware inference (tiered) ----
    parser.add_argument("--comp_t1",
                        type=float,
                        default=0,
                        help="Low/Mid boundary for complexity-aware inference. If c < comp_t1, skip aux generation.")
    parser.add_argument("--comp_t2",
                        type=float,
                        default=0,
                        help="Mid/High boundary for complexity-aware inference. If c >= comp_t2, use HIGH aux settings.")
    parser.add_argument("--aux_templates_mid",
                        type=int,
                        default=0,
                        help="Number of auxiliary templates for MID complexity samples (comp_t1 <= c < comp_t2).")
    parser.add_argument("--aux_vote_mid",
                        type=int,
                        default=0,
                        help="Vote threshold for MID complexity samples.")
    parser.add_argument("--aux_templates_high",
                        type=int,
                        default=0,
                        help="Number of auxiliary templates for HIGH complexity samples (c >= comp_t2).")
    parser.add_argument("--aux_vote_high",
                        type=int,
                        default=0,
                        help="Vote threshold for HIGH complexity samples.")
    parser.add_argument("--aux_high_merge_only_add",
                        action="store_true",
                        help="For HIGH complexity samples: only ADD new quads from aux voting (avoid hurting precision).")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=25,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--load_path_cache",
                        action='store_true',
                        help="load decoded path from cache")
    parser.add_argument("--sort_label",
                        action='store_true',
                        help="sort tuple by order of appearance")
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--constrained_decode",
                        action="store_true",
                        help='constrained decoding when evaluating')

    
    # Optional supervised contrastive loss (stage-2 only)
    parser.add_argument("--loss_lambda", type=float, default=0.0,
                        help="Weight for the supervised contrastive loss (stage-2). Set >0 to enable.")
    parser.add_argument("--cont_temp", type=float, default=0.25,
                        help="Temperature for supervised contrastive loss.")
    parser.add_argument("--cont_loss", type=float, default=0.05,
                        help="Scaling factor for supervised contrastive loss.")
    parser.add_argument("--proj_dim", type=int, default=1024,
                        help="Projection dimension for SSLOT contrastive head.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout used to create the second view for SSLOT.")
    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return args

# T5 for First Phase
class T5FineTuner_1(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None,
                output_hidden_states: bool = False):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    def _step(self, batch):
            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

            outputs = self(input_ids=batch["source_ids"],
                        attention_mask=batch["source_mask"],
                        labels=lm_labels,
                        decoder_attention_mask=batch['target_mask'])

            loss = outputs[0]
            return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        #print("evaluating f1...")
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        scores, _, _ = compute_scores(dec, target, verbose=False)
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    task_name=args.task,
                                    data_name=args.dataset,
                                    data_type="train",
                                    n_tuples=0, #dummy
                                    wave=1,
                                    args=self.config,
                                    max_len=self.config.max_seq_length)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=10)
        
        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  task_name=args.task,
                                  data_name=args.dataset,
                                  data_type="dev",
                                  n_tuples=0,
                                  wave=1,
                                  args=self.config,
                                  max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=10)

    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                 input_ids):
        if not os.path.exists('./force_tokens.json'):
            dic = {'special_tokens':[]}
            special_tokens_tokenize_res = []
            for w in ['[O','[A','[S','[C','[SS']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]
        
            dic['special_tokens'] = special_tokens_tokenize_res
            import json
            with open("force_tokens.json", 'w') as f:
                json.dump(dic, f, indent=4)
        
        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
            
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
    
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens['special_tokens']
            
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 
        
        

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [

        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]

        else:
            cur_term = input_ids[last_left_brace_pos + 1]

         
        ret = []
        
        if cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
            
        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            
            for w in force_tokens['special_tokens']:
                ret.discard(w)

            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 

        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        # print("prefix_allowed_tokens_fn: ", ret[0])
        return ret
    
# T5 for Second Phase
class T5FineTuner_2(pl.LightningModule):
    """
    Fine tud_model = getattr(self.model.config, "d_model", 768)
proj_dim = getattr(self.config, "proj_dim", d_model)
ne a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer):



        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer
        d_model = getattr(self.model.config, "d_model", 768)
        proj_dim = getattr(self.config, "proj_dim", d_model)
        # --- Stable-content loss (stage-2 only) ---
        # Goal: make the *content* after each field tag ([A]/[O]/[C]/[S]) more consistent,
        # while NOT constraining the order of these tags.
        self._use_sslot = getattr(self.config, "loss_lambda", 0.0) > 0
        if self._use_sslot:
            d_model = getattr(self.model.config, "d_model", 768)
            proj_dim = getattr(self.config, "proj_dim", d_model)
            self.slot_dropout = nn.Dropout(getattr(self.config, "dropout", 0.1))
            self.slot_proj = nn.Identity() if proj_dim == d_model else nn.Linear(d_model, proj_dim, bias=False)
            self.criterion = SupConLoss(
                temperature=getattr(self.config, "cont_temp", 0.07),
                loss_scaling_factor=getattr(self.config, "cont_loss", 1.0),
            )

        # Cache special tag token ids. If any tag is not a single token, fall back to encode().
        def _tok_id(tok: str):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid == self.tokenizer.unk_token_id:
                ids = self.tokenizer.encode(tok, add_special_tokens=False)
                return ids[0] if len(ids) > 0 else None
            return tid

        self._slot_token_ids = {
            "A": _tok_id("[A]"),
            "O": _tok_id("[O]"),
            "C": _tok_id("[C]"),
            "S": _tok_id("[S]"),
        }
        self._slot_label_map = {"A": 0, "O": 1, "C": 2, "S": 3}


        # --- Structure-level contrastive branch (scheme-2): sentence-level structure control token ---
        # This is intentionally separate from slot-level ([A]/[O]/[C]/[S]) stable-content loss.
        self._use_struct_l = bool(getattr(self.config, "struct_with_l", False)) and (getattr(self.config, "struct_lambda", 0.0) > 0)
        self._struct_lambda = float(getattr(self.config, "struct_lambda", 0.0))
        if self._use_struct_l:
            # Prefer the 3 structure-type tokens if present; fall back to a generic [STRUCT] if user adds it.
            self._struct_token_ids = {
                "IMPLICIT": _tok_id("[STRUCT_IMPLICIT]"),
                "EXPLICIT": _tok_id("[STRUCT_EXPLICIT]"),
                "MIXED": _tok_id("[STRUCT_MIXED]"),
                "STRUCT": _tok_id("[STRUCT]"),
            }
            # label map for SupCon
            self._struct_label_map = {"IMPLICIT": 0, "EXPLICIT": 1, "MIXED": 2, "STRUCT": 0}
            # separate projection head (kept same dims as slot head by default)
            self.struct_dropout = nn.Dropout(getattr(self.config, "dropout", 0.1))
            self.struct_proj = nn.Identity() if proj_dim == d_model else nn.Linear(d_model, proj_dim, bias=False)
    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
    def _step(self, batch):
        #lm_labels = batch["target_ids"]
        #lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        lm_labels = batch["target_ids"].clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            output_hidden_states=self._use_sslot,
            return_dict=True,
        )

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Add optional stable-content loss on field tags
        # This does NOT constrain the tag order; it only encourages consistent representations
        # for content anchored at [A]/[O]/[C]/[S] across the batch.
        if self._use_sslot:
            # We need decoder hidden states aligned with target token positions.
            dec_h = getattr(outputs, "decoder_hidden_states", None)
            if dec_h is not None:
                dec_last = dec_h[-1]  # [B, T, H]
                target_ids = batch["target_ids"]

                feats_all = []
                labs_all = []

                device = dec_last.device
                for slot, tok_id in self._slot_token_ids.items():
                    if tok_id is None:
                        continue
                    mask = (target_ids == tok_id)
                    if mask.any():
                        feats = dec_last[mask]  # [N, H]
                        # two stochastic views via dropout
                        v1 = self.slot_proj(self.slot_dropout(feats))
                        v2 = self.slot_proj(self.slot_dropout(feats))
                        feat2 = torch.stack([v1, v2], dim=1)  # [N, 2, D]
                        feats_all.append(feat2)
                        labs_all.append(torch.full((feats.size(0),), self._slot_label_map[slot], device=device, dtype=torch.long))

                if len(feats_all) > 0:
                    feats_all = torch.cat(feats_all, dim=0)  # [N, 2, D]
                    labs_all = torch.cat(labs_all, dim=0)    # [N]
                    if feats_all.size(0) >= 2:
                        loss = self.criterion(F.normalize(feats_all, p=2, dim=2), labs_all)
                        loss = loss + self.config.loss_lambda * loss
                                        
                        
                        log_name = "train/loss" if self.training else "val/loss"
                        self.log(
                            log_name,
                            loss,
                            prog_bar=True,
                            on_step=self.training,
                            on_epoch=not self.training,
                            logger=True,
                        )

        # --- Structure-level contrastive branch (scheme-2) ---
        # Use the decoder hidden state at the first STRUCT token position in each sample as a sentence-level representation.
        if getattr(self, "_use_struct_l", False) and getattr(self, "_struct_lambda", 0.0) > 0:
            struct_feats = []
            struct_labs = []
            # build quick lookup: token_id -> label
            id2lab = {}
            for k, tid in getattr(self, "_struct_token_ids", {}).items():
                if tid is not None:
                    id2lab[tid] = getattr(self, "_struct_label_map", {}).get(k, 0)

            # one STRUCT token per sample (first occurrence)
            for b in range(target_ids.size(0)):
                row = target_ids[b]
                # find first matching position
                pos = None
                lab = None
                for t in range(row.size(0)):
                    tid = int(row[t].item())
                    if tid in id2lab:
                        pos = t
                        lab = id2lab[tid]
                        break
                if pos is not None:
                    struct_feats.append(dec_last[b, pos])  # [H]
                    struct_labs.append(lab)

            if len(struct_feats) >= 2:
                struct_feats = torch.stack(struct_feats, dim=0)  # [N, H]
                struct_labs = torch.tensor(struct_labs, device=device, dtype=torch.long)  # [N]
                sv1 = self.struct_proj(self.struct_dropout(struct_feats))
                sv2 = self.struct_proj(self.struct_dropout(struct_feats))
                s_feat2 = torch.stack([sv1, sv2], dim=1)  # [N, 2, D]
                struct_loss = self.criterion(F.normalize(s_feat2, p=2, dim=2), struct_labs)
                loss = loss + self._struct_lambda * struct_loss

                # log struct loss
                s_log_name = "train/struct_loss" if self.training else "val/struct_loss"
                self.log(
                    s_log_name,
                    struct_loss,
                    prog_bar=False,
                    on_step=self.training,
                    on_epoch=True,
                    logger=True,
                )



        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        #print("evaluating f1...")
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        print("dec: ", dec[0])
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        scores, _, _ = compute_scores(dec, target, verbose=False)
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    task_name=args.task,
                                    data_name=args.dataset,
                                    data_type="train",
                                    n_tuples=0,
                                    wave=2,
                                    args=self.config,
                                    max_len=self.config.max_seq_length)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=10)
        
        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  task_name=args.task,
                                  data_name=args.dataset,
                                  data_type="dev",
                                  n_tuples=0,
                                  wave=2,
                                  args=self.config,
                                  max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=10)

    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                 input_ids):

        if not os.path.exists('./force_tokens_full.json'):
            dic = {"cate_tokens":{}, "all_tokens":{}, "sentiment_tokens":{}, 'special_tokens':[]}
            for task in force_words.keys():
                dic["all_tokens"][task] = {}
                for dataset in force_words[task].keys():
                    cur_list = force_words[task][dataset]
                    tokenize_res = []
                    for w in cur_list:
                        tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0])
                    dic["all_tokens"][task][dataset] = tokenize_res
                    # all_tokens = {task: {dataset: cate + paraphrased sentiment + [SSEP], ...}, ...}
            for k,v in cate_list.items():
                tokenize_res = []
                for w in v:
                    tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
                dic["cate_tokens"][k] = tokenize_res
                # cate_tokens = {dataset: cate, ...}
            sp_tokenize_res = []
            for sp in ['great', 'ok', 'bad']:
                sp_tokenize_res.extend(self.tokenizer(sp, return_tensors='pt')['input_ids'].tolist()[0])
            for task in force_words.keys():
                dic['sentiment_tokens'][task] = sp_tokenize_res
            # sentiment_tokens = sp_tokenize_res
            special_tokens_tokenize_res = []
            for w in ['[O','[A','[S','[C','[SS']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]

            dic['special_tokens'] = special_tokens_tokenize_res
            import json
            with open("force_tokens_full.json", 'w') as f:
                json.dump(dic, f, indent=4)
        
        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
            'it': [34],
            'null': [206,195]
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens_full['special_tokens']
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['EP'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
        else:
            cur_term = input_ids[last_left_brace_pos + 1]

        ret = []
        if cur_term in to_id['SP']:  # SP
            ret = force_tokens_full['sentiment_tokens'][task]
        elif cur_term in to_id['AT']:  # AT
            force_list = source_ids[batch_id].tolist()
            force_list.extend(to_id['it'] + [1])  
            ret = force_list  
        elif cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
        elif cur_term in to_id['AC']:  # AC
            ret = force_tokens_full['cate_tokens'][data_name]
        elif cur_term in to_id['OT']:  # OT
            force_list = source_ids[batch_id].tolist()
            if task in ["acos", "memd"]:
                force_list.extend(to_id['null'])  # null
            ret = force_list
        else:
            raise ValueError(cur_term)

        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            for w in force_tokens_full['special_tokens']:
                ret.discard(w)
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        return ret


def evaluate(model1, model2, task, data, data_type):
    """
    Compute scores given the predictions and gold labels
    """
    tasks, datas, sents, _ = read_line_examples_from_file(
        f'../data/{task}/{data}/{data_type}.txt', data_type, task, data, lowercase=False)
    
    # ===== load complexity by idx (jsonl) =====
    complexity = {}

    comp_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", task, data, f"complexity_{data_type}_by_idx.jsonl"
    )
    comp_path = os.path.abspath(comp_path)

    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                complexity[str(rec["idx"])] = rec["complexity"]
        print(f"[complexity] loaded idx-based complexity from {comp_path}")
    else:
        print(f"[complexity] not found: {comp_path}")


    orders, order_targets, outputs, targets, probs = [], [], [], [], []
    cache_file = os.path.join(
        args.output_dir, "result_{}{}{}_{}_path{}_beam{}.pickle".format(
            "best_" if args.load_ckpt_name else "",
            "cd_" if args.constrained_decode else "", task, data, 5,
            args.beam_size))
    if args.load_path_cache:
        with open(cache_file, 'rb') as handle:
            (outputs, targets, probs) = pickle.load(handle)
    else:
        dataset1 = ABSADataset(model1.tokenizer,
                              task_name=task,
                              data_name=data,
                              data_type=data_type,
                              args=args,
                              n_tuples=0,
                              wave=1,
                              max_len=args.max_seq_length)
        data_loader1 = DataLoader(dataset1,
                                 batch_size=args.eval_batch_size,
                                 num_workers=10)
        for i in range(0, 24):
            data_sample = dataset1[i]
            print(
                'Input1 :',
                model1.tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            '''print('Input :',
                  model.tokenizer.convert_ids_to_tokens(data_sample['source_ids']))'''
            print(
                'Output1:',
                model1.tokenizer.decode(data_sample['target_ids'],
                                 skip_special_tokens=True))
            print()
            
        device = torch.device('cuda:0')
        model1.model.to(device)
        model1.model.eval()

        for batch in data_loader1:
            # beam search
            outs = model1.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=1,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    model1.prefix_allowed_tokens_fn, task, data,
                    batch['source_ids']) if args.constrained_decode else None,
            )
            dec = [
                model1.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs.sequences
            ]
            target = [
                model1.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["target_ids"]
            ]
            orders.extend(dec)
            order_targets.extend(target)

        # first stage performance evaluation
        orders_texts = list(orders)
        orders = [len(order.split(' [SSEP] ')) for order in orders]
        order_targets = [len(order.split(' [SSEP] ')) for order in order_targets]
        mse = mean_squared_error(orders, order_targets)   
        rmse = round(np.sqrt(mse), 2)
        accuracy = accuracy_score(orders, order_targets)
        with open("my_order_score.txt", 'a') as file:
            file.write(f"first_stage_result: {task}, {data}, RMSE : " + str(rmse) + '\n')
            file.write(f"first_stage_result: {task}, {data}, Acc : " + str(accuracy) + '\n')
        
        dataset2 = ABSADataset(model2.tokenizer,
                              task_name=task,
                              data_name=data,
                              data_type=data_type,
                              args=args,
                              n_tuples=orders,
                              wave=2,
                              max_len=args.max_seq_length)
        data_loader2 = DataLoader(dataset2,
                                 batch_size=args.eval_batch_size,
                                 num_workers=10)
        
        for i in range(0, 24):
            data_sample = dataset2[i]
            print(
                'Input_evaluate: ',
                model2.tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            print()
        model2.model.to(device)
        model2.model.eval()

        complexity_cache = _load_complexity_cache(args.complexity_cache_path) if args.enable_complexity_aux else {}

        for batch in data_loader2:
            # beam search
            outs = model2.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=1,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    model2.prefix_allowed_tokens_fn, task, data,
                    batch['source_ids']) if args.constrained_decode else None,
            )
            dec = [
                model2.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs.sequences
            ]

            # ---- complexity-controlled auxiliary generation + vote + merge (inference-time only) ----
            if args.enable_complexity_aux:
                # per-sample processing (kept simple; can be optimized later)
                for bi in range(len(dec)):
                    raw_sent = batch.get("raw_sentence", [None]*len(dec))[bi]
                    ex_id = int(batch.get("example_id", [bi]*len(dec))[bi])
                    if not raw_sent:
                        continue
                    
                    # ---- get complexity (idx-based preferred) ----
                    comp_val = None
                    try:
                        # 'complexity' is idx-based dict loaded earlier from complexity_{split}_by_idx.jsonl
                        comp_val = complexity.get(str(ex_id), None) if isinstance(complexity, dict) else None
                    except Exception:
                        comp_val = None
                    if comp_val is None:
                        comp_val = complexity_cache.get(raw_sent, 0.5)
                    comp = float(comp_val)

                    # ---- tiered complexity-aware inference: Low / Mid / High ----
                    if comp < float(args.comp_t1):
                        continue  # low complexity: skip aux generation

                    if comp < float(args.comp_t2):
                        m_aux = int(args.aux_templates_mid)
                        vote_k = int(args.aux_vote_mid)
                        high_mode = False
                    else:
                        m_aux = int(args.aux_templates_high)
                        vote_k = int(args.aux_vote_high)
                        high_mode = True

                    # baseline predicted view count from stage-1 (used to slice entropy-ordered templates)
                    base_k = int(orders[ex_id]) if ex_id < len(orders) else 1
                    # full entropy-ordered templates for this example (from data_utils.get_orders)
                    all_tmpls = dataset2.all_orders[ex_id] if hasattr(dataset2, "all_orders") else []
                    rest_tmpls = all_tmpls[base_k:] if all_tmpls else []

                    tri_tags = _extract_stage1_tri_tags(orders_texts[ex_id]) if args.aux_use_stage1_order and ex_id < len(orders_texts) else []
                    aux_tmpls = _select_aux_templates(
                        rest_orders=rest_tmpls,
                        m=m_aux,
                        w_entropy=float(args.aux_entropy_weight),
                        tri_tags=tri_tags,
                        w_tri=float(args.aux_tri_weight) if args.aux_use_stage1_order else 0.0,
                    )

                    if not aux_tmpls:
                        continue

                    aux_quad_lists = []
                    for t in aux_tmpls:
                        aux_inp = " ".join(raw_sent.split() + t.split())
                        enc = model2.tokenizer(
                            aux_inp,
                            return_tensors="pt",
                            truncation=True,
                            max_length=args.max_seq_length,
                        )
                        aux_out = model2.model.generate(
                            input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device),
                            max_length=args.max_seq_length,
                            num_beams=1,
                            early_stopping=True,
                            prefix_allowed_tokens_fn=partial(
                                model2.prefix_allowed_tokens_fn, task, data, enc["input_ids"]
                            ) if args.constrained_decode else None,
                        )
                        aux_dec = model2.tokenizer.decode(aux_out[0], skip_special_tokens=True)
                        aux_quads = extract_spans_para(seq=aux_dec, seq_type="pred")
                        aux_quad_lists.append(aux_quads)

                    voted_aux = _vote_quads(aux_quad_lists, threshold=vote_k)
                    if not voted_aux:
                        continue

                    base_quads = extract_spans_para(seq=dec[bi], seq_type="pred")

                    # ---- precision guard: only keep aux quads that overlap with base (reduce systematic FP) ----
                    if voted_aux and base_quads:
                        base_aspects = set(q[1] for q in base_quads)
                        base_cats = set(q[0] for q in base_quads)
                        filtered = []
                        for q in voted_aux:
                            try:
                                ac, at, sp, ot = q
                            except Exception:
                                continue
                            if (at in base_aspects) or (ac in base_cats):
                                filtered.append(q)
                        voted_aux = filtered

                    if not voted_aux:
                        continue


                    # ---- merge strategy ----
                    if high_mode and bool(getattr(args, "aux_high_merge_only_add", False)):
                        # HIGH complexity: conservative merge (only add new quads)
                        merged = list(base_quads)
                        seen = set(map(tuple, base_quads))
                        for q in voted_aux:
                            tq = tuple(q)
                            if tq not in seen:
                                seen.add(tq)
                                merged.append(q)
                    else:
                        # MID complexity: aggressive merge + dedup
                        merged = []
                        seen = set()
                        for q in base_quads + voted_aux:
                            tq = tuple(q)
                            if tq not in seen:
                                seen.add(tq)
                                merged.append(q)

                    dec[bi] = _quads_to_canonical_str(merged)

            target = [
                model2.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["target_ids"]
            ]
            outputs.extend(dec)
            targets.extend(target)
            
        # save outputs and targets
        with open(cache_file, 'wb') as handle:
            pickle.dump((outputs, targets, probs), handle)
    
    _outputs = outputs # backup
    outputs = [] # new outputs
    new_targets = [] # new targets
    for i in range(len(targets)):
        output_quads = []
        output_quads.extend(extract_spans_para(seq=_outputs[i], seq_type='pred'))
        # recover output
        output = []
        target = []
        for q in output_quads:
            ac, at, sp, ot = q

            if tasks[i] in ["asqp", "acos", "memd"]:
                output.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")

            else:
                raise NotImplementedError
        new_targets.append(targets[i])
        
        # if no output, use the first path
        output_str = " [SSEP] ".join(
            output)
        outputs.append(output_str)

            
    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    #print("pred labels count", labels_counts)
    scores, all_labels, all_preds = compute_scores(outputs,
                                                   new_targets,
                                                   verbose=True)

    # ===== dump prediction / gold / complexity to jsonl =====
    dump_path = os.path.join(
        args.output_dir,
        f"pred_gold_complexity_{task}_{data}_{data_type}.jsonl"
    )

    with open(dump_path, "w", encoding="utf-8") as f:
        for i in range(len(outputs)):
            sent = sents[i]
            # ===== 兜底：sent 可能是 list =====
            if isinstance(sent, list):
                sent_str = sent[0]
            else:
                sent_str = sent

            rec = {
            "idx": i,
            "split": data_type,
            "sentence": sent_str,          # 和 test.txt 顺序、文本完全一致
            "complexity": complexity.get(str(i), None),
            "gold_text": new_targets[i],
            "pred_text": outputs[i],
            "gold_quads": all_labels[i],
            "pred_quads": all_preds[i],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[dump] pred/gold/complexity saved to {dump_path}")

    print("targets: ", new_targets[:20])
    print("outputs: ", outputs[:20])
    
    #print("scores: ", scores)
    return scores


def train_function(args):

    # training process
    if args.do_train:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        if args.use_struct_token:
            tokenizer.add_special_tokens({'additional_special_tokens': ['[STRUCT_IMPLICIT]', '[STRUCT_EXPLICIT]', '[STRUCT_MIXED]']})

        # sanity check
        dataset = ABSADataset(tokenizer=tokenizer,
                              task_name=args.task,
                              data_name=args.dataset,
                              data_type='train',
                              n_tuples=0,
                              wave=1,
                              args=args,
                              max_len=args.max_seq_length)
        dataset2 = ABSADataset(tokenizer=tokenizer,
                              task_name=args.task,
                              data_name=args.dataset,
                              data_type='train',
                              n_tuples=0,
                              wave=2,
                              args=args,
                              max_len=args.max_seq_length)
        for i in range(0, 10):
            data_sample = dataset[i]
            print(
                'Input_train1: ',
                tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            print(
                'Target_train: ',
                tokenizer.decode(data_sample['target_ids'],
                                 skip_special_tokens=True))
            print()
            data_sample2 = dataset2[i]
            print(
                'Input_train2: ',
                tokenizer.decode(data_sample2['source_ids'],
                                 skip_special_tokens=True))
            print(
                'Target_train2: ',
                tokenizer.decode(data_sample2['target_ids'],
                                 skip_special_tokens=True))
            print()
        print("\n****** Conduct Training ******")

        # initialize the T5 model
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name)
        if args.use_struct_token:
            tfm_model.resize_token_embeddings(len(tokenizer))  
        model_1 = T5FineTuner_1(args, tfm_model, tokenizer)
        # load data
        train_loader_1 = model_1.train_dataloader()
        # config optimizer
        t_total = ((len(train_loader_1.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(args.num_train_epochs))

        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total
        }

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=False)

        early_stop_callback = EarlyStopping(monitor="val_f1",
                                            min_delta=0.00,
                                            patience=20,
                                            verbose=True,
                                            mode="max")
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # prepare for trainer
        train_params = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[
                checkpoint_callback, early_stop_callback,
                TQDMProgressBar(refresh_rate=10), lr_monitor
            ],
        )

        trainer = pl.Trainer(**train_params)
        
        trainer.fit(model_1)
        
        print("\n****** Phase 1 finished! ******")
                
        tfm_model2 = copy.deepcopy(tfm_model) # Reuse First Model
        
        model_2 = T5FineTuner_2(args, tfm_model2, tokenizer)
        
        train_loader_2 = model_2.train_dataloader()
        
        t_total2 = ((len(train_loader_2.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(40))
        
        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total2
        }

        checkpoint_callback2 = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=False)

        early_stop_callback2 = EarlyStopping(monitor="val_f1",
                                            min_delta=0.00,
                                            patience=20,
                                            verbose=True,
                                            mode="max")
        lr_monitor2 = LearningRateMonitor(logging_interval='step')

        # prepare for trainer
        train_params2 = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=40,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[
                checkpoint_callback2, early_stop_callback2,
                TQDMProgressBar(refresh_rate=10), lr_monitor2
            ],
        )
        trainer2 = pl.Trainer(**train_params2)
        trainer2.fit(model_2)
        print("\n****** Phase 2 finished! ******")

        # save best stage-2 model from checkpoint
        best_ckpt = trainer2.checkpoint_callback.best_model_path

        if best_ckpt:
            print("Best stage-2 checkpoint:", best_ckpt)

            best_model = T5FineTuner_2.load_from_checkpoint(
                best_ckpt,
                args=args,
                tokenizer=tokenizer,
            )
            best_dir = os.path.join(args.output_dir, "best_stage2")
            os.makedirs(best_dir, exist_ok=True)
            best_model.model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            print("[WARN] best_model_path is empty, BEST model will not be saved.")



        # save the final model
        model_1.model.save_pretrained(os.path.join(args.output_dir, "first"))
        model_2.model.save_pretrained(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "first"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print("Finish training and saving the model!")

    scores_last = None
    scores_best = None

    if args.do_inference:
            print("\n****** Conduct inference on trained checkpoint ******")

            # initialize the T5 model from previous checkpoint
            #print(f"Load trained model from {args.output_dir}")
            '''print(
            'Note that a pretrained model is required and `do_true` should be False'
            )'''
            model_path1 = os.path.join(args.output_dir, "first")
            model_path2 = os.path.join(args.output_dir, "final")
            # model_path = args.model_name_or_path  # for loading ckpt

            tokenizer = T5Tokenizer.from_pretrained(model_path2)
            tfm_model1 = MyT5ForConditionalGeneration.from_pretrained(model_path1)
            tfm_model2 = MyT5ForConditionalGeneration.from_pretrained(model_path2)
            model1 = T5FineTuner_1(args, tfm_model1, tokenizer)
            model2_last = T5FineTuner_2(args, tfm_model2, tokenizer)

            # load best stage-2 model if exists
            best_stage2_dir = os.path.join(args.output_dir, "best_stage2")
            model2_best = None

            def _is_hf_dir_ok(d: str) -> bool:
                if not os.path.isdir(d):
                    return False
                has_config = os.path.isfile(os.path.join(d, "config.json"))
                has_weights = (
                    os.path.isfile(os.path.join(d, "pytorch_model.bin")) or
                    os.path.isfile(os.path.join(d, "model.safetensors"))
                )
                return has_config and has_weights

            if _is_hf_dir_ok(best_stage2_dir):
                tfm_best = MyT5ForConditionalGeneration.from_pretrained(best_stage2_dir)
                model2_best = T5FineTuner_2(args, tfm_best, tokenizer)
            else:
                print(f"[WARN] best_stage2 not found or incomplete: {best_stage2_dir}. Skip BEST inference.")


            if args.load_ckpt_name:
                ckpt_path = os.path.join(args.output_dir, args.load_ckpt_name)
                #print("Loading ckpt:", ckpt_path)
                checkpoint = torch.load(ckpt_path)
                model1.load_state_dict(checkpoint["state_dict"])

            log_file_path = os.path.join(args.output_dir, "result.txt")
            time0 = time.time()
            # compute the performance scores
            with open(log_file_path, "a+") as f:
                config_str = f"seed: {args.seed}, beam: {args.beam_size}, constrained: {args.constrained_decode}\n"
                #print(config_str)
                f.write(config_str)
                scores_last = evaluate(model1,
                                    model2_last,
                                    args.task,
                                    args.dataset,
                                    data_type=args.eval_data_split)

                if model2_best is not None:
                    scores_best = evaluate(model1,
                                        model2_best,
                                        args.task,
                                        args.dataset,
                                        data_type=args.eval_data_split)
                else:
                    scores_best = None
                

                exp_last = "{} LAST precision: {:.2f} recall: {:.2f} F1 = {:.2f}".format(
                    args.eval_data_split, scores_last['precision'], scores_last['recall'], scores_last['f1'])
                print(exp_last)
                f.write(exp_last + "\n")
                if scores_best is not None:
                    exp_best = "{} BEST precision: {:.2f} recall: {:.2f} F1 = {:.2f}".format(
                        args.eval_data_split, scores_best['precision'], scores_best['recall'], scores_best['f1'])
                    print(exp_best)
                    f.write(exp_best + "\n")
                f.flush()
            # Report Inference time    
            infer_time = time.time() - time0
            with open("inference_time.txt", 'a') as file:
                file.write(str(infer_time) + '\n')
    result = {}
    if args.do_inference:
        result['last'] = scores_last
        result['best'] = scores_best
    return result



if __name__ == '__main__':
    args = init_args()
    set_seed(args.seed)
    train_function(args)