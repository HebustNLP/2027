import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class JointPredictor(nn.Module):
    """
    Token-wise (same-position) bilinear interaction.
    NOTE: This does NOT build an LxL span score matrix; it produces [B,L,H] features.
    """
    def __init__(self, hidden_size: int):
        super(JointPredictor, self).__init__()
        self.ffn_start = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.ffn_end = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.biffine = nn.Bilinear(hidden_size, hidden_size, hidden_size)

    def forward(self, start_representations, end_representations):
        start_h = self.ffn_start(start_representations)
        end_h = self.ffn_end(end_representations)
        return self.biffine(start_h, end_h)


class BertSpanForNer(BertPreTrainedModel):
    """
    Span-based NER:
      - start head predicts start label per token
      - end head predicts end label per token, conditioned on start information
    Extension (Scheme B):
      - encode explanation with the same BERT
      - cross-attention: text queries attend to explanation key/value
      - gated residual fusion to inject explanation signal
    """
    def __init__(self, config):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Cross-attention: text attends to explanation
        # Use the same hidden size as BERT.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=getattr(config, "num_attention_heads", 12),
            dropout=getattr(config, "attention_probs_dropout_prob", config.hidden_dropout_prob),
            batch_first=True,
        )
        self.cross_attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_attn_ln = nn.LayerNorm(config.hidden_size)

        # Gate: per-token scalar in (0,1), decides how much explanation info to inject
        self.cross_attn_gate = nn.Linear(config.hidden_size * 2, 1)
        # Start conservatively: initial gate ~ sigmoid(-2)=0.12
        nn.init.constant_(self.cross_attn_gate.bias, -2.0)

        # Optional extra encoder
        self.bilstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Heads
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)

        self.joint_predictor = JointPredictor(config.hidden_size)

        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        expl_input_ids=None,
        expl_token_type_ids=None,
        expl_attention_mask=None,
    ):
        # -------- Text encoding --------
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # [B,L,H]

        # -------- Explanation encoding + Cross-Attention fusion --------
        # If explanation not provided, we skip fusion (backward compatible).
        if expl_input_ids is not None:
            if expl_token_type_ids is None:
                expl_token_type_ids = torch.zeros_like(expl_input_ids)
            if expl_attention_mask is None:
                expl_attention_mask = torch.ones_like(expl_input_ids)

            expl_outputs = self.bert(
                input_ids=expl_input_ids,
                attention_mask=expl_attention_mask,
                token_type_ids=expl_token_type_ids
            )
            expl_sequence_output = expl_outputs[0]  # [B,Le,H]

            # key_padding_mask: True means "ignore this position"
            key_padding_mask = (expl_attention_mask == 0) if expl_attention_mask is not None else None

            attn_out, _ = self.cross_attn(
                query=sequence_output,
                key=expl_sequence_output,
                value=expl_sequence_output,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )  # [B,L,H]

            # Gated residual fusion
            gate = torch.sigmoid(
                self.cross_attn_gate(torch.cat([sequence_output, attn_out], dim=-1))
            )  # [B,L,1]
            attn_out = self.cross_attn_dropout(attn_out)
            sequence_output = self.cross_attn_ln(sequence_output + gate * attn_out)

        # -------- Downstream encoder --------
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.bilstm(sequence_output)

        # -------- Joint predictor (token-wise feature enhancement) --------
        sequence_output = sequence_output + self.joint_predictor(sequence_output, sequence_output)

        # -------- Heads --------
        start_logits = self.start_fc(sequence_output)

        # Build label features for end head
        if start_positions is not None and self.training:
            if self.soft_label:
                # one-hot for gold start labels
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.zeros(batch_size, seq_len, self.num_labels, device=input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1.0)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        # -------- Loss --------
        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()

            start_logits_flat = start_logits.view(-1, self.num_labels)
            end_logits_flat = end_logits.view(-1, self.num_labels)

            # NOTE: This keeps your original masking behavior.
            # If you later switch to ignore_index=-100, you should mask on labels != -100 instead.
            active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else torch.ones(
                start_logits_flat.size(0), dtype=torch.bool, device=start_logits_flat.device
            )

            active_start_logits = start_logits_flat[active_loss]
            active_end_logits = end_logits_flat[active_loss]
            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2.0
            outputs = (total_loss,) + outputs

        return outputs
