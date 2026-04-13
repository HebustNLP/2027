import torch
import torch.nn as nn
from module.wordRep import *
from module.module import *
from module.tourism_head import JointPredictor, PoolerStartLogits, PoolerEndLogits, BiLSTMEncoder
from torch.nn import functional as F

class JointModel(nn.Module):
    def __init__(self, args, num_intent_labels, num_slot_labels):
        super().__init__()
        self.args = args
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels

        self.wordrep = WordRep(args)
        hidden_size = self.wordrep.bert.config.hidden_size

        # Soft Intent
        self.soft_intent_classifier = IntentClassifier(
            input_dim=hidden_size,
            num_labels=num_intent_labels,
            hidden_dim=args.hidden_dim_ffw,
            dropout_rate=args.dropout_rate
        )
        # Hard Intent
        hard_input_dim = hidden_size + self.num_slot_labels 
        self.hard_intent_classifier = IntentClassifier(
            input_dim=hard_input_dim,
            num_labels=num_intent_labels,
            hidden_dim=args.hidden_dim_ffw,
            dropout_rate=args.dropout_rate
        )
        self.intent_count_predictor = IntentCountPredictor(
            hidden_dim=hard_input_dim,
            max_k=args.max_intent_num
        )
        slot_input_dim = hidden_size
        self.dropout = nn.Dropout(args.dropout_rate)
        self.joint_predictor = JointPredictor(hidden_size)
        self.seq_encoder = BiLSTMEncoder(hidden_size, dropout=0.1)
        self.start_fc = PoolerStartLogits(slot_input_dim, num_slot_labels)
        self.end_fc = PoolerEndLogits(slot_input_dim, num_slot_labels)
    def forward(self, input_ids, attention_mask, word_attention_mask, words_lengths, 
                start_positions=None):
        
        # 1. BERT 编码
        cls_output, context_embedding = self.wordrep(
            input_ids, attention_mask, word_attention_mask, words_lengths
        )

        soft_intent_logits = self.soft_intent_classifier(cls_output)

        word_mask = (words_lengths > 0)
        word_len = (words_lengths > 0).long().sum(dim=1).clamp(min=1) 
        
        sequence_output = self.joint_predictor(context_embedding)
        sequence_output = self.seq_encoder(sequence_output, word_mask)

        # C. 预测 Start
        start_logits = self.start_fc(sequence_output)

        # D. 预测 End (Teacher Forcing)
        if self.training and start_positions is not None:
            batch_size, seq_len, _ = start_logits.size()
            label_logits = torch.zeros(batch_size, seq_len, self.num_slot_labels).to(input_ids.device)
            index = start_positions.unsqueeze(-1)
            label_logits = label_logits.scatter(2, index, 1.0).detach()
        else:
            label_logits = F.softmax(start_logits, -1)
        end_logits = self.end_fc(sequence_output, label_logits)

        # ==================== Hard Intent (Slot -> Intent) ====================
        # 1. 提取槽位特征 
        slot_start_features = self.get_slot_features_max(start_logits, word_mask)
        slot_end_features = self.get_slot_features_max(end_logits, word_mask).detach()
        slot_features = (slot_start_features + slot_end_features)/2
        # 2. 拼接
        intent_feature_concat = torch.cat([cls_output, slot_features], dim=-1)
        
        # 3. 最终意图
        hard_intent_logits = self.hard_intent_classifier(intent_feature_concat)
        count_logits = self.intent_count_predictor(intent_feature_concat)

        return soft_intent_logits, hard_intent_logits, start_logits, end_logits, count_logits

    def get_slot_features_max(self, slot_logits, attention_mask):
        seq_len = slot_logits.size(1)
        mask = attention_mask[:, :seq_len].unsqueeze(-1).float()   # [B, L, 1]
        slot_probs = F.softmax(slot_logits, dim=-1)
        slot_probs = slot_probs             # [B, L, C]
        slot_probs = slot_probs * mask
        slot_features, _ = torch.max(slot_probs, dim=1)            # [B, C]
        return slot_features
