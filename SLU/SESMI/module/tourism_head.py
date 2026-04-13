import torch
import torch.nn as nn
import torch.nn.functional as F
from module.lowRankBilinear import *

class JointPredictor(nn.Module):
    """
    论文核心组件：Co-Predictor Layer
    融合 MLP 和 Biaffine 机制，增强边界特征
    """
    def __init__(self, hidden_size):
        super(JointPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # self.biffine = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.lowRankBilinear = LowRankBilinear(hidden_size, hidden_size, hidden_size, rank = 128)
        self.bilin_scale = 0.5
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # MLP 分支
        mlp_output = self.mlp(x)
        # Biaffine 分支 (计算自相关性)
        biffine_output = self.lowRankBilinear(x, x)* self.bilin_scale
        out = x + mlp_output + biffine_output               # ✅ 把输入也做残差（更稳）
        out = self.ln(out)
        out = self.dropout(out)
        return out

class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.ln(hidden_states)
        x = self.drop(x)
        return self.dense(x)

    
class PoolerEndLogits(nn.Module):
    """预测 End 位置 (结合 Start 的信息)"""
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size + num_classes, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_logits):
        x = self.dense_0(torch.cat([hidden_states, start_logits], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

class BiLSTMEncoder(nn.Module):
    """
    输入:  x [B, W, H], word_mask [B, W] (1=有效word, 0=padding)
    输出:  y [B, W, H]
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size 必须是偶数"
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, word_mask):
        # word_mask: [B,W] -> bool
        if word_mask.dtype != torch.bool:
            word_mask = word_mask.bool()

        lengths = word_mask.long().sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        out = self.dropout(out)
        out = self.ln(out)

        # padding 位置清零，避免污染后续分类
        out = out.masked_fill(~word_mask.unsqueeze(-1), 0.0)
        return out
