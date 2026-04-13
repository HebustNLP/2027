import torch
import torch.nn as nn
from torch.nn import functional as F

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.proj(x)
        h = F.gelu(h)
        h = self.dropout(h)
        logits = self.out(h)
        return logits
    

class IntentCountPredictor(nn.Module):
    def __init__(self, hidden_dim, max_k=3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, max_k)
        )

    def forward(self, cls_emb):
        """
        cls_emb: [B, hidden_dim]
        return: count_logits [B, max_k]
        """
        return self.classifier(cls_emb)


