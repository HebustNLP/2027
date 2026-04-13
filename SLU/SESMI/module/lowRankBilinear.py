import torch
import torch.nn as nn


class LowRankBilinear(nn.Module):
    """
    Low-Rank Bilinear Layer
    实现：f(x, y) = (x W1 ⊙ y W2) W3

    输入:
        x: [B, L, in1]
        y: [B, L, in2]
    输出:
        out: [B, L, out]
    """

    def __init__(self, in1_features, in2_features, out_features, rank=128, bias=True):
        super().__init__()

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.rank = rank

        # 低秩分解
        self.linear1 = nn.Linear(in1_features, rank, bias=False)
        self.linear2 = nn.Linear(in2_features, rank, bias=False)
        self.linear_out = nn.Linear(rank, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, x, y):
        """
        x: [B, L, in1_features]
        y: [B, L, in2_features]
        """

        # 投影到低秩空间
        x_proj = self.linear1(x)   # [B, L, rank]
        y_proj = self.linear2(y)   # [B, L, rank]

        # Hadamard product（二阶交互）
        fused = x_proj * y_proj    # [B, L, rank]

        # 映射回输出空间
        out = self.linear_out(fused)  # [B, L, out_features]

        return out
