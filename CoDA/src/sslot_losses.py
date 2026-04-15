import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).
    Expects features: [bsz, n_views, dim] and labels: [bsz].
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all",
                 base_temperature: float = 0.07, loss_scaling_factor: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.scaling_factor = loss_scaling_factor

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        if features.dim() < 3:
            raise ValueError("features must be [bsz, n_views, dim]")
        if labels is None:
            raise ValueError("labels required for supervised contrastive loss")

        bsz = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != bsz:
            raise ValueError("Num labels does not match num features")
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, dim]

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # logits: [bsz*anchor_count, bsz*contrast_count]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases
        logits_mask = torch.ones_like(mask)
        diag = torch.arange(bsz * anchor_count, device=device)
        logits_mask[diag, diag] = 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, bsz).mean() * self.scaling_factor
        return loss

class LinearModel(nn.Module):
    """Create two 'views' of sentence representation via dropout, then project to proj_dim."""
    def __init__(self, d_model: int, proj_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.Linear(d_model, proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # encoder_last_hidden_state: [bsz, seq, d_model], attention_mask: [bsz, seq]
        mask = attention_mask.unsqueeze(-1).float()                 # [B,L,1]
        last_state = encoder_last_hidden_state * mask               # [B,L,H]

        lengths = mask.sum(dim=1).clamp(min=1e-6)                   # [B,1]
        pooled = last_state.sum(dim=1) / lengths

        pooled_drop = self.dropout(pooled)
        v1 = self.layer(pooled)
        v2 = self.layer(pooled_drop)

        
        return torch.stack((v1, v2), dim=1)  # [bsz, 2, proj_dim]
