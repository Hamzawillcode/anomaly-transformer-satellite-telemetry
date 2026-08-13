import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AnomalyAttention(nn.Module):
    def __init__(self, d_model, heads, seq_len):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.seq_len = seq_len

        # Standard Linear layers for Query, Key, Value
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # ── FIX 1 ──────────────────────────────────────────────────────────────
        # Paper Eq 2: σ = X^{l-1} * W_σ  →  σ is an INPUT-DEPENDENT projection,
        # NOT a static nn.Parameter.  A static parameter never adapts per sample
        # and produces exploding/vanishing gradients when the KL tries to use it.
        # Shape: (N, h)  i.e. one scale per time-point per head.
        self.sigma_proj = nn.Linear(d_model, heads)

        # Pre-compute relative temporal distances  |j - i|^2  (Paper Eq 2)
        idx = torch.arange(seq_len, dtype=torch.float32)
        distances = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2   # (N, N)
        self.register_buffer('distances', distances)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. Project Q, K, V → multi-head
        Q = self.query(x).view(batch_size, self.seq_len, self.heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, self.seq_len, self.heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, self.seq_len, self.heads, self.d_k).transpose(1, 2)

        # ── FIX 2 ──────────────────────────────────────────────────────────────
        # σ is now computed from the input (paper: σ = X W_σ).
        # Shape after projection: (B, N, h)
        # We clamp to a safe positive range — too-small σ → near-zero Gaussian
        # rows → NaN after row-normalisation; too-large → uniform prior (useless).
        sigma = self.sigma_proj(x)                          # (B, N, h)
        sigma = torch.clamp(sigma, min=0.5, max=10.0)       # stable range
        # Rearrange to (B, h, N, 1) so it broadcasts over the (N, N) distance matrix
        sigma = sigma.permute(0, 2, 1).unsqueeze(-1)        # (B, h, N, 1)

        # ─────────────────────────────────────────────────────────────────────
        # BRANCH 1: SERIES-ASSOCIATION  S = Softmax(QK^T / sqrt(d_k))
        # ─────────────────────────────────────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        series_assoc = F.softmax(scores, dim=-1)            # (B, h, N, N)

        # ─────────────────────────────────────────────────────────────────────
        # BRANCH 2: PRIOR-ASSOCIATION  P = Rescale(G(|j-i|; σ))
        # Paper Eq 2: G = (1/√(2π)σ) * exp(-|j-i|² / 2σ²)
        # ── FIX 3 ──────────────────────────────────────────────────────────────
        # Include the 1/√(2π)σ prefactor — without it the kernel scale is wrong
        # and the row-normalisation still works, but the gradient magnitudes differ
        # from what the paper expects (minor but contributes to spikes).
        # ─────────────────────────────────────────────────────────────────────
        dist = self.distances.unsqueeze(0).unsqueeze(0)     # (1, 1, N, N)
        gaussian = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * \
                   torch.exp(-dist / (2.0 * sigma ** 2))    # (B, h, N, N)

        # Rescale so each row sums to 1 → valid probability distribution
        prior_assoc = gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-8)  # (B, h, N, N)

        # ─────────────────────────────────────────────────────────────────────
        # RECONSTRUCTION  Z = S * V
        # ─────────────────────────────────────────────────────────────────────
        context = torch.matmul(series_assoc, V)             # (B, h, N, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.seq_len, self.d_model)
        out = self.out_proj(context)

        return out, series_assoc, prior_assoc


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, seq_len, dropout=0.1):
        super(AnomalyTransformerBlock, self).__init__()
        self.attention = AnomalyAttention(d_model, heads, seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, series_assoc, prior_assoc = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, series_assoc, prior_assoc


class AnomalyTransformer(nn.Module):
    def __init__(self, num_features, seq_len, d_model=512, heads=8, layers=3, dropout=0.1):
        super(AnomalyTransformer, self).__init__()
        self.embedding    = nn.Linear(num_features, d_model)
        self.blocks       = nn.ModuleList([
            AnomalyTransformerBlock(d_model, heads, seq_len, dropout)
            for _ in range(layers)
        ])
        self.reconstruction = nn.Linear(d_model, num_features)

    def forward(self, x):
        series_list = []
        prior_list  = []
        x = self.embedding(x)
        for block in self.blocks:
            x, series_assoc, prior_assoc = block(x)
            series_list.append(series_assoc)
            prior_list.append(prior_assoc)
        out = self.reconstruction(x)
        return out, series_list, prior_list
