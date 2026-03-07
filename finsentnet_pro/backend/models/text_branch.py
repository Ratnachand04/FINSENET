"""
FINSENT NET PRO — Text Branch
FinBERT embeddings → TextCNN → 3-layer BiLSTM → Multi-Head Self-Attention
Extracts sentiment-aware representations from financial text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════
#  TextCNN — Kim (2014) parallel n-gram convolutions
# ═══════════════════════════════════════════════════════════


class TextCNN(nn.Module):
    """
    Parallel convolutions with different n-gram window sizes.
    Captures short / medium / long-range sentiment patterns.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim) — FinBERT token embeddings
        Returns:
            features: (batch, num_filters * len(kernel_sizes))
        """
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            activated = F.relu(conv(x))
            pooled.append(activated.max(dim=2).values)
        out = torch.cat(pooled, dim=1)
        return self.dropout(out)


# ═══════════════════════════════════════════════════════════
#  Multi-Head Self-Attention
# ═══════════════════════════════════════════════════════════


class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention.
    Applied to BiLSTM outputs — learns which words / time-steps matter most.

    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = np.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional padding mask
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, heads, seq, seq)
        """
        residual = x
        x = self.layer_norm(x)  # Pre-norm (more stable)
        batch, seq, _ = x.shape

        Q = self.W_q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        output = self.W_o(context) + residual  # Residual connection
        return output, attention_weights


# ═══════════════════════════════════════════════════════════
#  Text Branch — Full Pipeline
# ═══════════════════════════════════════════════════════════


class TextBranch(nn.Module):
    """
    Embedding → TextCNN → 3-layer BiLSTM → Self-Attention → sentence vector.

    For production: replace the Embedding layer with actual FinBERT encoder.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_lstm_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Embedding layer (replace with FinBERT in production)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)

        # TextCNN for local n-gram sentiment features
        self.text_cnn = TextCNN(embed_dim, num_filters=128, dropout=dropout)
        self.cnn_projection = nn.Linear(self.text_cnn.output_dim, output_dim)

        # 3-layer BiLSTM for sequential context  (256 × 2 = 512)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Multi-head self-attention on BiLSTM outputs
        self.self_attention = MultiHeadSelfAttention(
            output_dim, num_attention_heads, dropout,
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) — tokenized text
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            sentence_vec: (batch, output_dim)
            attention_weights: (batch, heads, seq, seq)
        """
        embeddings = self.dropout(self.embedding(input_ids))

        # TextCNN path (parallel)
        cnn_features = self.cnn_projection(self.text_cnn(embeddings))

        # BiLSTM path
        lstm_out, _ = self.bilstm(embeddings)  # (batch, seq, 512)

        # Self-attention on BiLSTM outputs
        attended, attn_weights = self.self_attention(lstm_out)

        # Sentence representation via masked mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (attended * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
        else:
            mean_pooled = attended.mean(dim=1)

        # Combine CNN features + LSTM attention features
        sentence_vec = self.layer_norm(mean_pooled + cnn_features)

        return sentence_vec, attn_weights
