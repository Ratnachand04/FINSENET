"""
FinSentNet Pro — Complete PyTorch Architecture
Based on FinSentNet Research Blueprint: Modules 1-9

Architecture:
  Text Branch:  Embedding → TextCNN → 3-layer BiLSTM → Multi-Head Self-Attention
  Price Branch: Multi-scale Conv1D → Dilated CNN → LSTM Encoder
  Fusion:       Cross-Modal Attention (Sentiment Query × Price KV)
  Output:       Dual-Head (Direction Classifier + Return Regressor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════
#  TEXT BRANCH COMPONENTS
# ═══════════════════════════════════════════════════════════


class TextCNN(nn.Module):
    """Kim (2014) TextCNN — parallel convolutions with different n-gram sizes."""

    def __init__(self, embed_dim: int = 768, num_filters: int = 128,
                 kernel_sizes: list = None, dropout: float = 0.3):
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
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            activated = F.relu(conv(x))
            pooled.append(activated.max(dim=2).values)
        out = torch.cat(pooled, dim=1)
        return self.dropout(out)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
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

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.layer_norm(x)
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
        output = self.W_o(context) + residual
        return output, attention_weights


# ═══════════════════════════════════════════════════════════
#  CROSS-MODAL ATTENTION FUSION — THE CORE INNOVATION
# ═══════════════════════════════════════════════════════════


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention where:
      Q = Sentiment vector  ("What is the market feeling?")
      K = Price sequence    ("What patterns exist?")
      V = Price sequence    ("What information to retrieve?")

    Gating: α = sigmoid(Wg · [sent; price] + bg)
    Fused  = α · cross_attended + (1-α) · sentiment
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = np.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.gate_layer = nn.Linear(d_model * 2, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)

    def forward(self, sentiment_vec: torch.Tensor,
                price_sequence: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.layer_norm_q(sentiment_vec.unsqueeze(1))
        KV = self.layer_norm_kv(price_sequence)

        q = self.W_q(Q)
        k = self.W_k(KV)
        v = self.W_v(KV)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        cross_context = torch.matmul(attn_weights, v).squeeze(1)
        cross_context = self.W_o(cross_context)

        gate_input = torch.cat([sentiment_vec, cross_context], dim=-1)
        alpha = torch.sigmoid(self.gate_layer(gate_input))
        fused = alpha * cross_context + (1 - alpha) * sentiment_vec

        return fused, attn_weights.squeeze(1)


# ═══════════════════════════════════════════════════════════
#  PRICE BRANCH
# ═══════════════════════════════════════════════════════════


class PriceBranch(nn.Module):
    """Multi-scale Conv1D → Dilated CNN → LSTM → price context vector."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 256,
                 output_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.conv_3 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv_7 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.conv_15 = nn.Conv1d(input_dim, 64, kernel_size=15, padding=7)
        self.bn_multiscale = nn.BatchNorm1d(192)

        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(192, 192, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4, 8]
        ])
        self.dilated_bns = nn.ModuleList([nn.BatchNorm1d(192) for _ in range(4)])
        self.dilated_res = nn.Conv1d(192, 192, kernel_size=1)

        self.lstm = nn.LSTM(192, hidden_dim, num_layers=2, batch_first=True,
                            dropout=dropout, bidirectional=False)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = x.transpose(1, 2)
        c3 = F.gelu(self.conv_3(x_conv))
        c7 = F.gelu(self.conv_7(x_conv))
        c15 = F.gelu(self.conv_15(x_conv))
        multi_scale = self.bn_multiscale(torch.cat([c3, c7, c15], dim=1))

        dilated_out = multi_scale
        for conv, bn in zip(self.dilated_convs, self.dilated_bns):
            dilated_out = F.gelu(bn(conv(dilated_out)))
        dilated_out = dilated_out + self.dilated_res(multi_scale)

        lstm_input = dilated_out.transpose(1, 2)
        lstm_out, (h_n, _) = self.lstm(lstm_input)

        price_context = self.projection(h_n[-1])
        price_sequence = self.projection(self.dropout(lstm_out))
        return price_context, price_sequence


# ═══════════════════════════════════════════════════════════
#  TEXT BRANCH
# ═══════════════════════════════════════════════════════════


class TextBranch(nn.Module):
    """Embedding → TextCNN → 3-layer BiLSTM → Self-Attention → sentence vector."""

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 768,
                 hidden_dim: int = 256, output_dim: int = 512,
                 num_lstm_layers: int = 3, num_attention_heads: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.text_cnn = TextCNN(embed_dim, num_filters=128, dropout=dropout)
        self.cnn_projection = nn.Linear(self.text_cnn.output_dim, output_dim)

        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_lstm_layers,
                              batch_first=True,
                              dropout=dropout if num_lstm_layers > 1 else 0,
                              bidirectional=True)

        self.self_attention = MultiHeadSelfAttention(output_dim, num_attention_heads, dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.dropout(self.embedding(input_ids))
        cnn_features = self.cnn_projection(self.text_cnn(embeddings))
        lstm_out, _ = self.bilstm(embeddings)  # (batch, seq, 512)

        attended, attn_weights = self.self_attention(lstm_out)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            mean_pooled = (attended * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            mean_pooled = attended.mean(dim=1)

        sentence_vec = self.layer_norm(mean_pooled + cnn_features)
        return sentence_vec, attn_weights


# ═══════════════════════════════════════════════════════════
#  FULL MODEL
# ═══════════════════════════════════════════════════════════


class FinSentNetCore(nn.Module):
    """
    Complete FinSentNet Architecture.

    Text → TextBranch → 512-d sentiment
    Price → PriceBranch → 512-d price context + sequence
    Cross-Modal Fusion → 512-d fused vector
    Dual-Head:
      A: Direction (UP / NEUTRAL / DOWN)
      B: Return magnitude (%)
    """

    def __init__(self, price_feature_dim: int = 20, d_model: int = 512,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.text_branch = TextBranch(output_dim=d_model, dropout=dropout)
        self.price_branch = PriceBranch(input_dim=price_feature_dim,
                                        output_dim=d_model, dropout=dropout)
        self.fusion = CrossModalAttentionFusion(d_model=d_model, dropout=dropout)

        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.return_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, text_tokens: torch.Tensor,
                price_sequence: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        sentiment_vec, text_attn = self.text_branch(text_tokens, text_mask)
        price_ctx, price_seq = self.price_branch(price_sequence)
        fused_vec, cross_attn = self.fusion(sentiment_vec, price_seq)

        direction_logits = self.direction_head(fused_vec)
        return_pred = self.return_head(fused_vec)

        return {
            "direction_logits": direction_logits,
            "direction_probs": F.softmax(direction_logits, dim=-1),
            "return_pred": return_pred,
            "cross_attention": cross_attn,
            "text_attention": text_attn,
            "sentiment_vec": sentiment_vec,
            "fused_vec": fused_vec,
        }
