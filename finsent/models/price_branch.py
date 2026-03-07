"""
Price Feature Encoding Branch — 1D-CNN → LSTM Hybrid.
======================================================

Architecture:
    Input: (batch, window_size, n_features)
    → Transpose to (batch, n_features, window_size) for CNN
    → Stack of 1D Causal Conv blocks (local pattern extraction)
    → Transpose back to (batch, window_size, cnn_channels)
    → LSTM (temporal dependency modeling)
    → Attention pooling → GRN → Price Representation

Financial Intuition:
    The CNN layers act as local pattern detectors:
    - Kernels learn to recognize candlestick patterns (doji, engulfing, etc.)
    - Multi-scale dilations capture patterns at different timescales
    - Batch normalization stabilizes training on non-stationary financial data
    
    The LSTM layers capture longer-range temporal dependencies:
    - Momentum persistence (trending behavior)
    - Mean reversion dynamics
    - Volatility clustering (GARCH-like effects)
    
    This CNN→LSTM design is motivated by the observation that financial 
    time series exhibit both local patterns (intraday) and global regime 
    dynamics (multi-day trends). CNNs excel at the former, LSTMs at the latter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from finsent.models.layers import (
    TemporalConvBlock,
    MultiHeadAttention,
    GatedResidualNetwork,
)


class PriceBranch(nn.Module):
    """1D-CNN → LSTM hybrid encoder for price features.
    
    Design choices:
    1. Causal convolutions — no look-ahead within the window
    2. Increasing channel widths — hierarchical feature extraction
    3. Dilated convolutions — exponentially growing receptive field
    4. LSTM with orthogonal init — preserves gradient flow over 30+ timesteps
    5. Attention pooling — learns which timesteps matter most for prediction
    """
    
    def __init__(
        self,
        n_features: int = 15,
        window_size: int = 30,
        cnn_channels: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.2,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3, 3, 3]
        
        assert len(cnn_channels) == len(cnn_kernel_sizes)
        
        self.n_features = n_features
        self.window_size = window_size
        self.output_dim = output_dim
        
        # ─── 1D CNN Feature Extractor ─────────────────────────────
        # Input: (batch, n_features, window_size)
        cnn_layers = []
        in_channels = n_features
        
        for i, (out_ch, kernel) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            dilation = 2 ** i  # exponentially increasing dilation
            cnn_layers.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_output_channels = cnn_channels[-1]
        
        # ─── LSTM Temporal Encoder ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,  # Unidirectional — causal for price
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        
        # ─── Temporal Self-Attention ──────────────────────────────
        self.temporal_attention = MultiHeadAttention(
            d_model=lstm_hidden,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(lstm_hidden)
        
        # ─── Attention Pooling ────────────────────────────────────
        self.pool_query = nn.Parameter(torch.randn(1, 1, lstm_hidden))
        nn.init.xavier_uniform_(self.pool_query)
        
        # ─── Output Projection ────────────────────────────────────
        self.output_grn = GatedResidualNetwork(
            input_dim=lstm_hidden,
            hidden_dim=lstm_hidden * 2,
            output_dim=output_dim,
            dropout=dropout,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM with orthogonal weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden = self.lstm.hidden_size
                param.data[hidden:2 * hidden].fill_(1.0)  # forget gate bias
    
    def forward(
        self,
        price: torch.Tensor,  # (batch, window_size, n_features)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            price: Price feature tensor, (batch, window_size, n_features)
        
        Returns:
            price_repr: (batch, output_dim) — aggregated price representation
            temporal_weights: (batch, num_heads, 1, window_size) — temporal attention
        """
        batch_size = price.size(0)
        
        # ─── CNN: Extract Local Patterns ──────────────────────────
        # Transpose: (batch, n_features, window_size) for Conv1D
        x = price.transpose(1, 2)
        x = self.cnn(x)  # (batch, cnn_channels[-1], window_size)
        
        # Transpose back: (batch, window_size, cnn_channels[-1])
        x = x.transpose(1, 2)
        
        # ─── LSTM: Temporal Dependencies ──────────────────────────
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, window_size, lstm_hidden)
        
        # ─── Temporal Self-Attention ──────────────────────────────
        attended, _ = self.temporal_attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
        )
        attended = self.attn_norm(attended + lstm_out)  # residual
        
        # ─── Attention-Weighted Pooling ──────────────────────────
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pooled, temporal_weights = self.temporal_attention(
            query=pool_query,       # (batch, 1, d_model)
            key=attended,           # (batch, window_size, d_model)
            value=attended,         # (batch, window_size, d_model)
        )
        pooled = pooled.squeeze(1)  # (batch, lstm_hidden)
        
        # ─── Output GRN ──────────────────────────────────────────
        price_repr = self.output_grn(pooled)  # (batch, output_dim)
        
        return price_repr, temporal_weights
    
    def get_temporal_importance(
        self,
        price: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-timestep importance for interpretability.
        
        Shows which days in the lookback window the model considers
        most informative for prediction.
        
        Returns: (batch, window_size) importance scores
        """
        _, attn_weights = self.forward(price)
        importance = attn_weights.mean(dim=1).squeeze(1)  # (batch, window_size)
        return importance
