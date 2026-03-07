"""
FinSentNet — Full Model Assembly.
==================================

Assembles all branches into the complete FinSentNet architecture:

    ┌─────────────────┐         ┌─────────────────┐
    │   News Text     │         │   Price OHLCV    │
    │  (batch, seq)   │         │ (batch, win, feat)│
    └────────┬────────┘         └────────┬────────┘
             │                           │
    ┌────────▼────────┐         ┌────────▼────────┐
    │  Text Branch    │         │  Price Branch   │
    │ BiLSTM + Attn   │         │ CNN → LSTM      │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             │  (batch, d_model)         │  (batch, d_model)
             │                           │
    ┌────────▼───────────────────────────▼────────┐
    │        Cross-Modal Attention Fusion          │
    │    (bidirectional cross-attention × N)       │
    │    + modal importance gating                 │
    └────────────────────┬────────────────────────┘
                         │
                    (batch, d_model)
                         │
    ┌────────────────────▼────────────────────────┐
    │              Dual-Head Output                │
    │  ┌─────────────┐    ┌─────────────────┐    │
    │  │  Direction   │    │   Confidence    │    │
    │  │  (3-class)   │    │  (calibrated)   │    │
    │  └─────────────┘    └─────────────────┘    │
    └─────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from finsent.models.text_branch import TextBranch
from finsent.models.price_branch import PriceBranch
from finsent.models.fusion import CrossModalFusion
from finsent.models.dual_head import DualHead


class FinSentNet(nn.Module):
    """Complete FinSentNet model.
    
    Combines text sentiment analysis with price feature encoding
    through cross-modal attention fusion for financial prediction.
    
    Total parameters: ~8-12M (trainable on RTX 4060 8GB)
    """
    
    def __init__(
        self,
        # Text branch
        vocab_size: int = 50000,
        text_embedding_dim: int = 256,
        text_hidden_dim: int = 256,
        text_num_layers: int = 2,
        text_attention_heads: int = 8,
        text_dropout: float = 0.3,
        # Price branch
        n_price_features: int = 15,
        price_window: int = 30,
        cnn_channels: list = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        price_dropout: float = 0.2,
        # Fusion
        d_model: int = 256,
        fusion_heads: int = 8,
        fusion_layers: int = 3,
        fusion_ff_dim: int = 512,
        fusion_dropout: float = 0.1,
        # Dual head
        n_classes: int = 3,
        temperature_init: float = 1.5,
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        # ─── Text Branch ──────────────────────────────────────────
        self.text_branch = TextBranch(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            output_dim=d_model,
            num_layers=text_num_layers,
            num_attention_heads=text_attention_heads,
            dropout=text_dropout,
        )
        
        # ─── Price Branch ─────────────────────────────────────────
        self.price_branch = PriceBranch(
            n_features=n_price_features,
            window_size=price_window,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            output_dim=d_model,
            dropout=price_dropout,
        )
        
        # ─── Cross-Modal Fusion ───────────────────────────────────
        self.fusion = CrossModalFusion(
            d_model=d_model,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            feedforward_dim=fusion_ff_dim,
            dropout=fusion_dropout,
        )
        
        # ─── Dual-Head Output ─────────────────────────────────────
        self.dual_head = DualHead(
            input_dim=d_model,
            n_classes=n_classes,
            dropout=0.2,
            temperature_init=temperature_init,
        )
    
    def forward(
        self,
        price: torch.Tensor,          # (batch, window_size, n_features)
        text_ids: torch.Tensor,        # (batch, max_seq_length)
        text_mask: torch.Tensor,       # (batch, max_seq_length)
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through FinSentNet.
        
        Returns dict with:
            direction_logits: (batch, n_classes)
            direction_probs: (batch, n_classes)
            confidence: (batch, 1)
            temperature: scalar
            text_repr: (batch, d_model)
            price_repr: (batch, d_model)
            fused_repr: (batch, d_model)
            attention_info: dict of cross-modal attention weights
            text_attention: text self-attention weights
            temporal_attention: price temporal attention weights
        """
        # ─── Branch Encoding ──────────────────────────────────────
        text_repr, text_attn = self.text_branch(text_ids, text_mask)
        price_repr, temporal_attn = self.price_branch(price)
        
        # ─── Cross-Modal Fusion ───────────────────────────────────
        fused_repr, attention_info = self.fusion(text_repr, price_repr)
        
        # ─── Dual-Head Prediction ─────────────────────────────────
        outputs = self.dual_head(fused_repr)
        
        # Augment with intermediate representations
        outputs["text_repr"] = text_repr
        outputs["price_repr"] = price_repr
        outputs["fused_repr"] = fused_repr
        outputs["attention_info"] = attention_info
        outputs["text_attention"] = text_attn
        outputs["temporal_attention"] = temporal_attn
        
        return outputs
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by module."""
        param_counts = {}
        for name, module in [
            ("text_branch", self.text_branch),
            ("price_branch", self.price_branch),
            ("fusion", self.fusion),
            ("dual_head", self.dual_head),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_counts[name] = {"total": total, "trainable": trainable}
        
        total_all = sum(v["total"] for v in param_counts.values())
        trainable_all = sum(v["trainable"] for v in param_counts.values())
        param_counts["TOTAL"] = {"total": total_all, "trainable": trainable_all}
        
        return param_counts
    
    def print_architecture(self) -> None:
        """Print architecture summary."""
        print("\n" + "=" * 70)
        print("FinSentNet Architecture Summary")
        print("=" * 70)
        
        counts = self.count_parameters()
        for name, info in counts.items():
            total = info["total"]
            trainable = info["trainable"]
            if total > 1e6:
                print(f"  {name:20s}: {total/1e6:7.2f}M params ({trainable/1e6:.2f}M trainable)")
            else:
                print(f"  {name:20s}: {total:>10,d} params ({trainable:,d} trainable)")
        
        print("=" * 70)

    @classmethod
    def from_config(cls, config: dict) -> "FinSentNet":
        """Construct model from config dict."""
        return cls(
            vocab_size=config["data"]["vocab_size"],
            text_embedding_dim=config["text_branch"]["embedding_dim"],
            text_hidden_dim=config["text_branch"]["hidden_dim"],
            text_num_layers=config["text_branch"]["num_layers"],
            text_attention_heads=config["text_branch"]["attention_heads"],
            text_dropout=config["text_branch"]["dropout"],
            n_price_features=config["data"]["price_features"],
            price_window=config["data"]["price_window"],
            cnn_channels=config["price_branch"]["cnn_channels"],
            lstm_hidden=config["price_branch"]["lstm_hidden"],
            lstm_layers=config["price_branch"]["lstm_layers"],
            price_dropout=config["price_branch"]["lstm_dropout"],
            d_model=config["fusion"]["d_model"],
            fusion_heads=config["fusion"]["num_heads"],
            fusion_layers=config["fusion"]["num_layers"],
            fusion_ff_dim=config["fusion"]["feedforward_dim"],
            fusion_dropout=config["fusion"]["dropout"],
            n_classes=config["dual_head"]["direction_classes"],
            temperature_init=config["dual_head"]["temperature_init"],
        )
