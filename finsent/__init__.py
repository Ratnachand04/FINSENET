"""
FinSentNet — Financial Sentiment Network
=========================================
Cross-modal deep learning framework for financial prediction
fusing news sentiment with price dynamics via attention mechanisms.

Architecture:
    Text Branch (BiLSTM + Multi-Head Attention)
    ↕ Cross-Modal Attention Fusion
    Price Branch (1D-CNN → LSTM)
    ↓
    Dual-Head Output (Direction + Calibrated Confidence)

Author: FinSentNet Research Team
"""

__version__ = "0.1.0"
__all__ = ["data", "models", "training", "backtest", "portfolio", "utils"]
