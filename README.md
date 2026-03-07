```
███████╗██╗███╗   ██╗███████╗███████╗███╗   ██╗████████╗
██╔════╝██║████╗  ██║██╔════╝██╔════╝████╗  ██║╚══██╔══╝
█████╗  ██║██╔██╗ ██║███████╗█████╗  ██╔██╗ ██║   ██║
██╔══╝  ██║██║╚██╗██║╚════██║██╔══╝  ██║╚██╗██║   ██║
██║     ██║██║ ╚████║███████║███████╗██║ ╚████║   ██║
╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝
                 N E T   P R O
      AI-Powered Quantitative Trading Intelligence
```

# FinSentNet — Financial Sentiment Network

> Cross-modal deep learning framework fusing news sentiment with price dynamics via attention mechanisms for financial market prediction.

---

## Architecture

```
┌──────────────────┐           ┌──────────────────┐
│   News Text      │           │   Price OHLCV     │
│   (headlines)    │           │ + Technical Ind.  │
└────────┬─────────┘           └─────────┬────────┘
         │                               │
┌────────▼─────────┐           ┌─────────▼────────┐
│  Text Branch     │           │  Price Branch    │
│  BiLSTM +        │           │  1D-CNN →        │
│  Multi-Head Attn │           │  LSTM + Attn     │
└────────┬─────────┘           └─────────┬────────┘
         │   (batch, 256)                │   (batch, 256)
         │                               │
┌────────▼───────────────────────────────▼────────┐
│         Cross-Modal Attention Fusion             │
│   Bidirectional: Text→Price + Price→Text         │
│   Gated modal importance weighting               │
│   3 layers × 8 attention heads                   │
└───────────────────────┬─────────────────────────┘
                        │   (batch, 256)
┌───────────────────────▼─────────────────────────┐
│               Dual-Head Output                   │
│  ┌──────────────┐        ┌────────────────────┐ │
│  │  Direction    │        │   Confidence       │ │
│  │  (↑ / — / ↓) │        │ (temp-calibrated)  │ │
│  └──────────────┘        └────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Key Features

| Component | Implementation |
|-----------|---------------|
| **Text Encoder** | BiLSTM + Multi-Head Self-Attention + Learnable Query Pooling |
| **Price Encoder** | Dilated Causal Conv → LSTM + Temporal Attention |
| **Fusion** | Bidirectional Cross-Modal Attention with learned modal gating |
| **Output** | 3-class direction (Up/Neutral/Down) + calibrated confidence |
| **Loss** | Focal Loss (γ=2.0) + Confidence Calibration (ECE penalty) |
| **GAN Augmentation** | WGAN-GP conditioned on market regime for crisis events |
| **Position Sizing** | Confidence-scaled quarter-Kelly criterion |
| **Risk Management** | Max drawdown halt, VaR limits, correlation constraints |
| **Calibration** | Post-training temperature scaling (LBFGS optimization) |

## Project Structure

```
FINSENT/
├── config/
│   └── config.yaml              # Full system configuration
├── finsent/
│   ├── data/
│   │   ├── pipeline.py          # End-to-end data orchestrator
│   │   ├── price_loader.py      # OHLCV fetching (Yahoo Finance + CSV)
│   │   ├── news_loader.py       # News loading, tokenization, alignment
│   │   ├── features.py          # Technical indicators (from scratch)
│   │   ├── temporal_align.py    # Temporal integrity & label creation
│   │   └── dataset.py           # PyTorch Dataset + temporal sampler
│   ├── models/
│   │   ├── layers.py            # Custom attention, GRN, causal conv
│   │   ├── text_branch.py       # BiLSTM + attention text encoder
│   │   ├── price_branch.py      # CNN-LSTM price encoder
│   │   ├── fusion.py            # Cross-modal attention fusion
│   │   ├── dual_head.py         # Direction + confidence heads
│   │   ├── finsent_net.py       # Full model assembly
│   │   └── gan.py               # WGAN-GP crisis augmentation
│   ├── training/
│   │   ├── trainer.py           # Training loop (early stop on Sharpe)
│   │   ├── losses.py            # Focal loss + calibration loss
│   │   ├── schedulers.py        # Cosine warmup scheduler
│   │   └── calibration.py       # Temperature scaling
│   ├── backtest/
│   │   ├── engine.py            # Event-driven backtesting engine
│   │   ├── metrics.py           # Sharpe, Sortino, MaxDD, VaR, etc.
│   │   ├── position_sizing.py   # Kelly, volatility parity
│   │   └── risk.py              # Risk management engine
│   ├── portfolio/
│   │   └── optimizer.py         # Mean-variance, risk parity, Black-Litterman
│   └── utils/
│       ├── seed.py              # Reproducibility
│       ├── logging_utils.py     # Structured logging
│       └── visualization.py     # Financial-grade plotting
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python main.py --config config/config.yaml --ticker AAPL
```

### 3. Training Only

```bash
python main.py --mode train --ticker NVDA
```

### 4. Backtest from Checkpoint

```bash
python main.py --mode backtest --checkpoint outputs/checkpoints/finsent_final.pt
```

## Configuration

All hyperparameters are centralized in `config/config.yaml`:

- **Data**: tickers, date range, lookback windows, train/val/test splits
- **Text Branch**: embedding dim, BiLSTM layers, attention heads
- **Price Branch**: CNN channels, LSTM layers, dilated convolutions
- **Fusion**: cross-modal attention depth, feedforward dim
- **Training**: focal loss gamma, learning rate, early stopping patience
- **Backtest**: capital, commission/slippage, Kelly fraction, risk limits
- **Portfolio**: optimization method, target volatility, rebalance frequency

## Design Principles

1. **No Look-Ahead Bias**: All features use strictly causal computation. News is aligned with configurable temporal lag (min 24h). Walk-forward splits ensure temporal ordering.

2. **Financial Metric Optimization**: Early stopping monitors validation Sharpe ratio, not cross-entropy loss. A model with lower loss but worse Sharpe is _not_ a better model.

3. **From-Scratch Implementation**: All technical indicators, attention mechanisms, and neural network layers are implemented from mathematical first principles — no black-box libraries.

4. **Calibrated Confidence**: Temperature scaling ensures predicted confidence maps to empirical probability of correctness, enabling meaningful position sizing.

5. **Production-Grade Backtesting**: Event-driven simulation with realistic transaction costs (5bps commission + 2bps slippage), risk management limits, and max drawdown circuit breakers.

## Mathematical Foundation

### Focal Loss
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Down-weights easy examples (high $p_t$), focusing training on hard misclassifications near decision boundaries.

### Cross-Modal Attention
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Applied bidirectionally: text queries attend to price keys (what price patterns are relevant to this news?) and vice versa.

### Kelly Criterion Position Sizing
$$f^* = \frac{p(b+1) - 1}{b}$$

where $p$ = win probability, $b$ = win/loss ratio. Scaled by model confidence and halved (quarter-Kelly) for safety.

### Black-Litterman
$$\mu_{BL} = \left[(\tau\Sigma)^{-1} + P'\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}Q\right]$$

Model predictions serve as investor "views" ($Q$) with uncertainty ($\Omega$) inversely proportional to calibrated confidence.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4GB VRAM | 8GB+ (RTX 4060) |
| RAM | 8GB | 16GB+ |
| Storage | 2GB | 10GB (with cached data) |

Model: ~8-12M parameters. Fits comfortably in 8GB VRAM with mixed precision.

## License

Research use only. Not financial advice.
