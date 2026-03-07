"""
FINSENT NET PRO — Analysis Routes
/api/analyze endpoint — the main analysis pipeline.
"""

import time
import logging
import numpy as np
import torch
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["Analysis"])
logger = logging.getLogger("finsent.routes.analysis")

# Singletons injected from main.py
_fetcher = None
_indicators = None
_sentiment = None
_regime = None
_aligner = None
_model = None
_signal_gen = None
_optimizer = None
_allocator = None
_backtester = None
_risk_engine = None


def init(fetcher, indicators, sentiment, regime, aligner, model,
         signal_gen, optimizer, allocator, backtester, risk_engine):
    """Initialize route with shared service instances."""
    global _fetcher, _indicators, _sentiment, _regime, _aligner
    global _model, _signal_gen, _optimizer, _allocator, _backtester, _risk_engine
    _fetcher = fetcher
    _indicators = indicators
    _sentiment = sentiment
    _regime = regime
    _aligner = aligner
    _model = model
    _signal_gen = signal_gen
    _optimizer = optimizer
    _allocator = allocator
    _backtester = backtester
    _risk_engine = risk_engine


class AnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=10)
    market: str = "SP500"
    total_capital: float = Field(default=100000, alias="investment_amount", ge=1000)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    currency: str = "USD"
    horizon: str = "1M"

    class Config:
        populate_by_name = True


@router.post("/analyze")
async def analyze_stocks(request: AnalysisRequest):
    """
    Complete multi-stock analysis pipeline.
    Fetches data → computes indicators → runs FINSENT → generates signals →
    optimizes portfolio → runs backtest → computes risk metrics.
    """
    start = time.time()
    logger.info(
        f"Analysis: {request.tickers} | Market: {request.market} "
        f"| Capital: {request.total_capital}"
    )

    signals = []
    charts = {}
    stock_details = []

    for ticker in request.tickers[:10]:
        try:
            # 1. Fetch OHLCV data
            df = _fetcher.fetch_ohlcv(ticker, market=request.market, period="1y")
            if df.empty:
                df = _fetcher._fallback_synthetic(ticker, "1y")

            # 2. Compute technical indicators
            df = _indicators.compute_all(df)

            # 3. Sentiment analysis
            sentiment = _sentiment.generate_demo_sentiment(ticker)

            # 4. Regime detection
            regime_result = _regime.detect(df)

            # 5. Prepare model input & run FINSENT
            price_cols = [
                "Open", "High", "Low", "Close", "Volume",
                "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_Upper", "BB_Lower", "BB_Position", "ATR_14",
                "OBV", "Volume_Ratio", "EMA_20", "EMA_50",
                "Log_Return", "Volatility_20", "Stoch_K",
            ]
            available_cols = [c for c in price_cols if c in df.columns]
            price_window = df[available_cols].iloc[-30:].values

            # Z-score normalization (causal)
            price_mean = price_window.mean(axis=0, keepdims=True)
            price_std = price_window.std(axis=0, keepdims=True) + 1e-8
            price_norm = (price_window - price_mean) / price_std

            # Pad to 20 features
            if price_norm.shape[1] < 20:
                padding = np.zeros((price_norm.shape[0], 20 - price_norm.shape[1]))
                price_norm = np.concatenate([price_norm, padding], axis=1)
            price_tensor = torch.FloatTensor(price_norm).unsqueeze(0)

            # Dummy text tokens (production: real news tokenization)
            text_tokens = torch.randint(0, 1000, (1, 50))

            # Run FINSENT model
            with torch.no_grad():
                model_output = _model(text_tokens, price_tensor)

            # 6. Generate trade signal
            signal = _signal_gen.generate_signal(
                ticker=ticker,
                market=request.market,
                model_output=model_output,
                price_data=df,
                total_capital=request.total_capital,
                risk_tolerance=request.risk_tolerance,
            )
            signals.append(signal)

            # 7. Chart data (last 60 days)
            chart_df = df.tail(60)
            charts[ticker] = [
                {
                    "time": int(idx.timestamp()) if hasattr(idx, "timestamp") else i,
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                }
                for i, (idx, row) in enumerate(chart_df.iterrows())
            ]

            # 8. Stock detail
            latest = df.iloc[-1]
            stock_details.append({
                "ticker": ticker,
                "price": round(float(latest["Close"]), 2),
                "rsi": round(float(latest.get("RSI_14", 50)), 1),
                "macd": round(float(latest.get("MACD", 0)), 4),
                "sma_50": round(float(latest.get("SMA_50", latest["Close"])), 2),
                "sma_200": round(float(latest.get("SMA_200", latest["Close"])), 2),
                "volume": int(latest.get("Volume", 0)),
                "atr": round(float(latest.get("ATR_14", 0)), 2),
                "bb_position": round(float(latest.get("BB_Position", 0.5)), 2),
                "sentiment": sentiment,
                "regime": regime_result.get("detail", "UNKNOWN"),
            })

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            continue

    if not signals:
        raise HTTPException(status_code=400, detail="No valid signals could be generated")

    # ── Portfolio Optimization ──
    n = len(signals)
    expected_rets = np.array([s.predicted_return / 100 for s in signals])
    cov = np.eye(n) * 0.02 ** 2
    opt = _optimizer.optimize_weights(expected_rets, cov, method="max_sharpe")

    # ── Allocation ──
    allocation = _allocator.allocate(signals, opt, request.total_capital)

    # ── Backtest (demo) ──
    bt = _backtester.generate_demo_backtest(request.total_capital)

    # ── Risk metrics ──
    equity = np.array(bt["equity_curve"])
    risk_report = _risk_engine.full_risk_report(
        np.diff(equity) / equity[:-1], equity,
    )

    elapsed = time.time() - start
    logger.info(f"Analysis complete in {elapsed:.2f}s for {len(signals)} stocks")

    return {
        "status": "success",
        "analysis_time_seconds": round(elapsed, 2),
        "signals": [
            {
                "ticker": s.ticker,
                "direction": s.direction.value,
                "confidence": s.confidence,
                "predicted_return": s.predicted_return,
                "predicted_downside": s.predicted_downside,
                "entry_price": s.entry_price,
                "target_price": s.target_price,
                "stop_loss": s.stop_loss,
                "risk_reward": s.risk_reward_ratio,
                "kelly_fraction": s.kelly_fraction,
                "quantity": s.recommended_quantity,
                "capital_required": s.capital_required,
                "time_horizon": s.time_horizon,
                "reasoning": s.reasoning,
                "regime": s.regime,
                "sentiment_score": s.sentiment_score,
                "technical_score": s.technical_score,
            }
            for s in signals
        ],
        "stock_details": stock_details,
        "charts": charts,
        "portfolio": {
            "optimization": opt,
            "allocation": allocation,
        },
        "backtest": bt,
        "risk": risk_report,
    }
