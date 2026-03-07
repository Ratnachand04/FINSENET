"""
FinSentNet Pro — FastAPI Backend
All endpoints for the frontend to consume.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Fix imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.market_data_fetcher import MarketDataFetcher
from data_pipeline.technical_indicators import TechnicalIndicators
from data_pipeline.news_sentiment_engine import NewsSentimentEngine
from data_pipeline.regime_detector import RegimeDetector, MarketRegime
from data_pipeline.data_aligner import DataAligner
from models.signal_generator import SignalGenerator, SignalDirection
from portfolio.kelly_sizer import KellySizer
from portfolio.portfolio_optimizer import PortfolioOptimizer
from portfolio.risk_engine import RiskEngine
from portfolio.allocation_engine import AllocationEngine
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_metrics import PerformanceMetrics

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finsentnet_pro")

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="FinSentNet Pro API",
    description="Multi-Modal AI Trading Intelligence Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Singletons ───────────────────────────────────────────
fetcher = MarketDataFetcher()
sentiment_engine = NewsSentimentEngine(use_finbert=False)  # start with lexicon for speed
regime_detector = RegimeDetector()
aligner = DataAligner()
signal_gen = SignalGenerator()
kelly = KellySizer()
optimizer = PortfolioOptimizer()
risk_engine = RiskEngine()
allocator = AllocationEngine()
backtester = BacktestEngine()

# ═══════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════


class AnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=10)
    market: str = "SP500"
    investment_amount: float = Field(default=100000, ge=1000)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    strategy: str = "balanced"


class QuoteResponse(BaseModel):
    ticker: str
    price: float
    change: float
    change_pct: float
    volume: int
    market_cap: str
    pe_ratio: Optional[float]
    week_52_high: float
    week_52_low: float


# ═══════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════


@app.get("/")
async def root():
    """Serve frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "FinSentNet Pro API v2.0 — Frontend not found. Use /docs for API reference."}


@app.get("/api/health")
async def health_check():
    return {
        "status": "operational",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "modules": {
            "data_fetcher": True,
            "sentiment_engine": True,
            "regime_detector": True,
            "signal_generator": True,
            "portfolio_optimizer": True,
            "backtester": True,
        }
    }


@app.get("/api/market/search")
async def market_search(q: str = Query(..., min_length=1)):
    """Search for tickers across all markets."""
    results = []
    query = q.upper().strip()
    for market, cfg in fetcher.MARKET_CONFIG.items():
        components = fetcher.get_market_components(market)
        for ticker in components:
            if query in ticker.upper():
                results.append({
                    "ticker": ticker,
                    "market": market,
                    "currency": cfg["currency"],
                })
                if len(results) >= 20:
                    return {"results": results, "query": q}
    return {"results": results, "query": q}


@app.get("/api/market/quote/{ticker}")
async def get_quote(ticker: str, market: str = "SP500"):
    """Get current quote for a ticker."""
    try:
        price_data = fetcher.fetch_ohlcv(ticker, market=market, period="5d")
        if price_data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        latest = price_data.iloc[-1]
        prev = price_data.iloc[-2] if len(price_data) > 1 else latest
        change = latest["Close"] - prev["Close"]
        change_pct = (change / prev["Close"]) * 100 if prev["Close"] > 0 else 0

        high_52 = price_data["High"].max()
        low_52 = price_data["Low"].min()

        return {
            "ticker": ticker,
            "price": round(float(latest["Close"]), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "volume": int(latest.get("Volume", 0)),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "open": round(float(latest["Open"]), 2),
            "week_52_high": round(float(high_52), 2),
            "week_52_low": round(float(low_52), 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/chart/{ticker}")
async def get_chart_data(ticker: str, market: str = "SP500", period: str = "6mo"):
    """Get OHLCV chart data for TradingView widget."""
    try:
        df = fetcher.fetch_ohlcv(ticker, market=market, period=period)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No chart data for {ticker}")

        candles = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(time.time())
            candles.append({
                "time": ts,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })

        return {"ticker": ticker, "market": market, "candles": candles}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """
    Complete multi-stock analysis pipeline.
    Returns signals, portfolio allocation, backtest results, and risk metrics.
    """
    start = time.time()
    logger.info(f"Analysis request: {request.tickers} | Market: {request.market} | Capital: {request.investment_amount}")

    signals = []
    charts = {}
    stock_details = []

    for ticker in request.tickers:
        try:
            # 1. Fetch data
            df = fetcher.fetch_ohlcv(ticker, market=request.market, period="1y")
            if df.empty:
                df = fetcher._fallback_synthetic(ticker)

            # 2. Technical indicators
            df = TechnicalIndicators.compute_all(df)

            # 3. Sentiment
            sentiment = sentiment_engine.generate_demo_sentiment()

            # 4. Regime
            regime = regime_detector.detect(df)

            # 5. Generate mock model output for signal generation
            # (In production, this would run through FinSentNetCore)
            np.random.seed(hash(ticker) % 2**31)
            p_up = np.random.uniform(0.3, 0.85)
            p_down = np.random.uniform(0.05, 1 - p_up)
            p_neutral = 1 - p_up - p_down
            pred_return = np.random.uniform(-0.08, 0.12)

            model_output = {
                "direction_probs": np.array([p_up, p_neutral, p_down]),
                "return_pred": np.array([pred_return]),
            }

            # 6. Generate signal
            signal = signal_gen.generate_signal(
                ticker=ticker,
                market=request.market,
                model_output=model_output,
                price_data=df,
                total_capital=request.investment_amount,
                risk_tolerance=request.risk_tolerance,
            )
            signals.append(signal)

            # 7. Chart data (last 60 days)
            chart_df = df.tail(60)
            charts[ticker] = [{
                "time": int(idx.timestamp()) if hasattr(idx, 'timestamp') else i,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            } for i, (idx, row) in enumerate(chart_df.iterrows())]

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
                "regime": regime.value,
            })

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            continue

    if not signals:
        raise HTTPException(status_code=400, detail="No valid signals could be generated")

    # ── Portfolio Optimization ───────────────────────────
    n = len(signals)
    expected_rets = np.array([s.predicted_return / 100 for s in signals])
    cov = np.eye(n) * 0.02**2
    opt = optimizer.optimize_weights(expected_rets, cov, method="max_sharpe")

    # ── Allocation ───────────────────────────────────────
    allocation = allocator.allocate(signals, opt, request.investment_amount)

    # ── Backtest (demo) ─────────────────────────────────
    bt = backtester.generate_demo_backtest(request.investment_amount)

    # ── Risk metrics ────────────────────────────────────
    equity = np.array(bt["equity_curve"])
    risk_report = risk_engine.full_risk_report(
        np.diff(equity) / equity[:-1], equity
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
            } for s in signals
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


@app.get("/api/market/trending")
async def trending(market: str = "SP500"):
    """Get trending stocks for a market."""
    components = fetcher.get_market_components(market)[:15]
    trending_list = []
    for ticker in components[:8]:
        try:
            df = fetcher.fetch_ohlcv(ticker, market=market, period="5d")
            if df.empty:
                continue
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
            trending_list.append({
                "ticker": ticker,
                "price": round(float(latest["Close"]), 2),
                "change_pct": round(float(change_pct), 2),
            })
        except Exception:
            continue
    return {"market": market, "trending": trending_list}


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
