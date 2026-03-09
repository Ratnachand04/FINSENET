"""
FINSENT NET PRO — Live Data & Prediction Routes
/api/live/* — Real-time market data, candle charts, news, and predictions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/live", tags=["Live Data"])
logger = logging.getLogger("finsent.routes.live")

# ── Singletons (injected from main.py) ─────────────
_live_service = None
_predictor = None
_trainer = None


def init(live_service, predictor, trainer):
    """Initialize route with shared services."""
    global _live_service, _predictor, _trainer
    _live_service = live_service
    _predictor = predictor
    _trainer = trainer


# ═══════════════════════════════════════════════════════
#  Request Models
# ═══════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    ticker: str
    market: str = "SP500"
    capital: float = Field(default=100000, ge=1000)
    risk_tolerance: float = Field(default=0.5, ge=0.1, le=1.0)


class ApiKeyRequest(BaseModel):
    alpha_vantage: Optional[str] = None
    finnhub: Optional[str] = None
    news_api: Optional[str] = None


# ═══════════════════════════════════════════════════════
#  REAL-TIME QUOTE
# ═══════════════════════════════════════════════════════

@router.get("/quote/{ticker}")
async def get_realtime_quote(ticker: str, market: str = "SP500"):
    """
    Get real-time price quote for a ticker.
    Uses Finnhub (if API key set) → yfinance fallback.
    """
    if not _live_service:
        raise HTTPException(status_code=503, detail="Live data service not ready")

    try:
        quote = _live_service.get_realtime_quote(ticker.upper(), market)
        return {"status": "success", **quote}
    except Exception as e:
        logger.error(f"Quote error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════
#  CANDLE DATA (for TradingView Lightweight Charts)
# ═══════════════════════════════════════════════════════

@router.get("/candles/{ticker}")
async def get_intraday_candles(
    ticker: str,
    market: str = "SP500",
    interval: str = Query(default="5m", description="1m, 5m, 15m, 1h"),
    period: str = Query(default="1d", description="1d, 5d"),
):
    """
    Fetch intraday candle data for live candlestick charts.
    Returns OHLCV data formatted for TradingView Lightweight Charts.
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    try:
        data = _predictor.get_live_candles(
            ticker=ticker.upper(),
            market=market,
            interval=interval,
            period=period,
        )
        return data
    except Exception as e:
        logger.error(f"Candle error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily/{ticker}")
async def get_daily_candles(
    ticker: str,
    market: str = "SP500",
    period: str = Query(default="6mo", description="3mo, 6mo, 1y, 2y"),
):
    """
    Fetch daily candle data with indicator overlays (SMA, BB).
    Returns data for the main chart display with volume histogram.
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    try:
        data = _predictor.get_daily_candles(
            ticker=ticker.upper(),
            market=market,
            period=period,
        )
        return data
    except Exception as e:
        logger.error(f"Daily candle error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════
#  NEWS
# ═══════════════════════════════════════════════════════

@router.get("/news/{ticker}")
async def get_live_news(ticker: str, market: str = "SP500"):
    """
    Fetch latest news headlines for a ticker.
    Sources: Finnhub → NewsAPI → yfinance → demo fallback.
    """
    if not _live_service:
        raise HTTPException(status_code=503, detail="Live data service not ready")

    try:
        data = _live_service.get_live_news(ticker.upper(), market)
        return {"status": "success", **data}
    except Exception as e:
        logger.error(f"News error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════
#  PREDICTION (using trained model)
# ═══════════════════════════════════════════════════════

@router.post("/predict")
async def predict_signal(request: PredictRequest):
    """
    Generate a real-time prediction/signal using the trained model.
    The model must be trained first via /api/train/start.
    Returns direction, confidence, entry/target/stop-loss prices.
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    ticker = request.ticker.upper().strip()

    result = _predictor.predict(
        ticker=ticker,
        market=request.market,
        total_capital=request.capital,
        risk_tolerance=request.risk_tolerance,
    )

    if result.get("status") == "not_trained":
        raise HTTPException(
            status_code=400,
            detail=f"Model not trained for {ticker}. POST /api/train/start first.",
        )

    return result


@router.get("/predict/{ticker}")
async def predict_signal_get(
    ticker: str,
    market: str = "SP500",
    capital: float = 100000,
    risk_tolerance: float = 0.5,
):
    """
    GET variant of prediction endpoint (for easy polling).
    """
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    ticker = ticker.upper().strip()

    result = _predictor.predict(
        ticker=ticker,
        market=market,
        total_capital=capital,
        risk_tolerance=risk_tolerance,
    )

    if result.get("status") == "not_trained":
        raise HTTPException(
            status_code=400,
            detail=f"Model not trained for {ticker}. POST /api/train/start first.",
        )

    return result


@router.post("/predict/batch")
async def batch_predict(
    tickers: list,
    market: str = "SP500",
    capital: float = 100000,
    risk_tolerance: float = 0.5,
):
    """Generate predictions for multiple tickers."""
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")

    results = _predictor.batch_predict(
        tickers=[t.upper().strip() for t in tickers],
        market=market,
        total_capital=capital,
        risk_tolerance=risk_tolerance,
    )
    return {"results": results, "count": len(results)}


# ═══════════════════════════════════════════════════════
#  API KEY CONFIGURATION
# ═══════════════════════════════════════════════════════

@router.post("/settings/api-keys")
async def configure_api_keys(request: ApiKeyRequest):
    """
    Set API keys at runtime for enhanced data sources.
    Keys are stored in memory only (not persisted).
    """
    if not _live_service:
        raise HTTPException(status_code=503, detail="Live data service not ready")

    _live_service.configure_api_keys(
        alpha_vantage=request.alpha_vantage,
        finnhub=request.finnhub,
        news_api=request.news_api,
    )

    return {
        "status": "ok",
        "message": "API keys updated",
        "sources": {
            "alpha_vantage": bool(_live_service.alpha_vantage_key),
            "finnhub": bool(_live_service.finnhub_key),
            "news_api": bool(_live_service.news_api_key),
        },
    }


@router.get("/settings/api-keys")
async def get_api_key_status():
    """Check which API keys are configured (does NOT reveal keys)."""
    if not _live_service:
        return {"sources": {}}

    return {
        "sources": {
            "alpha_vantage": bool(_live_service.alpha_vantage_key),
            "finnhub": bool(_live_service.finnhub_key),
            "news_api": bool(_live_service.news_api_key),
            "yfinance": True,  # always available
        },
    }
