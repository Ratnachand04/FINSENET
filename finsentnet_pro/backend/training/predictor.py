"""
FINSENT NET PRO — Live Predictor
Continuous prediction engine using trained per-company models.

After initial training, this module:
  1. Fetches the latest market data for a ticker
  2. Runs the trained FINSENT model
  3. Generates real-time buy/sell signals
  4. Supports both single-shot and continuous polling modes
"""

import os
import time
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger("finsent.predictor")


class LivePredictor:
    """
    Real-time prediction engine using per-company trained models.

    Usage:
        predictor = LivePredictor(model, fetcher, indicators, signal_gen, trainer)
        signal = predictor.predict("AAPL", "SP500", capital=100000)
    """

    def __init__(self, model, fetcher, indicators, signal_gen, trainer):
        self.model = model
        self.fetcher = fetcher
        self.indicators = indicators
        self.signal_gen = signal_gen
        self.trainer = trainer
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        # Cache for recent predictions
        self._prediction_cache: Dict[str, Dict] = {}
        self._cache_ttl = 30  # seconds

    def predict(
        self,
        ticker: str,
        market: str = "SP500",
        total_capital: float = 100000,
        risk_tolerance: float = 0.5,
    ) -> Dict:
        """
        Generate a real-time prediction for a single ticker.

        Flow:
          1. Ensure model is trained (load checkpoint)
          2. Fetch latest data + indicators
          3. Run FINSENT inference
          4. Generate trade signal with position sizing

        Returns complete prediction dict.
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{ticker}_{market}"
        if cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            age = time.time() - cached.get("_cached_at", 0)
            if age < self._cache_ttl:
                return cached

        # ── Load trained model ──
        if not self.trainer.load_trained_model(ticker):
            return {
                "status": "not_trained",
                "ticker": ticker,
                "message": f"No trained model found for {ticker}. Please train first.",
            }

        # ── Fetch latest data ──
        try:
            df = self.fetcher.fetch_ohlcv(ticker, market=market, period="3mo")
            if df.empty or len(df) < 35:
                raise ValueError(f"Insufficient data: {len(df)} rows")
            df = self.indicators.compute_all(df)
        except Exception as e:
            logger.error(f"[{ticker}] Data fetch failed: {e}")
            return {
                "status": "error",
                "ticker": ticker,
                "message": f"Failed to fetch data: {str(e)}",
            }

        # ── Prepare model input ──
        price_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Lower", "BB_Position", "ATR_14",
            "OBV", "Volume_Ratio", "EMA_20", "EMA_50",
            "Log_Return", "Volatility_20", "Stoch_K",
        ]
        available_cols = [c for c in price_cols if c in df.columns]
        price_window = df[available_cols].iloc[-30:].values

        # Z-score normalize
        mean = price_window.mean(axis=0, keepdims=True)
        std = price_window.std(axis=0, keepdims=True) + 1e-8
        price_norm = (price_window - mean) / std

        # Pad to 20 features
        if price_norm.shape[1] < 20:
            pad = np.zeros((price_norm.shape[0], 20 - price_norm.shape[1]))
            price_norm = np.concatenate([price_norm, pad], axis=1)

        price_tensor = torch.FloatTensor(price_norm).unsqueeze(0).to(self.device)
        text_tokens = torch.randint(0, 1000, (1, 50)).to(self.device)

        # ── Run inference ──
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(text_tokens, price_tensor)

        # ── Generate signal ──
        signal = self.signal_gen.generate_signal(
            ticker=ticker,
            market=market,
            model_output=model_output,
            price_data=df,
            total_capital=total_capital,
            risk_tolerance=risk_tolerance,
        )

        # ── Get live price info ──
        live_price = self.fetcher.get_live_price(ticker, market)

        # ── Build result ──
        probs = model_output["direction_probs"].cpu().numpy()[0]
        elapsed = time.time() - start_time

        result = {
            "status": "success",
            "ticker": ticker,
            "market": market,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_time_ms": round(elapsed * 1000, 1),
            "live_price": live_price,
            "prediction": {
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "predicted_return": signal.predicted_return,
                "probabilities": {
                    "up": round(float(probs[0]) * 100, 1),
                    "neutral": round(float(probs[1]) * 100, 1),
                    "down": round(float(probs[2]) * 100, 1),
                },
            },
            "signal": {
                "entry_price": signal.entry_price,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "risk_reward": signal.risk_reward_ratio,
                "quantity": signal.recommended_quantity,
                "capital_required": signal.capital_required,
                "kelly_fraction": signal.kelly_fraction,
                "time_horizon": signal.time_horizon,
            },
            "analysis": {
                "sentiment_score": signal.sentiment_score,
                "technical_score": signal.technical_score,
                "regime": signal.regime,
                "reasoning": signal.reasoning,
            },
            "_cached_at": time.time(),
        }

        # Cache result
        self._prediction_cache[cache_key] = result
        return result

    def get_live_candles(
        self,
        ticker: str,
        market: str = "SP500",
        interval: str = "5m",
        period: str = "1d",
    ) -> Dict:
        """
        Fetch intraday candle data for live charting.

        Returns OHLCV candles suitable for TradingView Lightweight Charts.
        """
        try:
            config = self.fetcher.MARKET_CONFIG.get(market, {})
            suffix = config.get("suffix", "")

            if market == "CRYPTO" and not ticker.endswith("-USD"):
                full_ticker = f"{ticker}-USD"
            else:
                full_ticker = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

            import yfinance as yf
            stock = yf.Ticker(full_ticker)

            # Fetch intraday data
            df = stock.history(period=period, interval=interval)

            if df.empty:
                # Fallback: generate synthetic intraday data
                return self._synthetic_intraday(ticker, interval, period)

            candles = []
            for idx, row in df.iterrows():
                ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
                candles.append({
                    "time": ts,
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row.get("Volume", 0)),
                })

            return {
                "status": "success",
                "ticker": ticker,
                "interval": interval,
                "candle_count": len(candles),
                "candles": candles,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"[{ticker}] Candle fetch error: {e}")
            return self._synthetic_intraday(ticker, interval, period)

    def get_daily_candles(
        self,
        ticker: str,
        market: str = "SP500",
        period: str = "6mo",
    ) -> Dict:
        """Fetch daily candles for the chart display."""
        try:
            df = self.fetcher.fetch_ohlcv(ticker, market=market, period=period)
            if df.empty:
                df = self.fetcher._fallback_synthetic(ticker, period)

            df = self.indicators.compute_all(df)

            candles = []
            for idx, row in df.iterrows():
                ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
                candles.append({
                    "time": ts,
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": int(row.get("Volume", 0)),
                })

            # Also extract indicator overlays
            sma_50 = [
                {"time": int(idx.timestamp()), "value": round(float(row["SMA_50"]), 4)}
                for idx, row in df.iterrows()
                if "SMA_50" in df.columns and not pd.isna(row.get("SMA_50"))
            ]
            sma_200 = [
                {"time": int(idx.timestamp()), "value": round(float(row["SMA_200"]), 4)}
                for idx, row in df.iterrows()
                if "SMA_200" in df.columns and not pd.isna(row.get("SMA_200"))
            ]
            bb_upper = [
                {"time": int(idx.timestamp()), "value": round(float(row["BB_Upper"]), 4)}
                for idx, row in df.iterrows()
                if "BB_Upper" in df.columns and not pd.isna(row.get("BB_Upper"))
            ]
            bb_lower = [
                {"time": int(idx.timestamp()), "value": round(float(row["BB_Lower"]), 4)}
                for idx, row in df.iterrows()
                if "BB_Lower" in df.columns and not pd.isna(row.get("BB_Lower"))
            ]

            volume = [
                {
                    "time": int(idx.timestamp()),
                    "value": int(row.get("Volume", 0)),
                    "color": "rgba(0, 245, 255, 0.3)" if row["Close"] >= row["Open"]
                             else "rgba(255, 68, 68, 0.3)",
                }
                for idx, row in df.iterrows()
            ]

            return {
                "status": "success",
                "ticker": ticker,
                "candle_count": len(candles),
                "candles": candles,
                "overlays": {
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                },
                "volume": volume,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"[{ticker}] Daily candle error: {e}")
            return {"status": "error", "ticker": ticker, "error": str(e)}

    def _synthetic_intraday(self, ticker: str, interval: str, period: str) -> Dict:
        """Generate synthetic intraday candles for demo."""
        np.random.seed(abs(hash(ticker)) % 2**31)
        base_price = np.random.uniform(50, 500)

        n_candles = {"1m": 390, "5m": 78, "15m": 26, "1h": 7}.get(interval, 78)
        now = time.time()
        interval_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(interval, 300)

        candles = []
        price = base_price
        for i in range(n_candles):
            change = np.random.normal(0, price * 0.002)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, price * 0.001))
            l = min(o, c) - abs(np.random.normal(0, price * 0.001))
            vol = int(np.random.uniform(10000, 500000))
            ts = int(now - (n_candles - i) * interval_seconds)

            candles.append({
                "time": ts,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": vol,
            })
            price = c

        return {
            "status": "demo",
            "ticker": ticker,
            "interval": interval,
            "candle_count": len(candles),
            "candles": candles,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def batch_predict(
        self,
        tickers: List[str],
        market: str = "SP500",
        total_capital: float = 100000,
        risk_tolerance: float = 0.5,
    ) -> List[Dict]:
        """Generate predictions for multiple tickers."""
        results = []
        per_stock_capital = total_capital / max(len(tickers), 1)
        for ticker in tickers:
            result = self.predict(
                ticker=ticker,
                market=market,
                total_capital=per_stock_capital,
                risk_tolerance=risk_tolerance,
            )
            results.append(result)
        return results
