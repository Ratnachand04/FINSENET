"""
FINSENT NET PRO — Live Data Service
Multi-source real-time market data and news fetching.

Data Sources (priority order):
  1. Yahoo Finance (yfinance) — free, no API key needed
  2. Alpha Vantage — intraday OHLCV (free key: 25 req/day)
  3. Finnhub — real-time quotes + news (free key: 60 req/min)
  4. NewsAPI — financial headlines (free key: 100 req/day)

API keys are loaded from environment variables:
  ALPHA_VANTAGE_API_KEY
  FINNHUB_API_KEY
  NEWS_API_KEY
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("finsent.live_data")


class LiveDataService:
    """
    Unified live data provider for FINSENT NET PRO.
    Works out of the box with yfinance (no API key needed).
    Enhanced with optional premium APIs for better real-time data.
    """

    def __init__(self):
        # Load API keys from environment
        self.alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        self.news_api_key = os.environ.get("NEWS_API_KEY", "")

        # Rate limiting
        self._last_request_time: Dict[str, float] = {}
        self._min_interval = {"alpha_vantage": 12, "finnhub": 1, "newsapi": 1}

        # Cache
        self._news_cache: Dict[str, Dict] = {}
        self._news_cache_ttl = 300  # 5 min

        logger.info(
            f"LiveDataService initialized | "
            f"Alpha Vantage: {'✓' if self.alpha_vantage_key else '✗'} | "
            f"Finnhub: {'✓' if self.finnhub_key else '✗'} | "
            f"NewsAPI: {'✓' if self.news_api_key else '✗'}"
        )

    def configure_api_keys(
        self,
        alpha_vantage: Optional[str] = None,
        finnhub: Optional[str] = None,
        news_api: Optional[str] = None,
    ):
        """Set API keys at runtime."""
        if alpha_vantage:
            self.alpha_vantage_key = alpha_vantage
        if finnhub:
            self.finnhub_key = finnhub
        if news_api:
            self.news_api_key = news_api
        logger.info("API keys updated")

    # ═══════════════════════════════════════════════════════
    #  REAL-TIME PRICE
    # ═══════════════════════════════════════════════════════

    def get_realtime_quote(self, ticker: str, market: str = "SP500") -> Dict:
        """
        Get real-time price quote. Tries Finnhub first, falls back to yfinance.
        """
        # Try Finnhub (fastest real-time)
        if self.finnhub_key:
            try:
                result = self._finnhub_quote(ticker)
                if result and result.get("price", 0) > 0:
                    return result
            except Exception as e:
                logger.debug(f"Finnhub quote failed for {ticker}: {e}")

        # Fallback: yfinance
        try:
            return self._yfinance_quote(ticker, market)
        except Exception as e:
            logger.error(f"All quote sources failed for {ticker}: {e}")
            return self._synthetic_quote(ticker)

    def _finnhub_quote(self, ticker: str) -> Dict:
        """Fetch real-time quote from Finnhub."""
        self._rate_limit("finnhub")
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.finnhub_key}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data.get("c", 0) == 0:
            return {}

        return {
            "ticker": ticker,
            "price": round(data["c"], 2),
            "open": round(data["o"], 2),
            "high": round(data["h"], 2),
            "low": round(data["l"], 2),
            "prev_close": round(data["pc"], 2),
            "change": round(data["c"] - data["pc"], 2),
            "change_pct": round((data["c"] - data["pc"]) / data["pc"] * 100, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "finnhub",
        }

    def _yfinance_quote(self, ticker: str, market: str) -> Dict:
        """Fetch quote using yfinance."""
        import yfinance as yf

        suffix_map = {
            "BSE": ".BO", "NSE": ".NS", "CRYPTO": "-USD",
        }
        suffix = suffix_map.get(market, "")
        full = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        stock = yf.Ticker(full)
        info = stock.fast_info

        price = float(info.last_price)
        prev = float(info.previous_close)

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "open": round(float(info.open) if hasattr(info, "open") else price, 2),
            "high": round(float(info.day_high) if hasattr(info, "day_high") else price, 2),
            "low": round(float(info.day_low) if hasattr(info, "day_low") else price, 2),
            "prev_close": round(prev, 2),
            "change": round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2) if prev else 0,
            "volume": int(info.last_volume) if hasattr(info, "last_volume") else 0,
            "market_cap": float(info.market_cap) if hasattr(info, "market_cap") and info.market_cap else None,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "yfinance",
        }

    def _synthetic_quote(self, ticker: str) -> Dict:
        """Generate synthetic quote for demo."""
        np.random.seed(abs(hash(ticker)) % 2**31)
        price = round(np.random.uniform(50, 500), 2)
        change_pct = round(np.random.uniform(-3, 3), 2)
        return {
            "ticker": ticker,
            "price": price,
            "prev_close": round(price / (1 + change_pct / 100), 2),
            "change": round(price * change_pct / 100, 2),
            "change_pct": change_pct,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "synthetic",
        }

    # ═══════════════════════════════════════════════════════
    #  INTRADAY CANDLES
    # ═══════════════════════════════════════════════════════

    def get_intraday_candles(
        self,
        ticker: str,
        market: str = "SP500",
        interval: str = "5min",
        output_size: str = "compact",
    ) -> List[Dict]:
        """
        Fetch intraday OHLCV candles.
        Tries Alpha Vantage first, falls back to yfinance.
        """
        # Try Alpha Vantage (detailed intraday)
        if self.alpha_vantage_key:
            try:
                candles = self._alpha_vantage_intraday(ticker, interval)
                if candles:
                    return candles
            except Exception as e:
                logger.debug(f"Alpha Vantage intraday failed: {e}")

        # Fallback: yfinance intraday
        try:
            return self._yfinance_intraday(ticker, market, interval)
        except Exception as e:
            logger.error(f"All intraday sources failed for {ticker}: {e}")
            return []

    def _alpha_vantage_intraday(self, ticker: str, interval: str) -> List[Dict]:
        """Fetch intraday data from Alpha Vantage."""
        self._rate_limit("alpha_vantage")

        av_interval = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "60min",
            "1min": "1min", "5min": "5min", "15min": "15min",
            "30min": "30min", "60min": "60min",
        }.get(interval, "5min")

        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_INTRADAY"
            f"&symbol={ticker}&interval={av_interval}"
            f"&outputsize=compact&apikey={self.alpha_vantage_key}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        ts_key = f"Time Series ({av_interval})"
        if ts_key not in data:
            return []

        candles = []
        for timestamp_str, values in data[ts_key].items():
            ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            candles.append({
                "time": int(ts.timestamp()),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
            })

        candles.sort(key=lambda x: x["time"])
        return candles

    def _yfinance_intraday(self, ticker: str, market: str, interval: str) -> List[Dict]:
        """Fetch intraday candles from yfinance."""
        import yfinance as yf

        suffix_map = {"BSE": ".BO", "NSE": ".NS", "CRYPTO": "-USD"}
        suffix = suffix_map.get(market, "")
        full = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        # yfinance interval format
        yf_interval = {
            "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60min": "1h",
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
        }.get(interval, "5m")

        # yfinance limits: 1m data only for 7 days, 5m for 60 days
        period = "1d" if yf_interval == "1m" else "5d"

        stock = yf.Ticker(full)
        df = stock.history(period=period, interval=yf_interval)

        if df.empty:
            return []

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": int(idx.timestamp()),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row.get("Volume", 0)),
            })

        return candles

    # ═══════════════════════════════════════════════════════
    #  NEWS
    # ═══════════════════════════════════════════════════════

    def get_live_news(self, ticker: str, market: str = "SP500") -> Dict:
        """
        Fetch latest news headlines for a ticker.
        Tries: Finnhub → NewsAPI → yfinance → Demo fallback
        """
        cache_key = f"news_{ticker}"
        if cache_key in self._news_cache:
            cached = self._news_cache[cache_key]
            if time.time() - cached.get("_ts", 0) < self._news_cache_ttl:
                return cached

        articles = []

        # 1) Finnhub company news
        if self.finnhub_key:
            try:
                articles = self._finnhub_news(ticker)
            except Exception:
                pass

        # 2) NewsAPI
        if not articles and self.news_api_key:
            try:
                articles = self._newsapi_search(ticker)
            except Exception:
                pass

        # 3) yfinance news
        if not articles:
            try:
                articles = self._yfinance_news(ticker, market)
            except Exception:
                pass

        # 4) Fallback demo
        if not articles:
            articles = self._demo_news(ticker)

        result = {
            "ticker": ticker,
            "article_count": len(articles),
            "articles": articles[:20],
            "timestamp": datetime.utcnow().isoformat(),
            "_ts": time.time(),
        }
        self._news_cache[cache_key] = result
        return result

    def _finnhub_news(self, ticker: str) -> List[Dict]:
        """Fetch company news from Finnhub."""
        self._rate_limit("finnhub")
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={week_ago}&to={today}"
            f"&token={self.finnhub_key}"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
            }
            for item in data[:20]
        ]

    def _newsapi_search(self, ticker: str) -> List[Dict]:
        """Search for news via NewsAPI."""
        self._rate_limit("newsapi")
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock&language=en"
            f"&sortBy=publishedAt&pageSize=20"
            f"&apiKey={self.news_api_key}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "title": art.get("title", ""),
                "summary": art.get("description", ""),
                "source": art.get("source", {}).get("name", ""),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
            }
            for art in data.get("articles", [])
        ]

    def _yfinance_news(self, ticker: str, market: str) -> List[Dict]:
        """Fetch news from yfinance."""
        import yfinance as yf

        suffix_map = {"BSE": ".BO", "NSE": ".NS", "CRYPTO": "-USD"}
        suffix = suffix_map.get(market, "")
        full = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        stock = yf.Ticker(full)
        news = stock.news if hasattr(stock, "news") else []

        return [
            {
                "title": item.get("title", ""),
                "summary": item.get("title", ""),
                "source": item.get("publisher", ""),
                "url": item.get("link", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).isoformat() if item.get("providerPublishTime") else "",
            }
            for item in (news or [])[:20]
        ]

    def _demo_news(self, ticker: str) -> List[Dict]:
        """Generate demo news for display."""
        headlines = [
            f"{ticker} reports strong quarterly earnings, beating Wall Street estimates",
            f"Analysts upgrade {ticker} citing robust growth pipeline and market momentum",
            f"{ticker} announces strategic acquisition, expanding market presence",
            f"Market outlook for {ticker}: Institutional investors increase positions",
            f"{ticker} unveils new product lineup, shares respond positively",
            f"Technical analysis: {ticker} breaks above key resistance level",
            f"Options activity surges for {ticker} ahead of earnings",
            f"{ticker} CEO outlines 5-year growth strategy at investor conference",
        ]
        return [
            {
                "title": h,
                "summary": h,
                "source": "FINSENT Research",
                "url": "#",
                "published_at": datetime.utcnow().isoformat(),
            }
            for h in headlines
        ]

    # ═══════════════════════════════════════════════════════
    #  UTILITIES
    # ═══════════════════════════════════════════════════════

    def _rate_limit(self, source: str):
        """Simple rate limiting per source."""
        now = time.time()
        min_gap = self._min_interval.get(source, 1)
        last = self._last_request_time.get(source, 0)
        wait = min_gap - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request_time[source] = time.time()
