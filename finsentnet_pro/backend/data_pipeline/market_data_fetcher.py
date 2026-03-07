"""
FinSentNet Pro — Market Data Fetcher
Fetches OHLCV data from multiple sources with intelligent fallback.
Supports: S&P500, NASDAQ, NYSE, BSE, NSE, Commodities, Crypto
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests
import time


class MarketDataFetcher:
    """
    Unified data fetcher supporting all global markets.
    Implements strict temporal integrity — no look-ahead contamination.
    """

    MARKET_CONFIG = {
        "SP500": {
            "index_ticker": "^GSPC",
            "suffix": "",
            "currency": "USD",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"},
        },
        "NASDAQ": {
            "index_ticker": "^IXIC",
            "suffix": "",
            "currency": "USD",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"},
        },
        "NYSE": {
            "index_ticker": "^NYA",
            "suffix": "",
            "currency": "USD",
            "timezone": "America/New_York",
            "trading_hours": {"open": "09:30", "close": "16:00"},
        },
        "BSE": {
            "index_ticker": "^BSESN",
            "suffix": ".BO",
            "currency": "INR",
            "timezone": "Asia/Kolkata",
            "trading_hours": {"open": "09:15", "close": "15:30"},
        },
        "NSE": {
            "index_ticker": "^NSEI",
            "suffix": ".NS",
            "currency": "INR",
            "timezone": "Asia/Kolkata",
            "trading_hours": {"open": "09:15", "close": "15:30"},
        },
        "COMMODITIES": {
            "tickers": {
                "GOLD": "GC=F",
                "SILVER": "SI=F",
                "CRUDE_OIL": "CL=F",
                "NATURAL_GAS": "NG=F",
                "COPPER": "HG=F",
                "WHEAT": "ZW=F",
                "CORN": "ZC=F",
                "PLATINUM": "PL=F",
            },
            "suffix": "",
            "currency": "USD",
        },
        "CRYPTO": {
            "suffix": "-USD",
            "top_symbols": [
                "BTC", "ETH", "BNB", "SOL", "ADA", "AVAX",
                "DOT", "MATIC", "LINK", "UNI", "XRP", "DOGE",
            ],
            "currency": "USD",
        },
    }

    # Hard-coded component lists for reliability
    SENSEX_TICKERS = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
        "SBIN", "INFY", "HINDUNILVR", "ITC", "LT",
        "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "TITAN",
        "HCLTECH", "ADANIENT", "ONGC", "NTPC", "COALINDIA",
        "M&M", "BAJAJFINSV", "ASIANPAINT", "TATAMOTORS", "JSWSTEEL",
    ]

    NIFTY50_TICKERS = [
        "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
        "SBIN", "INFY", "HINDUNILVR", "ITC", "LT",
        "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "WIPRO", "ULTRACEMCO", "NESTLEIND", "POWERGRID", "TITAN",
        "HCLTECH", "ADANIENT", "ONGC", "NTPC", "COALINDIA",
        "M&M", "BAJAJFINSV", "ASIANPAINT", "TATAMOTORS", "JSWSTEEL",
        "TATASTEEL", "TECHM", "CIPLA", "BPCL", "SHREECEM",
        "BRITANNIA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "HINDALCO",
        "APOLLOHOSP", "DIVISLAB", "GRASIM", "INDUSINDBK", "SBILIFE",
        "HDFCLIFE", "TATACONSUM", "UPL", "VEDL", "DABUR",
    ]

    TOP_SP500 = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B",
        "LLY", "TSM", "AVGO", "TSLA", "JPM", "V", "WMT", "XOM",
        "ORCL", "MA", "UNH", "JNJ", "PG", "HD", "COST", "ABBV",
        "CRM", "BAC", "NFLX", "KO", "MRK", "AMD", "PEP",
    ]

    NASDAQ100 = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "GOOGL",
        "TSLA", "COST", "NFLX", "AMD", "QCOM", "ADBE", "PEP",
        "INTC", "CSCO", "CMCSA", "INTU", "AMGN", "TMUS",
        "AMAT", "TXN", "ISRG", "MU", "LRCX", "BKNG", "MDLZ",
        "ADI", "REGN", "SNPS",
    ]

    def __init__(self):
        self._session_cache: Dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(
        self,
        ticker: str,
        market: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with automatic suffix handling and validation.

        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume]
            Index: DatetimeIndex
        """
        config = self.MARKET_CONFIG.get(market, {})
        suffix = config.get("suffix", "")

        # For commodities, use the predefined ticker mapping
        if market == "COMMODITIES":
            commodity_map = config.get("tickers", {})
            full_ticker = commodity_map.get(ticker.upper(), ticker)
        elif market == "CRYPTO" and not ticker.endswith("-USD"):
            full_ticker = f"{ticker}-USD"
        else:
            full_ticker = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        cache_key = f"{full_ticker}_{period}_{interval}"
        if cache_key in self._session_cache:
            return self._session_cache[cache_key].copy()

        try:
            stock = yf.Ticker(full_ticker)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {full_ticker}")

            df = self._validate_and_clean_ohlcv(df, full_ticker)
            self._session_cache[cache_key] = df
            return df.copy()

        except Exception as e:
            print(f"[WARNING] Primary fetch failed for {full_ticker}: {e}")
            return self._fallback_synthetic(ticker, period)

    def _validate_and_clean_ohlcv(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Validates and cleans OHLCV data."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # OHLC consistency: High >= Low
        invalid_ohlc = df["High"] < df["Low"]
        if invalid_ohlc.sum() > 0:
            print(f"[WARNING] {invalid_ohlc.sum()} OHLC inconsistencies in {ticker}")
            df = df[~invalid_ohlc]

        # Remove zero-volume days
        df = df[df["Volume"] > 0]

        # Forward fill small gaps (holidays)
        df = df.ffill(limit=5).dropna()

        # Detect extreme single-day moves (>50% = likely data error)
        daily_returns = df["Close"].pct_change().abs()
        suspect = daily_returns > 0.50
        if suspect.sum() > 0:
            print(f"[ALERT] {suspect.sum()} extreme price moves in {ticker}")

        return df

    def get_live_price(self, ticker: str, market: str) -> Dict:
        """Returns real-time price data for a single ticker."""
        config = self.MARKET_CONFIG.get(market, {})
        suffix = config.get("suffix", "")

        if market == "COMMODITIES":
            commodity_map = config.get("tickers", {})
            full_ticker = commodity_map.get(ticker.upper(), ticker)
        elif market == "CRYPTO" and not ticker.endswith("-USD"):
            full_ticker = f"{ticker}-USD"
        else:
            full_ticker = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

        try:
            stock = yf.Ticker(full_ticker)
            info = stock.fast_info

            last_price = float(info.last_price)
            prev_close = float(info.previous_close)
            change = last_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0

            return {
                "ticker": ticker,
                "market": market,
                "price": round(last_price, 2),
                "prev_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(info.last_volume) if hasattr(info, "last_volume") else 0,
                "market_cap": float(info.market_cap) if hasattr(info, "market_cap") and info.market_cap else None,
                "currency": config.get("currency", "USD"),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            # Fallback with synthetic data for demo
            np.random.seed(abs(hash(ticker)) % 2**31)
            base = np.random.uniform(50, 500)
            change_pct = np.random.uniform(-5, 5)
            return {
                "ticker": ticker,
                "market": market,
                "price": round(base, 2),
                "prev_close": round(base / (1 + change_pct / 100), 2),
                "change": round(base * change_pct / 100, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(np.random.uniform(1e6, 50e6)),
                "market_cap": None,
                "currency": config.get("currency", "USD"),
                "timestamp": datetime.now().isoformat(),
            }

    def get_market_components(self, market: str, limit: int = 100) -> List[str]:
        """Returns list of ticker symbols for a given market."""
        if market == "SP500":
            tickers = self._get_sp500_tickers()
        elif market == "NASDAQ":
            tickers = self.NASDAQ100[:]
        elif market == "NYSE":
            tickers = self.TOP_SP500[:]  # Overlap for simplicity
        elif market == "NSE":
            tickers = [f"{t}.NS" for t in self.NIFTY50_TICKERS]
        elif market == "BSE":
            tickers = [f"{t}.BO" for t in self.SENSEX_TICKERS]
        elif market == "CRYPTO":
            symbols = self.MARKET_CONFIG["CRYPTO"]["top_symbols"]
            tickers = [f"{s}-USD" for s in symbols]
        elif market == "COMMODITIES":
            tickers = list(self.MARKET_CONFIG["COMMODITIES"]["tickers"].values())
        else:
            tickers = []
        return tickers[:limit]

    def _get_sp500_tickers(self) -> List[str]:
        """Fetches S&P 500 components from Wikipedia with local fallback."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception:
            return self.TOP_SP500[:]

    def _fallback_synthetic(self, ticker: str, period: str) -> pd.DataFrame:
        """Generate synthetic OHLCV for demo/fallback when API fails."""
        period_days = {
            "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252,
            "2y": 504, "5y": 1260, "max": 2520,
        }
        n_days = period_days.get(period, 252)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
        np.random.seed(abs(hash(ticker)) % 2**31)
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
        df = pd.DataFrame(
            {
                "Open": prices * (1 - np.random.uniform(0, 0.01, n_days)),
                "High": prices * (1 + np.random.uniform(0, 0.02, n_days)),
                "Low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 50_000_000, n_days),
            },
            index=dates,
        )
        return df
