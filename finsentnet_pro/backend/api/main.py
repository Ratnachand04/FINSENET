"""
FINSENT NET PRO — FastAPI Backend
REST API serving all analysis endpoints.
AI-Powered Quantitative Trading Intelligence.
"""

import os
import sys
import logging
from datetime import datetime

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Fix imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.market_data_fetcher import MarketDataFetcher
from data_pipeline.technical_indicators import TechnicalIndicators
from data_pipeline.news_sentiment_engine import NewsSentimentEngine
from data_pipeline.regime_detector import RegimeDetector
from data_pipeline.data_aligner import DataAligner
from models.finsentnet_core import FinSentNetCore
from models.model_registry import ModelRegistry
from models.signal_generator import SignalGenerator
from portfolio.kelly_sizer import KellySizer
from portfolio.portfolio_optimizer import PortfolioOptimizer
from portfolio.risk_engine import RiskEngine
from portfolio.allocation_engine import AllocationEngine
from backtesting.backtest_engine import BacktestEngine
from training.trainer import ModelTrainer
from training.predictor import LivePredictor
from data_pipeline.live_data_service import LiveDataService

# Route modules
from api.routes import market_data as market_routes
from api.routes import analysis as analysis_routes
from api.routes import signals as signal_routes
from api.routes import portfolio as portfolio_routes
from api.routes import training as training_routes
from api.routes import live as live_routes
from api.routes import system as system_routes
from api.websocket_handler import websocket_endpoint

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finsent")

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="FINSENT NET PRO API",
    description="AI-Powered Quantitative Trading Intelligence",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Singletons ───────────────────────────────────────────
fetcher = MarketDataFetcher()
indicators = TechnicalIndicators()
sentiment_engine = NewsSentimentEngine(use_finbert=False)
regime_detector = RegimeDetector()
aligner = DataAligner()
model = FinSentNetCore()
model.eval()
model_registry = ModelRegistry()
signal_gen = SignalGenerator()
kelly = KellySizer()
optimizer = PortfolioOptimizer()
risk_engine = RiskEngine()
allocator = AllocationEngine()
backtester = BacktestEngine()
trainer = ModelTrainer(model, fetcher, indicators)
predictor = LivePredictor(model, fetcher, indicators, signal_gen, trainer)
live_data_service = LiveDataService()

# ── Initialize route modules with shared services ────────
market_routes.init(fetcher, indicators)
analysis_routes.init(
    fetcher, indicators, sentiment_engine, regime_detector, aligner,
    model, signal_gen, optimizer, allocator, backtester, risk_engine,
)
signal_routes.init(signal_gen, fetcher, indicators)
portfolio_routes.init(optimizer, risk_engine, allocator, kelly)
training_routes.init(trainer, predictor)
live_routes.init(live_data_service, predictor, trainer)
system_routes.init(model_registry)

# ── Register routers ─────────────────────────────────────
app.include_router(market_routes.router)
app.include_router(analysis_routes.router)
app.include_router(signal_routes.router)
app.include_router(portfolio_routes.router)
app.include_router(training_routes.router)
app.include_router(live_routes.router)
app.include_router(system_routes.router)


# ── Core endpoints ───────────────────────────────────────
@app.get("/")
async def root():
    """Serve frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "FINSENT NET PRO API v2.0 — Use /docs for API reference.",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "FINSENT NET PRO is operational",
        "version": "2.0.0",
        "model": "FINSENT Core v1.0",
        "active_model_profile": model_registry.get_active().profile_id,
        "timestamp": datetime.utcnow().isoformat(),
        "modules": {
            "data_fetcher": True,
            "technical_indicators": True,
            "sentiment_engine": True,
            "regime_detector": True,
            "finsentnet_model": True,
            "signal_generator": True,
            "portfolio_optimizer": True,
            "risk_engine": True,
            "backtester": True,
            "model_trainer": True,
            "live_predictor": True,
            "live_data_service": True,
        },
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time price streaming."""
    await websocket_endpoint(websocket, fetcher=fetcher)


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
