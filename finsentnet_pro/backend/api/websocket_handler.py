"""
FINSENT NET PRO — WebSocket Handler
Real-time price streaming via WebSocket connections.
"""

import asyncio
import json
import logging
from typing import Set, Dict
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("finsent.websocket")


class ConnectionManager:
    """Manages active WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, tickers: list):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(tickers)

    def unsubscribe(self, websocket: WebSocket, tickers: list):
        if websocket in self.subscriptions:
            self.subscriptions[websocket] -= set(tickers)

    async def broadcast_price(self, ticker: str, price_data: dict):
        """Send price update to all subscribers of a ticker."""
        dead = []
        for ws, subs in self.subscriptions.items():
            if ticker in subs:
                try:
                    await ws.send_json({"type": "price_update", "data": price_data})
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, fetcher=None):
    """
    WebSocket endpoint for real-time price streaming.

    Client messages:
        {"action": "subscribe",   "tickers": ["AAPL", "MSFT"]}
        {"action": "unsubscribe", "tickers": ["AAPL"]}

    Server messages:
        {"type": "price_update", "data": {ticker, price, change_pct, ...}}
        {"type": "signal_alert", "data": {ticker, direction, confidence, ...}}
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action", "")

            if action == "subscribe":
                tickers = msg.get("tickers", [])
                manager.subscribe(websocket, tickers)
                await manager.send_personal(websocket, {
                    "type": "subscribed",
                    "tickers": list(manager.subscriptions.get(websocket, set())),
                })

            elif action == "unsubscribe":
                tickers = msg.get("tickers", [])
                manager.unsubscribe(websocket, tickers)
                await manager.send_personal(websocket, {
                    "type": "unsubscribed",
                    "tickers": tickers,
                })

            elif action == "ping":
                await manager.send_personal(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
