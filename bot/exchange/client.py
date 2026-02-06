"""Binance USDT-M Futures API client with algo order support.

Handles authentication, rate limiting, retries, and the /fapi/v1/algoOrder
endpoint for server-side conditional orders.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import deque
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from bot.core.config import APIConfig
from bot.utils.logger import get_logger

log = get_logger("exchange")


class RateLimiter:
    """Token-bucket rate limiter for Binance API."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: deque = deque()

    def acquire(self) -> None:
        now = time.time()
        while self._timestamps and now - self._timestamps[0] > self.window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_requests:
            sleep_time = self.window - (now - self._timestamps[0]) + 0.1
            log.warning("Rate limit approached — sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)
        self._timestamps.append(time.time())


class BinanceFuturesClient:
    """Low-level REST client for Binance USDT-M Futures."""

    def __init__(self, config: APIConfig):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": config.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self._limiter = RateLimiter(config.rate_limit_per_min, 60.0)
        self._order_limiter = RateLimiter(config.order_rate_limit_per_10s, 10.0)

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.cfg.recv_window
        query = urlencode(params)
        signature = hmac.new(
            self.cfg.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str,
                 params: Optional[Dict[str, Any]] = None,
                 signed: bool = True,
                 max_retries: int = 4) -> Dict[str, Any]:
        self._limiter.acquire()
        if params is None:
            params = {}
        if signed:
            params = self._sign(params)

        url = f"{self.cfg.base_url}{path}"
        last_err = None

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    resp = self.session.get(url, params=params, timeout=10)
                elif method == "POST":
                    resp = self.session.post(url, data=params, timeout=10)
                elif method == "DELETE":
                    resp = self.session.delete(url, params=params, timeout=10)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    log.warning("429 rate limited — backing off %ds", retry_after)
                    time.sleep(retry_after)
                    continue

                data = resp.json()
                if resp.status_code >= 400:
                    log.error("API error %d: %s", resp.status_code, data)
                    raise BinanceAPIError(resp.status_code, data)
                return data

            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                backoff = 2 ** (attempt + 1)
                log.warning("Network error (attempt %d/%d) — retrying in %ds: %s",
                            attempt + 1, max_retries, backoff, e)
                time.sleep(backoff)

        raise ConnectionError(f"Failed after {max_retries} retries: {last_err}")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v2/account")

    def get_balance(self) -> float:
        account = self.get_account()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["walletBalance"])
        return 0.0

    def get_positions(self) -> List[Dict[str, Any]]:
        account = self.get_account()
        return [
            p for p in account.get("positions", [])
            if float(p.get("positionAmt", 0)) != 0
        ]

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 500) -> List[List]:
        return self._request("GET", "/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit,
        }, signed=False)

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/ticker/price", {
            "symbol": symbol,
        }, signed=False)

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/exchangeInfo", params, signed=False)

    def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/premiumIndex", {
            "symbol": symbol,
        }, signed=False)

    # ------------------------------------------------------------------
    # Symbol info helpers
    # ------------------------------------------------------------------

    def get_symbol_precision(self, symbol: str) -> Dict[str, Any]:
        """Get tick size, step size, and minimum notional for a symbol."""
        info = self.get_exchange_info(symbol)
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                price_filter = filters.get("PRICE_FILTER", {})
                lot_filter = filters.get("LOT_SIZE", {})
                min_notional = filters.get("MIN_NOTIONAL", {})
                return {
                    "tick_size": float(price_filter.get("tickSize", 0.0001)),
                    "step_size": float(lot_filter.get("stepSize", 0.1)),
                    "min_qty": float(lot_filter.get("minQty", 0.1)),
                    "min_notional": float(min_notional.get("notional", 5.0)),
                    "price_precision": s.get("pricePrecision", 4),
                    "qty_precision": s.get("quantityPrecision", 1),
                }
        raise ValueError(f"Symbol {symbol} not found")

    # ------------------------------------------------------------------
    # Leverage & Margin
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        return self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol, "leverage": leverage,
        })

    def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        try:
            return self._request("POST", "/fapi/v1/marginType", {
                "symbol": symbol, "marginType": margin_type,
            })
        except BinanceAPIError as e:
            if e.code == -4046:  # Already set
                return {"msg": "No need to change margin type."}
            raise

    # ------------------------------------------------------------------
    # Standard orders
    # ------------------------------------------------------------------

    def place_market_order(self, symbol: str, side: str,
                           quantity: float) -> Dict[str, Any]:
        self._order_limiter.acquire()
        return self._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": self._format_qty(quantity, symbol),
        })

    def place_limit_order(self, symbol: str, side: str,
                          quantity: float, price: float,
                          time_in_force: str = "GTC") -> Dict[str, Any]:
        self._order_limiter.acquire()
        return self._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": self._format_qty(quantity, symbol),
            "price": self._format_price(price, symbol),
            "timeInForce": time_in_force,
        })

    def place_stop_market(self, symbol: str, side: str,
                          quantity: float, stop_price: float,
                          close_position: bool = False) -> Dict[str, Any]:
        self._order_limiter.acquire()
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": self._format_price(stop_price, symbol),
        }
        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = self._format_qty(quantity, symbol)
        return self._request("POST", "/fapi/v1/order", params)

    def place_take_profit_market(self, symbol: str, side: str,
                                 quantity: float, stop_price: float,
                                 close_position: bool = False) -> Dict[str, Any]:
        self._order_limiter.acquire()
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": self._format_price(stop_price, symbol),
        }
        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = self._format_qty(quantity, symbol)
        return self._request("POST", "/fapi/v1/order", params)

    # ------------------------------------------------------------------
    # Algo Orders (server-side conditional logic)
    # ------------------------------------------------------------------

    def place_algo_order(self, symbol: str, side: str, quantity: float,
                         algo_type: str = "VP",
                         urgency: str = "LOW",
                         extra_params: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
        """Place an algo order via POST /fapi/v1/algo/futures/newOrderVp.

        The algo order service handles server-side trigger logic,
        ensuring SL/TP execution even when the local bot is offline.
        """
        self._order_limiter.acquire()
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": self._format_qty(quantity, symbol),
            "urgency": urgency,
        }
        if extra_params:
            params.update(extra_params)
        return self._request("POST", "/fapi/v1/algo/futures/newOrderVp", params)

    def cancel_algo_order(self, algo_id: int) -> Dict[str, Any]:
        return self._request("DELETE", "/fapi/v1/algo/futures/order", {
            "algoId": algo_id,
        })

    def get_algo_open_orders(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v1/algo/futures/openOrders")

    def get_algo_historical_orders(self, symbol: Optional[str] = None,
                                   limit: int = 100) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/algo/futures/historicalOrders", params)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {
            "symbol": symbol,
        })

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openOrders", params)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    _precision_cache: Dict[str, Dict[str, Any]] = {}

    def _get_precision(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._precision_cache:
            self._precision_cache[symbol] = self.get_symbol_precision(symbol)
        return self._precision_cache[symbol]

    def _format_qty(self, qty: float, symbol: str) -> str:
        prec = self._get_precision(symbol)
        step = prec["step_size"]
        qty = max(qty, prec["min_qty"])
        qty = round(qty - (qty % step), prec["qty_precision"])
        return f"{qty:.{prec['qty_precision']}f}"

    def _format_price(self, price: float, symbol: str) -> str:
        prec = self._get_precision(symbol)
        tick = prec["tick_size"]
        price = round(price - (price % tick), prec["price_precision"])
        return f"{price:.{prec['price_precision']}f}"


class BinanceAPIError(Exception):
    """Structured Binance API error."""

    def __init__(self, status_code: int, data: Dict[str, Any]):
        self.status_code = status_code
        self.code = data.get("code", 0)
        self.msg = data.get("msg", str(data))
        super().__init__(f"Binance {status_code}: [{self.code}] {self.msg}")
