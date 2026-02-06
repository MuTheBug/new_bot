"""
Binance Futures API Client.
Handles authentication, order placement (regular + algo), position management,
and market data retrieval.
"""
import hashlib
import hmac
import logging
import time
from urllib.parse import urlencode

import requests

import config

logger = logging.getLogger(__name__)


class BinanceClient:
    """Wrapper around Binance USDT-M Futures REST API."""

    def __init__(self, api_key=None, api_secret=None, testnet=None):
        self.api_key = api_key or config.API_KEY
        self.api_secret = api_secret or config.API_SECRET
        self.testnet = testnet if testnet is not None else config.TESTNET
        self.base_url = config.BASE_URL_TESTNET if self.testnet else config.BASE_URL_LIVE
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self._recv_window = 5000

    def _sign(self, params: dict) -> dict:
        """Add timestamp and HMAC-SHA256 signature to request params."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method, path, params=None, signed=False, retries=4):
        """Execute an API request with optional signing and retry logic."""
        if params is None:
            params = {}
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{path}"
        last_exc = None

        for attempt in range(retries):
            try:
                if method == "GET":
                    resp = self.session.get(url, params=params, timeout=10)
                elif method == "POST":
                    resp = self.session.post(url, data=params, timeout=10)
                elif method == "DELETE":
                    resp = self.session.delete(url, params=params, timeout=10)
                elif method == "PUT":
                    resp = self.session.put(url, data=params, timeout=10)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. Retrying in %ds",
                        attempt + 1, retries, e, wait
                    )
                    time.sleep(wait)

        logger.error("Request failed after %d attempts: %s", retries, last_exc)
        raise last_exc

    # ── Market Data ──────────────────────────────────────────────────────

    def get_klines(self, symbol, interval="1h", limit=200):
        """Fetch candlestick/kline data."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return self._request("GET", "/fapi/v1/klines", params)

    def get_mark_price(self, symbol):
        """Get current mark price for a symbol."""
        params = {"symbol": symbol}
        data = self._request("GET", "/fapi/v1/premiumIndex", params)
        return float(data["markPrice"])

    def get_ticker(self, symbol):
        """Get 24hr ticker for a symbol."""
        params = {"symbol": symbol}
        return self._request("GET", "/fapi/v1/ticker/24hr", params)

    def get_exchange_info(self, symbol=None):
        """Get exchange trading rules and symbol information."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/exchangeInfo", params)

    # ── Account ──────────────────────────────────────────────────────────

    def get_account(self):
        """Get current account information."""
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_balance(self):
        """Get USDT available balance."""
        account = self.get_account()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return {
                    "total": float(asset["walletBalance"]),
                    "available": float(asset["availableBalance"]),
                    "unrealized_pnl": float(asset["unrealizedProfit"]),
                }
        return {"total": 0.0, "available": 0.0, "unrealized_pnl": 0.0}

    def get_positions(self, symbol=None):
        """Get open position(s)."""
        account = self.get_account()
        positions = []
        for pos in account.get("positions", []):
            amt = float(pos["positionAmt"])
            if amt == 0:
                continue
            if symbol and pos["symbol"] != symbol:
                continue
            positions.append({
                "symbol": pos["symbol"],
                "side": "LONG" if amt > 0 else "SHORT",
                "size": abs(amt),
                "entry_price": float(pos["entryPrice"]),
                "unrealized_pnl": float(pos["unrealizedProfit"]),
                "leverage": int(pos["leverage"]),
                "margin_type": pos["marginType"],
                "notional": abs(amt) * float(pos["entryPrice"]),
            })
        return positions

    # ── Leverage & Margin ────────────────────────────────────────────────

    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol."""
        params = {"symbol": symbol, "leverage": int(leverage)}
        try:
            result = self._request("POST", "/fapi/v1/leverage", params, signed=True)
            logger.info("Set leverage for %s to %dx", symbol, leverage)
            return result
        except Exception as e:
            logger.error("Failed to set leverage for %s: %s", symbol, e)
            raise

    def set_margin_type(self, symbol, margin_type="ISOLATED"):
        """Set margin type (ISOLATED or CROSSED)."""
        params = {"symbol": symbol, "marginType": margin_type}
        try:
            result = self._request(
                "POST", "/fapi/v1/marginType", params, signed=True
            )
            logger.info("Set margin type for %s to %s", symbol, margin_type)
            return result
        except Exception as e:
            if "No need to change margin type" in str(e):
                logger.debug("Margin type already %s for %s", margin_type, symbol)
                return None
            logger.error("Failed to set margin type for %s: %s", symbol, e)
            raise

    def set_position_mode(self, dual_side=False):
        """Set position mode (One-way or Hedge)."""
        params = {"dualSidePosition": str(dual_side).lower()}
        try:
            return self._request(
                "POST", "/fapi/v1/positionSide/dual", params, signed=True
            )
        except Exception as e:
            if "No need to change position side" in str(e):
                return None
            raise

    # ── Orders ───────────────────────────────────────────────────────────

    def place_order(
        self, symbol, side, order_type, quantity,
        price=None, stop_price=None, time_in_force=None,
        reduce_only=False, close_position=False,
    ):
        """Place a regular futures order."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if time_in_force:
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = "true"
        if close_position:
            params["closePosition"] = "true"

        result = self._request("POST", "/fapi/v1/order", params, signed=True)
        logger.info(
            "Order placed: %s %s %s qty=%s type=%s",
            symbol, side, order_type, quantity, order_type,
        )
        return result

    def place_market_order(self, symbol, side, quantity):
        """Place a market order."""
        return self.place_order(symbol, side, "MARKET", quantity)

    def place_limit_order(self, symbol, side, quantity, price):
        """Place a limit order."""
        return self.place_order(
            symbol, side, "LIMIT", quantity, price=price, time_in_force="GTC"
        )

    def place_stop_market(self, symbol, side, quantity, stop_price, reduce_only=True):
        """Place a stop market order (for stop-losses)."""
        return self.place_order(
            symbol, side, "STOP_MARKET", quantity,
            stop_price=stop_price, reduce_only=reduce_only,
        )

    def place_take_profit_market(
        self, symbol, side, quantity, stop_price, reduce_only=True
    ):
        """Place a take-profit market order."""
        return self.place_order(
            symbol, side, "TAKE_PROFIT_MARKET", quantity,
            stop_price=stop_price, reduce_only=reduce_only,
        )

    def cancel_order(self, symbol, order_id):
        """Cancel an open order."""
        params = {"symbol": symbol, "orderId": order_id}
        return self._request("DELETE", "/fapi/v1/order", params, signed=True)

    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        params = {"symbol": symbol}
        return self._request(
            "DELETE", "/fapi/v1/allOpenOrders", params, signed=True
        )

    def get_open_orders(self, symbol=None):
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    # ── Algo Orders (TWAP / VP) ──────────────────────────────────────────

    def place_twap_order(
        self, symbol, side, quantity, duration=300, limit_price=None
    ):
        """
        Place a TWAP algo order.
        Only for positions with notional >= 1000 USDT.
        Duration in seconds (min 300 = 5 min, max 86400 = 24 hrs).
        """
        params = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "duration": duration,
        }
        if limit_price is not None:
            params["limitPrice"] = limit_price

        # TWAP uses sapi base URL (main site, not futures)
        sapi_base = (
            "https://testnet.binance.vision"
            if self.testnet
            else "https://api.binance.com"
        )
        old_base = self.base_url
        self.base_url = sapi_base

        try:
            result = self._request(
                "POST", config.ALGO_TWAP_ENDPOINT, params, signed=True
            )
            logger.info(
                "TWAP order placed: %s %s qty=%s duration=%ds",
                symbol, side, quantity, duration,
            )
            return result
        finally:
            self.base_url = old_base

    def place_vp_order(
        self, symbol, side, quantity, urgency="LOW", limit_price=None
    ):
        """
        Place a Volume Participation (VP) algo order.
        urgency: LOW, MEDIUM, HIGH
        """
        params = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "urgency": urgency,
        }
        if limit_price is not None:
            params["limitPrice"] = limit_price

        sapi_base = (
            "https://testnet.binance.vision"
            if self.testnet
            else "https://api.binance.com"
        )
        old_base = self.base_url
        self.base_url = sapi_base

        try:
            result = self._request(
                "POST", config.ALGO_VP_ENDPOINT, params, signed=True
            )
            logger.info("VP order placed: %s %s qty=%s", symbol, side, quantity)
            return result
        finally:
            self.base_url = old_base

    def get_algo_open_orders(self):
        """Query open algo orders."""
        sapi_base = (
            "https://testnet.binance.vision"
            if self.testnet
            else "https://api.binance.com"
        )
        old_base = self.base_url
        self.base_url = sapi_base
        try:
            return self._request(
                "GET", config.ALGO_OPEN_ORDERS, signed=True
            )
        finally:
            self.base_url = old_base

    # ── Utility ──────────────────────────────────────────────────────────

    def get_server_time(self):
        """Get Binance server time."""
        data = self._request("GET", "/fapi/v1/time")
        return data["serverTime"]

    def ping(self):
        """Test API connectivity."""
        try:
            self._request("GET", "/fapi/v1/ping")
            return True
        except Exception:
            return False
