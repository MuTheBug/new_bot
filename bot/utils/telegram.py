"""Telegram notification service for the trading bot.

Sends alerts for position opens/closes, errors, and daily summaries.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import requests

from bot.utils.logger import get_logger

log = get_logger("telegram")

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Non-blocking Telegram message sender.

    Messages are sent in a background thread so they never block
    the trading loop. Failed sends are logged but silently dropped
    to avoid disrupting bot operation.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id)

        if not self._enabled:
            log.warning(
                "Telegram notifications disabled — "
                "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _send_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message synchronously. Returns True on success."""
        if not self._enabled:
            return False
        url = _TELEGRAM_API.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    return True
                if resp.status_code == 429:
                    retry_after = resp.json().get("parameters", {}).get(
                        "retry_after", 5
                    )
                    log.warning("Telegram rate limited, retry in %ds", retry_after)
                    time.sleep(retry_after)
                    continue
                log.warning(
                    "Telegram send failed (HTTP %d): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            except requests.RequestException as e:
                log.warning("Telegram send error (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return False

    def send(self, text: str) -> None:
        """Send a message in a background thread (non-blocking)."""
        if not self._enabled:
            return
        thread = threading.Thread(
            target=self._send_sync, args=(text,), daemon=True
        )
        thread.start()

    # ------------------------------------------------------------------
    # Pre-built notification templates
    # ------------------------------------------------------------------

    def notify_position_opened(
        self,
        side: str,
        symbol: str,
        qty: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        signal: float,
        equity: float,
    ) -> None:
        """Send alert when a new position is opened."""
        emoji = "\U0001f7e2" if side == "LONG" else "\U0001f534"
        notional = qty * entry_price
        text = (
            f"{emoji} <b>Position Opened</b>\n"
            f"\n"
            f"<b>Side:</b>   {side}\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Qty:</b>    {qty:.4f}\n"
            f"<b>Entry:</b>  ${entry_price:.4f}\n"
            f"<b>Notional:</b> ${notional:.2f}\n"
            f"<b>SL:</b>     ${sl_price:.4f}\n"
            f"<b>TP:</b>     ${tp_price:.4f}\n"
            f"<b>Signal:</b> {signal:.3f}\n"
            f"\n"
            f"\U0001f4b0 Equity: ${equity:.2f}"
        )
        self.send(text)

    def notify_position_closed(
        self,
        side: str,
        symbol: str,
        reason: str,
        equity: float,
    ) -> None:
        """Send alert when a position is closed."""
        reason_emoji = {
            "signal_reversal": "\U0001f504",
            "stop_loss": "\U0001f6d1",
            "take_profit": "\U0001f3af",
            "trailing_stop": "\U0001f4c8",
            "kill_switch": "\u26a0\ufe0f",
        }
        emoji = reason_emoji.get(reason, "\U0001f4e4")
        text = (
            f"{emoji} <b>Position Closed</b>\n"
            f"\n"
            f"<b>Side:</b>   {side}\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Reason:</b> {reason}\n"
            f"\n"
            f"\U0001f4b0 Equity: ${equity:.2f}"
        )
        self.send(text)

    def notify_error(self, error: str, context: str = "") -> None:
        """Send alert when an error occurs."""
        text = (
            f"\u26a0\ufe0f <b>Bot Error</b>\n"
            f"\n"
            f"<b>Context:</b> {context}\n"
            f"<code>{error[:500]}</code>"
        )
        self.send(text)

    def notify_kill_switch(self, drawdown_pct: float, equity: float) -> None:
        """Send alert when the kill-switch is triggered."""
        text = (
            f"\U0001f6a8 <b>KILL-SWITCH ACTIVATED</b>\n"
            f"\n"
            f"Drawdown: {drawdown_pct:.1f}%\n"
            f"Equity: ${equity:.2f}\n"
            f"\n"
            f"Trading halted. Manual review required."
        )
        self.send(text)

    def notify_startup(
        self,
        symbol: str,
        leverage: int,
        equity: float,
    ) -> None:
        """Send alert when the bot starts."""
        text = (
            f"\U0001f680 <b>Bot Started</b>\n"
            f"\n"
            f"<b>Symbol:</b>   {symbol}\n"
            f"<b>Leverage:</b> {leverage}x\n"
            f"<b>Equity:</b>   ${equity:.2f}"
        )
        self.send(text)

    def notify_shutdown(self, equity: float, total_trades: int) -> None:
        """Send alert when the bot stops."""
        text = (
            f"\U0001f6d1 <b>Bot Stopped</b>\n"
            f"\n"
            f"<b>Final Equity:</b> ${equity:.2f}\n"
            f"<b>Total Trades:</b> {total_trades}"
        )
        self.send(text)

    def notify_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        trades_today: int,
        drawdown_pct: float,
    ) -> None:
        """Send daily summary notification."""
        pnl_emoji = "\U0001f4c8" if daily_pnl >= 0 else "\U0001f4c9"
        text = (
            f"\U0001f4ca <b>Daily Summary</b>\n"
            f"\n"
            f"<b>Equity:</b>   ${equity:.2f}\n"
            f"{pnl_emoji} <b>Daily PnL:</b> ${daily_pnl:+.2f}\n"
            f"<b>Trades:</b>   {trades_today}\n"
            f"<b>Drawdown:</b> {drawdown_pct:.1f}%"
        )
        self.send(text)
