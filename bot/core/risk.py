"""Risk management: stops, circuit breakers, flash-crash protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from bot.core.config import RiskConfig
from bot.utils.logger import get_logger

log = get_logger("risk")


@dataclass
class TradeRecord:
    """Single trade result for equity-curve tracking."""

    pnl: float
    equity_after: float
    timestamp: str = ""


class RiskManager:
    """Centralised risk gate that must approve every trade action."""

    def __init__(self, config: RiskConfig, initial_equity: float = 10.0):
        self.cfg = config
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.consecutive_losses = 0
        self.trade_history: List[TradeRecord] = []
        self._cooldown_remaining = 0
        self._killed = False

    # ------------------------------------------------------------------
    # ATR-based stops
    # ------------------------------------------------------------------

    def compute_stop_loss(self, entry_price: float, atr: float,
                          side: str) -> float:
        """Dynamic SL based on ATR."""
        distance = atr * self.cfg.sl_atr_mult
        if side == "LONG":
            return entry_price - distance
        return entry_price + distance

    def compute_take_profit(self, entry_price: float, atr: float,
                            side: str) -> float:
        """Dynamic TP based on ATR."""
        distance = atr * self.cfg.tp_atr_mult
        if side == "LONG":
            return entry_price + distance
        return entry_price - distance

    def compute_trailing_stop(self, entry_price: float, current_price: float,
                              side: str) -> Optional[float]:
        """Activate trailing stop after price moves in favour."""
        if side == "LONG":
            pct_move = (current_price / entry_price - 1) * 100
            if pct_move >= self.cfg.trailing_activate_pct:
                return current_price * (1 - self.cfg.trailing_callback_pct / 100)
        else:
            pct_move = (1 - current_price / entry_price) * 100
            if pct_move >= self.cfg.trailing_activate_pct:
                return current_price * (1 + self.cfg.trailing_callback_pct / 100)
        return None

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def max_position_size(self, entry_price: float, atr: float,
                          leverage: int = 10) -> float:
        """Position size in base asset, risking `risk_per_trade_pct` of equity."""
        risk_amount = self.current_equity * 0.02  # 2% risk
        stop_distance = atr * self.cfg.sl_atr_mult
        if stop_distance <= 0:
            return 0.0
        # Size = risk_amount / (stop_distance / entry_price) / entry_price
        size = (risk_amount / stop_distance) * leverage
        return size

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def _check_drawdown(self) -> bool:
        if self.peak_equity <= 0:
            return True
        dd_pct = (1 - self.current_equity / self.peak_equity) * 100
        if dd_pct >= self.cfg.max_drawdown_pct:
            log.critical(
                "KILL-SWITCH: drawdown %.1f%% >= threshold %.1f%% — halting trading",
                dd_pct, self.cfg.max_drawdown_pct,
            )
            self._killed = True
            return True
        return False

    def _check_daily_loss(self) -> bool:
        if self.daily_start_equity <= 0:
            return True
        daily_loss_pct = (1 - self.current_equity / self.daily_start_equity) * 100
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            log.warning("Daily loss limit hit: %.1f%%", daily_loss_pct)
            return True
        return False

    def _check_consecutive_losses(self) -> bool:
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            log.warning(
                "Consecutive losses %d >= %d — pausing",
                self.consecutive_losses, self.cfg.max_consecutive_losses,
            )
            return True
        return False

    def detect_flash_crash(self, open_price: float, close_price: float,
                           high: float, low: float) -> bool:
        """Detect abnormal single-candle price movement."""
        if open_price <= 0:
            return False
        move_pct = abs(close_price - open_price) / open_price * 100
        range_pct = (high - low) / open_price * 100
        if move_pct >= self.cfg.flash_crash_pct or range_pct >= self.cfg.flash_crash_pct * 1.5:
            log.warning("Flash crash detected: move=%.2f%%, range=%.2f%%", move_pct, range_pct)
            self._cooldown_remaining = self.cfg.cooldown_candles
            return True
        return False

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def can_trade(self) -> bool:
        """Master gate — returns True only if all risk checks pass."""
        if self._killed:
            return False
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            log.info("Cooldown active: %d candles remaining", self._cooldown_remaining + 1)
            return False
        if self._check_drawdown():
            return False
        if self._check_daily_loss():
            return False
        if self._check_consecutive_losses():
            return False
        return True

    # ------------------------------------------------------------------
    # Trade tracking
    # ------------------------------------------------------------------

    def record_trade(self, pnl: float, timestamp: str = "") -> None:
        """Update equity curve and loss streaks after a trade closes."""
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.trade_history.append(
            TradeRecord(pnl=pnl, equity_after=self.current_equity, timestamp=timestamp)
        )

    def reset_daily(self) -> None:
        """Call at the start of each new UTC day."""
        self.daily_start_equity = self.current_equity

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (1 - self.current_equity / self.peak_equity) * 100

    def summary(self) -> dict:
        pnls = [t.pnl for t in self.trade_history]
        if not pnls:
            return {"trades": 0}
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        returns = np.array(pnls)
        sharpe = (
            (returns.mean() / returns.std() * np.sqrt(365 * 24))
            if returns.std() > 0 else 0.0
        )
        return {
            "trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(pnls) * 100,
            "profit_factor": gross_profit / gross_loss,
            "sharpe_ratio": sharpe,
            "total_pnl": sum(pnls),
            "max_drawdown_pct": self.drawdown_pct(),
            "final_equity": self.current_equity,
        }
