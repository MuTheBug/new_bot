"""Precision-based position sizing optimised for low-balance accounts.

Handles Binance's strict minimum notional, step sizes, and tick sizes
while maximising capital utilisation on a $10 account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from bot.utils.logger import get_logger

log = get_logger("sizer")


@dataclass
class SymbolSpec:
    """Exchange-provided symbol constraints."""

    tick_size: float = 0.0001
    step_size: float = 0.1
    min_qty: float = 0.1
    min_notional: float = 5.0
    price_precision: int = 4
    qty_precision: int = 1


class PositionSizer:
    """Precision position sizing for micro-balance accounts."""

    def __init__(self, spec: SymbolSpec, leverage: int = 10,
                 risk_pct: float = 2.0, max_position_pct: float = 90.0):
        self.spec = spec
        self.leverage = leverage
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct

    def _floor_to_step(self, qty: float) -> float:
        """Round quantity down to nearest step size."""
        if self.spec.step_size <= 0:
            return qty
        steps = math.floor(qty / self.spec.step_size)
        return round(steps * self.spec.step_size, self.spec.qty_precision)

    def _round_price(self, price: float) -> float:
        """Round price to nearest tick size."""
        if self.spec.tick_size <= 0:
            return price
        ticks = round(price / self.spec.tick_size)
        return round(ticks * self.spec.tick_size, self.spec.price_precision)

    def compute_quantity(self, equity: float, entry_price: float,
                         stop_distance: float) -> Optional[float]:
        """Compute the exact trade quantity respecting all Binance constraints.

        Args:
            equity: Current USDT balance.
            entry_price: Expected entry price.
            stop_distance: Absolute price distance to stop-loss.

        Returns:
            Quantity in base asset, or None if trade is infeasible.
        """
        if entry_price <= 0 or stop_distance <= 0 or equity <= 0:
            return None

        # 1. Risk-based sizing: risk_amount / stop_distance
        risk_amount = equity * self.risk_pct / 100
        qty_by_risk = risk_amount / stop_distance

        # 2. Max capital allocation
        max_notional = equity * self.max_position_pct / 100 * self.leverage
        qty_by_capital = max_notional / entry_price

        # 3. Take the smaller
        qty = min(qty_by_risk, qty_by_capital)

        # 4. Floor to step size
        qty = self._floor_to_step(qty)

        # 5. Check minimum quantity
        if qty < self.spec.min_qty:
            # Try using min_qty if it satisfies notional
            if self.spec.min_qty * entry_price >= self.spec.min_notional:
                qty = self.spec.min_qty
            else:
                log.debug(
                    "Position too small: qty=%.4f < min_qty=%.4f",
                    qty, self.spec.min_qty,
                )
                return None

        # 6. Check minimum notional
        notional = qty * entry_price
        if notional < self.spec.min_notional:
            # Try bumping qty to meet notional
            min_qty_for_notional = math.ceil(
                self.spec.min_notional / entry_price / self.spec.step_size
            ) * self.spec.step_size
            min_qty_for_notional = round(min_qty_for_notional, self.spec.qty_precision)
            required_margin = min_qty_for_notional * entry_price / self.leverage
            if required_margin <= equity * self.max_position_pct / 100:
                qty = min_qty_for_notional
            else:
                log.debug(
                    "Cannot meet min notional $%.2f with equity $%.2f",
                    self.spec.min_notional, equity,
                )
                return None

        # 7. Final margin check
        required_margin = qty * entry_price / self.leverage
        if required_margin > equity * self.max_position_pct / 100:
            # Scale down
            max_margin = equity * self.max_position_pct / 100
            qty = max_margin * self.leverage / entry_price
            qty = self._floor_to_step(qty)
            if qty < self.spec.min_qty or qty * entry_price < self.spec.min_notional:
                return None

        return qty

    def compute_stop_loss_price(self, entry_price: float,
                                atr: float, multiplier: float,
                                side: str) -> float:
        """ATR-based stop-loss price rounded to tick size."""
        distance = atr * multiplier
        if side == "LONG":
            return self._round_price(entry_price - distance)
        return self._round_price(entry_price + distance)

    def compute_take_profit_price(self, entry_price: float,
                                  atr: float, multiplier: float,
                                  side: str) -> float:
        """ATR-based take-profit price rounded to tick size."""
        distance = atr * multiplier
        if side == "LONG":
            return self._round_price(entry_price + distance)
        return self._round_price(entry_price - distance)

    def format_quantity(self, qty: float) -> str:
        return f"{qty:.{self.spec.qty_precision}f}"

    def format_price(self, price: float) -> str:
        return f"{price:.{self.spec.price_precision}f}"

    def feasibility_report(self, equity: float, price: float) -> Dict[str, object]:
        """Check if trading is feasible at all for the current equity."""
        min_margin = self.spec.min_notional / self.leverage
        can_trade = equity >= min_margin * 1.1  # 10% buffer
        return {
            "equity": equity,
            "min_notional": self.spec.min_notional,
            "leverage": self.leverage,
            "min_margin_required": min_margin,
            "can_trade": can_trade,
            "max_qty_at_price": self._floor_to_step(
                equity * self.max_position_pct / 100 * self.leverage / price
            ) if price > 0 else 0,
        }
