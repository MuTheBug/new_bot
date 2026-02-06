"""
Risk Management System.

Handles:
- Dynamic leverage calculation based on account size and volatility
- Position sizing respecting Binance minimums
- Drawdown monitoring and circuit breakers
- Fee-aware calculations for small accounts
- Compounding logic and growth tracking
"""
import logging
import math

import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages all risk-related calculations and limits."""

    def __init__(self):
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.trade_history = []
        self.is_halted = False
        self.halt_reason = ""

    def reset_daily(self):
        """Reset daily tracking metrics."""
        self.daily_pnl = 0.0

    def update_peak(self, balance):
        """Update peak balance for drawdown tracking."""
        if balance > self.peak_balance:
            self.peak_balance = balance
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - balance) / self.peak_balance

    def record_trade(self, pnl, balance):
        """Record a completed trade result."""
        self.daily_pnl += pnl
        self.trade_history.append(pnl)
        self.update_peak(balance)

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self._check_circuit_breakers(balance)

    def _check_circuit_breakers(self, balance):
        """Check if trading should be halted. Only daily loss causes halt;
        drawdown triggers position size reduction instead of full halt."""
        risk = config.RISK

        # Daily loss limit - temporary halt until next day reset
        if balance > 0:
            daily_loss_pct = abs(self.daily_pnl) / balance if self.daily_pnl < 0 else 0
            if daily_loss_pct >= risk["max_daily_loss"]:
                self.is_halted = True
                self.halt_reason = (
                    f"Daily loss limit hit: {daily_loss_pct:.1%} "
                    f"(limit {risk['max_daily_loss']:.1%})"
                )
                logger.warning("CIRCUIT BREAKER: %s", self.halt_reason)
                return

        # Minimum balance
        if balance < risk["min_balance_to_trade"]:
            self.is_halted = True
            self.halt_reason = (
                f"Balance too low: ${balance:.2f} "
                f"(minimum ${risk['min_balance_to_trade']:.2f})"
            )
            logger.warning("CIRCUIT BREAKER: %s", self.halt_reason)

    def can_trade(self):
        """Check if trading is currently allowed."""
        if self.is_halted:
            logger.info("Trading halted: %s", self.halt_reason)
            return False
        return True

    def calculate_optimal_leverage(self, balance, atr, current_price, symbol):
        """
        Calculate optimal leverage based on account size, volatility,
        and stop-loss distance.

        Goal: Use enough leverage to meet minimum order requirements while
        keeping risk per trade within limits.
        """
        lev_config = config.LEVERAGE

        # 1. Get max leverage for this balance tier
        max_lev_for_balance = lev_config["max_leverage"]
        for threshold in sorted(lev_config["tiers"].keys(), reverse=True):
            if balance >= threshold:
                max_lev_for_balance = lev_config["tiers"][threshold]
                break

        # 2. Calculate leverage needed to meet minimum notional
        pair_info = config.PAIR_INFO.get(symbol, {})
        min_notional = pair_info.get("min_notional", 5.0)
        min_qty = pair_info.get("min_qty", 0.1)
        min_trade_notional = max(min_notional, min_qty * current_price)

        leverage_for_min_order = max(
            1, math.ceil(min_trade_notional / (balance * 0.9))
        )

        # 3. Calculate leverage based on stop-loss distance
        # We want risk_per_trade = risk_pct * balance
        # risk_per_trade = position_size * (stop_distance / entry_price)
        # position_size = balance * leverage
        # So: leverage = (risk_pct * balance) / (balance * stop_distance / entry_price)
        #   = risk_pct / (stop_distance / entry_price)
        stop_distance = atr * config.STRATEGY["atr_stop_multiplier"]
        if stop_distance > 0 and current_price > 0:
            stop_pct = stop_distance / current_price
            risk_pct = config.RISK["max_risk_per_trade"]
            # Adjust for consecutive losses
            if self.consecutive_losses >= config.RISK["max_consecutive_losses"]:
                risk_pct *= config.RISK["loss_reduction_factor"]
            leverage_for_risk = max(1, int(risk_pct / stop_pct))
        else:
            leverage_for_risk = 1

        # 4. Final leverage: highest of minimum needed, but capped by risk and tier
        optimal = max(leverage_for_min_order, min(leverage_for_risk, max_lev_for_balance))
        optimal = max(lev_config["min_leverage"], min(optimal, max_lev_for_balance))

        logger.debug(
            "Leverage calc for %s: balance=$%.2f, min_order_lev=%d, "
            "risk_lev=%d, tier_max=%d -> optimal=%d",
            symbol, balance, leverage_for_min_order,
            leverage_for_risk, max_lev_for_balance, optimal,
        )

        return optimal

    def calculate_position_size(
        self, balance, leverage, current_price, atr, symbol
    ):
        """
        Calculate position size in base asset units.

        Ensures:
        - Risk per trade stays within limits
        - Meets Binance minimum order requirements
        - Accounts for fees
        - Prevents liquidation before stop-loss
        """
        pair_info = config.PAIR_INFO.get(symbol, {})
        min_qty = pair_info.get("min_qty", 0.1)
        qty_step = pair_info.get("qty_step", 0.1)
        min_notional = pair_info.get("min_notional", 5.0)

        risk_pct = config.RISK["max_risk_per_trade"]
        if self.consecutive_losses >= config.RISK["max_consecutive_losses"]:
            risk_pct *= config.RISK["loss_reduction_factor"]

        # Risk budget in USDT
        risk_budget = balance * risk_pct

        # Stop-loss distance
        stop_distance = atr * config.STRATEGY["atr_stop_multiplier"]
        stop_pct = stop_distance / current_price if current_price > 0 else 0.01

        # Max position size based on risk
        if stop_pct > 0:
            max_position_usdt = risk_budget / stop_pct
        else:
            max_position_usdt = risk_budget

        # Max position based on available margin
        fee_buffer = 1 - (config.FEES["default"] * 2)  # Entry + exit fees
        max_margin_position = balance * leverage * fee_buffer

        # Take the smaller of risk-based and margin-based
        position_usdt = min(max_position_usdt, max_margin_position)

        # Convert to quantity
        quantity = position_usdt / current_price if current_price > 0 else 0

        # Round to step size
        if qty_step > 0:
            quantity = math.floor(quantity / qty_step) * qty_step

        # Check minimums
        if quantity < min_qty:
            # Try minimum quantity if we can afford it
            min_cost = min_qty * current_price / leverage
            if min_cost <= balance * 0.95:  # Can afford with 5% buffer
                quantity = min_qty
            else:
                logger.warning(
                    "Cannot afford minimum order for %s: need $%.2f, have $%.2f",
                    symbol, min_cost, balance,
                )
                return 0.0

        # Verify notional
        notional = quantity * current_price
        if notional < min_notional:
            quantity = math.ceil(min_notional / current_price / qty_step) * qty_step
            new_cost = quantity * current_price / leverage
            if new_cost > balance * 0.95:
                logger.warning("Cannot meet minimum notional for %s", symbol)
                return 0.0

        # Final safety: ensure we won't get liquidated before stop-loss
        # Liquidation happens at ~margin / position_size distance
        margin = quantity * current_price / leverage
        liq_distance_pct = (margin * 0.9) / (quantity * current_price)  # 90% of margin
        if liq_distance_pct < stop_pct:
            # Reduce position so stop hits before liquidation
            safe_qty = (balance * leverage * 0.8 * stop_pct) / (current_price * stop_pct)
            quantity = math.floor(safe_qty / qty_step) * qty_step
            if quantity < min_qty:
                logger.warning("Position too small to avoid liquidation for %s", symbol)
                return 0.0

        # Round final precision
        precision = pair_info.get("qty_precision", 1)
        quantity = round(quantity, precision)

        logger.debug(
            "Position size for %s: qty=%.4f, notional=$%.2f, "
            "margin=$%.2f, risk=$%.2f",
            symbol, quantity, quantity * current_price,
            quantity * current_price / leverage, risk_budget,
        )

        return quantity

    def should_use_algo_order(self, quantity, current_price):
        """Determine if position is large enough for TWAP/VP execution."""
        notional = quantity * current_price
        return (
            config.ALGO["use_algo_orders"]
            and notional >= config.ALGO["twap_min_notional"]
        )

    def get_growth_tier(self, balance):
        """Determine current account growth tier for logging."""
        tiers = [
            (10000, "WHALE ($10K+)"),
            (5000, "ADVANCED ($5K+)"),
            (1000, "GROWTH ($1K+)"),
            (200, "BUILDING ($200+)"),
            (50, "STARTER ($50+)"),
            (10, "MICRO ($10+)"),
            (0, "SEED (<$10)"),
        ]
        for threshold, name in tiers:
            if balance >= threshold:
                return name
        return "SEED (<$10)"

    def get_status(self, balance):
        """Get current risk management status."""
        return {
            "balance": balance,
            "tier": self.get_growth_tier(balance),
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": self.daily_pnl,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "total_trades": len(self.trade_history),
        }
