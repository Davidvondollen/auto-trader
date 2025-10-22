"""
Live Trading Safety System
Additional safety checks and circuit breakers for live trading.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np


class SafetyMonitor:
    """
    Monitor for live trading safety.
    Implements circuit breakers and risk limits.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_position_loss_pct: float = 0.10,
        max_trades_per_day: int = 100,
        max_order_size_pct: float = 0.20,
        require_confirmation: bool = True
    ):
        """
        Initialize safety monitor.

        Args:
            max_daily_loss_pct: Maximum daily loss percentage before halt
            max_position_loss_pct: Maximum loss per position before force close
            max_trades_per_day: Maximum trades per day
            max_order_size_pct: Maximum order size as % of portfolio
            require_confirmation: Require manual confirmation for live trades
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_loss_pct = max_position_loss_pct
        self.max_trades_per_day = max_trades_per_day
        self.max_order_size_pct = max_order_size_pct
        self.require_confirmation = require_confirmation

        self.daily_trades = []
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        self.start_of_day_value = None
        self.last_reset = datetime.now().date()

        logger.info("SafetyMonitor initialized with circuit breakers")

    def check_order_safety(
        self,
        order: Dict,
        portfolio_value: float,
        positions: Dict,
        account_info: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an order is safe to execute.

        Args:
            order: Order details
            portfolio_value: Current portfolio value
            positions: Current positions
            account_info: Account information

        Returns:
            (is_safe, reason_if_not)
        """
        # Check if circuit breaker is active
        if self.circuit_breaker_active:
            return False, f"Circuit breaker active: {self.circuit_breaker_reason}"

        # Reset daily counters if new day
        self._check_day_reset(portfolio_value)

        # Check daily trade limit
        if len(self.daily_trades) >= self.max_trades_per_day:
            self._activate_circuit_breaker("Daily trade limit exceeded")
            return False, f"Daily trade limit ({self.max_trades_per_day}) exceeded"

        # Check order size
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        action = order.get('action')

        if action == 'buy':
            # Check buying power
            buying_power = account_info.get('buying_power', 0)

            if buying_power <= 0:
                return False, "Insufficient buying power"

            # Estimate order value
            estimated_price = order.get('limit_price') or order.get('estimated_price', 0)
            order_value = quantity * estimated_price

            # Check order size vs portfolio
            order_pct = order_value / portfolio_value if portfolio_value > 0 else 1

            if order_pct > self.max_order_size_pct:
                return False, f"Order size ({order_pct:.1%}) exceeds limit ({self.max_order_size_pct:.1%})"

        elif action == 'sell':
            # Check if we have the position
            if symbol not in positions:
                return False, f"No position in {symbol} to sell"

            position_qty = positions[symbol].get('qty', 0)

            if quantity > position_qty:
                return False, f"Insufficient shares (have {position_qty}, trying to sell {quantity})"

        # Check daily loss limit
        if self.start_of_day_value:
            daily_loss = portfolio_value - self.start_of_day_value
            daily_loss_pct = daily_loss / self.start_of_day_value

            if daily_loss_pct < -self.max_daily_loss_pct:
                self._activate_circuit_breaker(
                    f"Daily loss limit exceeded: {daily_loss_pct:.2%}"
                )
                return False, "Daily loss limit exceeded - trading halted"

        # All checks passed
        return True, None

    def record_trade(self, order: Dict) -> None:
        """
        Record a trade for monitoring.

        Args:
            order: Executed order
        """
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': order.get('symbol'),
            'action': order.get('action'),
            'quantity': order.get('quantity'),
            'price': order.get('price'),
            'order_id': order.get('order_id')
        }

        self.daily_trades.append(trade_record)
        logger.info(f"Recorded trade {len(self.daily_trades)}/{self.max_trades_per_day}")

    def check_position_safety(
        self,
        positions: Dict,
        current_prices: Dict
    ) -> List[str]:
        """
        Check positions for safety violations.

        Args:
            positions: Current positions
            current_prices: Current prices

        Returns:
            List of symbols that should be closed
        """
        symbols_to_close = []

        for symbol, position in positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = position.get('avg_entry_price', current_price)

            # Calculate loss
            loss_pct = (current_price - entry_price) / entry_price

            if loss_pct < -self.max_position_loss_pct:
                logger.warning(
                    f"Position {symbol} exceeds max loss: {loss_pct:.2%}"
                )
                symbols_to_close.append(symbol)

        return symbols_to_close

    def _activate_circuit_breaker(self, reason: str) -> None:
        """
        Activate circuit breaker.

        Args:
            reason: Reason for activation
        """
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        logger.critical(f"🚨 CIRCUIT BREAKER ACTIVATED: {reason}")

    def reset_circuit_breaker(self, manual_override: bool = False) -> bool:
        """
        Reset circuit breaker.

        Args:
            manual_override: Manual override required

        Returns:
            Success status
        """
        if not manual_override and self.require_confirmation:
            logger.warning("Manual override required to reset circuit breaker")
            return False

        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        logger.info("Circuit breaker reset")

        return True

    def _check_day_reset(self, portfolio_value: float) -> None:
        """
        Check if it's a new day and reset counters.

        Args:
            portfolio_value: Current portfolio value
        """
        today = datetime.now().date()

        if today > self.last_reset:
            logger.info(f"New trading day - resetting counters")
            self.daily_trades = []
            self.start_of_day_value = portfolio_value
            self.last_reset = today

            # Optionally reset circuit breaker at start of day
            if self.circuit_breaker_active:
                logger.info("Auto-resetting circuit breaker for new day")
                self.circuit_breaker_active = False
                self.circuit_breaker_reason = None

        elif self.start_of_day_value is None:
            self.start_of_day_value = portfolio_value

    def get_status(self) -> Dict:
        """
        Get safety monitor status.

        Returns:
            Status dictionary
        """
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_reason': self.circuit_breaker_reason,
            'daily_trades': len(self.daily_trades),
            'max_daily_trades': self.max_trades_per_day,
            'trades_remaining': max(0, self.max_trades_per_day - len(self.daily_trades)),
            'start_of_day_value': self.start_of_day_value,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_position_loss_pct': self.max_position_loss_pct
        }


class LiveTradingConfirmation:
    """
    Require manual confirmation for live trades.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize confirmation system.

        Args:
            enabled: Whether confirmations are enabled
        """
        self.enabled = enabled
        self.pending_orders = []

    def request_confirmation(self, order: Dict) -> str:
        """
        Request confirmation for an order.

        Args:
            order: Order to confirm

        Returns:
            Confirmation ID
        """
        if not self.enabled:
            return "AUTO_APPROVED"

        confirmation_id = f"CONF_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.pending_orders)}"

        pending_order = {
            'confirmation_id': confirmation_id,
            'order': order,
            'timestamp': datetime.now(),
            'status': 'pending'
        }

        self.pending_orders.append(pending_order)

        logger.warning(
            f"🔔 CONFIRMATION REQUIRED: {order.get('action')} "
            f"{order.get('quantity')} {order.get('symbol')} "
            f"[ID: {confirmation_id}]"
        )

        return confirmation_id

    def confirm_order(self, confirmation_id: str, approved: bool) -> bool:
        """
        Confirm or reject an order.

        Args:
            confirmation_id: Confirmation ID
            approved: Whether order is approved

        Returns:
            Success status
        """
        for pending in self.pending_orders:
            if pending['confirmation_id'] == confirmation_id:
                if approved:
                    pending['status'] = 'approved'
                    logger.info(f"✅ Order {confirmation_id} approved")
                else:
                    pending['status'] = 'rejected'
                    logger.info(f"❌ Order {confirmation_id} rejected")

                return True

        logger.warning(f"Confirmation ID {confirmation_id} not found")
        return False

    def get_pending_orders(self) -> List[Dict]:
        """
        Get pending orders.

        Returns:
            List of pending orders
        """
        return [p for p in self.pending_orders if p['status'] == 'pending']

    def clean_old_confirmations(self, max_age_hours: int = 24) -> None:
        """
        Remove old confirmations.

        Args:
            max_age_hours: Maximum age in hours
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        self.pending_orders = [
            p for p in self.pending_orders
            if p['timestamp'] > cutoff
        ]


if __name__ == "__main__":
    # Test safety monitor
    monitor = SafetyMonitor(
        max_daily_loss_pct=0.05,
        max_trades_per_day=10
    )

    # Test order safety
    order = {
        'symbol': 'AAPL',
        'action': 'buy',
        'quantity': 10,
        'estimated_price': 150
    }

    positions = {}
    account_info = {'buying_power': 10000}

    is_safe, reason = monitor.check_order_safety(
        order,
        portfolio_value=10000,
        positions=positions,
        account_info=account_info
    )

    print(f"Order safe: {is_safe}")
    if not is_safe:
        print(f"Reason: {reason}")

    # Record trade
    if is_safe:
        monitor.record_trade(order)

    # Check status
    status = monitor.get_status()
    print("\nSafety Monitor Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
