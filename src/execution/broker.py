"""
Broker interface and integrations.
Provides unified interface for multiple brokers with paper trading default.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import alpaca_trade_api as tradeapi
from enum import Enum


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class BrokerInterface(ABC):
    """Abstract base class for broker integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker API.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """
        Place an order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            order_type: Order type
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Order ID
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Dictionary with order details
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions.

        Returns:
            Dictionary mapping symbols to position details
        """
        pass

    @abstractmethod
    def get_account_info(self) -> Dict:
        """
        Get account information.

        Returns:
            Dictionary with account details
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        pass


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker integration.
    Defaults to paper trading for safety.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True
    ):
        """
        Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (default True)
        """
        self.paper = paper
        self.api_key = api_key
        self.secret_key = secret_key

        base_url = ("https://paper-api.alpaca.markets" if paper
                   else "https://api.alpaca.markets")

        self.api = tradeapi.REST(
            api_key,
            secret_key,
            base_url,
            api_version='v2'
        )

        mode = "PAPER" if paper else "LIVE"
        logger.info(f"Initialized Alpaca broker in {mode} mode")

    def connect(self) -> bool:
        """Test connection to Alpaca."""
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca - Account status: {account.status}")
            return account.status == 'ACTIVE'
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """Place order with Alpaca."""
        try:
            # Convert quantity to integer shares
            qty = int(quantity)

            if qty <= 0:
                logger.warning(f"Invalid quantity: {qty}")
                return None

            # Place order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day',
                limit_price=limit_price,
                stop_price=stop_price
            )

            logger.info(f"Order placed: {side} {qty} {symbol} @ {order_type} - ID: {order.id}")
            return order.id

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        try:
            order = self.api.get_order(order_id)

            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'filled_qty': float(order.filled_qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'created_at': order.created_at,
                'filled_at': order.filled_at,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {}

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        try:
            positions = self.api.list_positions()

            result = {}
            for position in positions:
                result[position.symbol] = {
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': position.side
                }

            logger.debug(f"Retrieved {len(result)} positions")
            return result

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_account_info(self) -> Dict:
        """Get account information."""
        try:
            account = self.api.get_account()

            return {
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        try:
            # Get latest trade
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0

    def get_bars(self, symbol: str, timeframe: str = '1Day', limit: int = 100) -> List[Dict]:
        """
        Get historical bars.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            limit: Number of bars

        Returns:
            List of bar data
        """
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                limit=limit
            ).df

            return bars.to_dict('records')

        except Exception as e:
            logger.error(f"Failed to get bars: {e}")
            return []


class OrderManager:
    """
    Order management system.
    Handles order execution, tracking, and validation.
    """

    def __init__(self, broker: BrokerInterface):
        """
        Initialize order manager.

        Args:
            broker: Broker interface instance
        """
        self.broker = broker
        self.pending_orders = {}
        self.filled_orders = []
        self.cancelled_orders = []

        logger.info("Initialized OrderManager")

    def execute_signal(self, signal: Dict) -> Optional[str]:
        """
        Execute a trading signal.

        Args:
            signal: Signal dictionary with action, symbol, quantity

        Returns:
            Order ID if successful
        """
        if not self._validate_signal(signal):
            logger.error(f"Invalid signal: {signal}")
            return None

        symbol = signal['symbol']
        action = signal['action']
        quantity = signal.get('quantity', 0)

        if action == 'hold':
            logger.info(f"Signal is HOLD for {symbol}, no order placed")
            return None

        # Get current positions
        positions = self.broker.get_positions()
        current_position = positions.get(symbol, {}).get('qty', 0)

        # Determine order parameters
        if action == 'buy':
            side = 'buy'
            # Calculate shares to buy
            account_info = self.broker.get_account_info()
            buying_power = account_info.get('buying_power', 0)
            current_price = self.broker.get_current_price(symbol)

            if current_price == 0:
                logger.error(f"Could not get price for {symbol}")
                return None

            max_shares = int((buying_power * quantity) / current_price)
            order_qty = max_shares

        elif action == 'sell':
            side = 'sell'
            # Sell portion or all of position
            order_qty = int(abs(current_position) * quantity)

            if order_qty == 0:
                logger.warning(f"No position to sell for {symbol}")
                return None

        else:
            logger.error(f"Unknown action: {action}")
            return None

        # Place order
        order_type = signal.get('order_type', 'market')
        limit_price = signal.get('limit_price')
        stop_price = signal.get('stop_price')

        order_id = self.broker.place_order(
            symbol=symbol,
            quantity=order_qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )

        if order_id:
            self.pending_orders[order_id] = {
                'id': order_id,
                'symbol': symbol,
                'qty': order_qty,
                'side': side,
                'type': order_type,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }

        return order_id

    def execute_strategy_signals(self, signals: List[Dict]) -> List[str]:
        """
        Execute multiple signals from strategies.

        Args:
            signals: List of signal dictionaries

        Returns:
            List of order IDs
        """
        order_ids = []

        for signal in signals:
            order_id = self.execute_signal(signal)
            if order_id:
                order_ids.append(order_id)

        logger.info(f"Executed {len(order_ids)} orders from {len(signals)} signals")
        return order_ids

    def monitor_orders(self) -> None:
        """Monitor pending orders and update statuses."""
        completed_orders = []

        for order_id, order_info in self.pending_orders.items():
            try:
                status = self.broker.get_order_status(order_id)

                order_status = status.get('status', '').lower()

                if order_status in ['filled', 'partially_filled']:
                    logger.info(f"Order {order_id} filled: {status}")
                    self.filled_orders.append(status)
                    completed_orders.append(order_id)

                elif order_status in ['cancelled', 'rejected']:
                    logger.warning(f"Order {order_id} {order_status}")
                    self.cancelled_orders.append(status)
                    completed_orders.append(order_id)

            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")

        # Remove completed orders from pending
        for order_id in completed_orders:
            del self.pending_orders[order_id]

        if completed_orders:
            logger.info(f"Updated status for {len(completed_orders)} orders")

    def cancel_all_pending(self) -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0

        for order_id in list(self.pending_orders.keys()):
            if self.broker.cancel_order(order_id):
                cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} pending orders")
        return cancelled_count

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate signal format."""
        required_keys = ['action', 'symbol']

        if not all(key in signal for key in required_keys):
            return False

        if signal['action'] not in ['buy', 'sell', 'hold']:
            return False

        return True

    def get_pending_orders(self) -> Dict:
        """Get pending orders."""
        return self.pending_orders.copy()

    def get_filled_orders(self) -> List[Dict]:
        """Get filled orders."""
        return self.filled_orders.copy()

    def get_order_history(self) -> List[Dict]:
        """Get all order history."""
        return self.filled_orders + self.cancelled_orders


if __name__ == "__main__":
    import os

    # Test broker connection
    api_key = os.getenv('ALPACA_PAPER_KEY')
    secret_key = os.getenv('ALPACA_PAPER_SECRET')

    if not api_key or not secret_key:
        print("Please set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET")
        exit(1)

    # Initialize broker
    broker = AlpacaBroker(api_key, secret_key, paper=True)

    # Test connection
    if broker.connect():
        print("Connected successfully")

        # Get account info
        account = broker.get_account_info()
        print(f"Account Value: ${account['portfolio_value']:.2f}")

        # Get positions
        positions = broker.get_positions()
        print(f"Current Positions: {len(positions)}")

        # Initialize order manager
        order_manager = OrderManager(broker)

        # Test signal (DON'T ACTUALLY EXECUTE)
        test_signal = {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 0.01,  # 1% of buying power
            'order_type': 'market'
        }

        # Uncomment to test order execution
        # order_id = order_manager.execute_signal(test_signal)
        # print(f"Order ID: {order_id}")
