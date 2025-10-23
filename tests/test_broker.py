"""
Tests for broker and order management.
"""

import pytest
from unittest.mock import Mock, patch
from src.execution.broker import AlpacaBroker, OrderManager, OrderStatus, OrderSide, OrderType


@pytest.fixture
def mock_broker():
    """Fixture for a mocked broker."""
    broker = Mock()
    broker.get_positions.return_value = {'AAPL': {'qty': 10}}
    broker.get_account_info.return_value = {'buying_power': 10000}
    broker.get_current_price.return_value = 150.0
    broker.place_order.return_value = 'test_order_id'
    return broker


@pytest.mark.unit
class TestOrderManager:
    """Test OrderManager class."""

    def test_initialization(self, mock_broker):
        """Test initialization."""
        order_manager = OrderManager(mock_broker)
        assert order_manager.broker == mock_broker

    def test_execute_signal_buy(self, mock_broker):
        """Test executing a buy signal."""
        order_manager = OrderManager(mock_broker)
        signal = {'action': 'buy', 'symbol': 'AAPL', 'quantity': 0.1}
        order_id = order_manager.execute_signal(signal)
        assert order_id == 'test_order_id'
        mock_broker.place_order.assert_called_once()

    def test_execute_signal_sell(self, mock_broker):
        """Test executing a sell signal."""
        order_manager = OrderManager(mock_broker)
        signal = {'action': 'sell', 'symbol': 'AAPL', 'quantity': 0.5}
        order_id = order_manager.execute_signal(signal)
        assert order_id == 'test_order_id'
        mock_broker.place_order.assert_called_once()

    def test_execute_strategy_signals(self, mock_broker):
        """Test executing multiple signals."""
        order_manager = OrderManager(mock_broker)
        signals = [
            {'action': 'buy', 'symbol': 'AAPL', 'quantity': 0.1},
            {'action': 'sell', 'symbol': 'GOOGL', 'quantity': 1.0}
        ]
        mock_broker.get_positions.return_value = {
            'AAPL': {'qty': 10},
            'GOOGL': {'qty': 5}
        }
        order_ids = order_manager.execute_strategy_signals(signals)
        assert len(order_ids) == 2
        assert mock_broker.place_order.call_count == 2

    def test_monitor_orders(self, mock_broker):
        """Test monitoring orders."""
        order_manager = OrderManager(mock_broker)
        order_manager.pending_orders = {'test_order_id': {}}
        mock_broker.get_order_status.return_value = {'status': 'filled'}
        order_manager.monitor_orders()
        assert not order_manager.pending_orders
        assert order_manager.filled_orders

    def test_cancel_all_pending(self, mock_broker):
        """Test cancelling all pending orders."""
        order_manager = OrderManager(mock_broker)
        order_manager.pending_orders = {'test_order_id': {}}
        mock_broker.cancel_order.return_value = True
        cancelled_count = order_manager.cancel_all_pending()
        assert cancelled_count == 1
