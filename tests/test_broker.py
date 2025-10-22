"""
Tests for broker interface and order management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.execution.broker import AlpacaBroker, OrderManager, OrderStatus, OrderSide
from tests.conftest import assert_valid_signal


@pytest.mark.unit
class TestOrderManager:
    """Test OrderManager class."""

    def test_initialization(self, mock_broker):
        """Test order manager initialization."""
        order_mgr = OrderManager(mock_broker)

        assert order_mgr.broker == mock_broker
        assert isinstance(order_mgr.pending_orders, dict)
        assert isinstance(order_mgr.filled_orders, list)

    def test_execute_signal_buy(self, mock_broker):
        """Test executing buy signal."""
        order_mgr = OrderManager(mock_broker)

        # Setup mock broker
        mock_broker.get_positions.return_value = {}
        mock_broker.get_account_info.return_value = {
            'buying_power': 10000
        }
        mock_broker.get_current_price.return_value = 150.00

        signal = {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 0.1,
            'order_type': 'market'
        }

        order_id = order_mgr.execute_signal(signal)

        assert order_id is not None
        assert order_id in order_mgr.pending_orders

    def test_execute_signal_sell(self, mock_broker):
        """Test executing sell signal."""
        order_mgr = OrderManager(mock_broker)

        # Setup mock broker with existing position
        mock_broker.get_positions.return_value = {
            'AAPL': {'qty': 100}
        }
        mock_broker.get_current_price.return_value = 150.00

        signal = {
            'symbol': 'AAPL',
            'action': 'sell',
            'quantity': 0.5,  # Sell 50%
            'order_type': 'market'
        }

        order_id = order_mgr.execute_signal(signal)

        assert order_id is not None

    def test_execute_signal_hold(self, mock_broker):
        """Test executing hold signal."""
        order_mgr = OrderManager(mock_broker)

        signal = {
            'symbol': 'AAPL',
            'action': 'hold'
        }

        order_id = order_mgr.execute_signal(signal)

        assert order_id is None
        assert len(order_mgr.pending_orders) == 0

    def test_execute_signal_invalid(self, mock_broker):
        """Test executing invalid signal."""
        order_mgr = OrderManager(mock_broker)

        invalid_signal = {
            'action': 'invalid'
        }

        order_id = order_mgr.execute_signal(invalid_signal)

        assert order_id is None

    def test_execute_strategy_signals(self, mock_broker):
        """Test executing multiple signals."""
        order_mgr = OrderManager(mock_broker)

        mock_broker.get_positions.return_value = {}
        mock_broker.get_account_info.return_value = {'buying_power': 10000}
        mock_broker.get_current_price.return_value = 150.00

        signals = [
            {'symbol': 'AAPL', 'action': 'buy', 'quantity': 0.1},
            {'symbol': 'GOOGL', 'action': 'buy', 'quantity': 0.1},
            {'symbol': 'MSFT', 'action': 'hold'}
        ]

        order_ids = order_mgr.execute_strategy_signals(signals)

        assert len(order_ids) == 2  # Only buy orders

    def test_monitor_orders(self, mock_broker):
        """Test monitoring pending orders."""
        order_mgr = OrderManager(mock_broker)

        # Add a pending order
        order_mgr.pending_orders['order_1'] = {
            'id': 'order_1',
            'symbol': 'AAPL',
            'qty': 10
        }

        # Mock order status as filled
        mock_broker.get_order_status.return_value = {
            'id': 'order_1',
            'status': 'filled'
        }

        order_mgr.monitor_orders()

        # Order should be moved to filled
        assert 'order_1' not in order_mgr.pending_orders
        assert len(order_mgr.filled_orders) == 1

    def test_cancel_all_pending(self, mock_broker):
        """Test cancelling all pending orders."""
        order_mgr = OrderManager(mock_broker)

        # Add pending orders
        order_mgr.pending_orders = {
            'order_1': {'id': 'order_1'},
            'order_2': {'id': 'order_2'}
        }

        cancelled = order_mgr.cancel_all_pending()

        assert cancelled == 2

    def test_get_pending_orders(self, mock_broker):
        """Test getting pending orders."""
        order_mgr = OrderManager(mock_broker)

        order_mgr.pending_orders = {'order_1': {'id': 'order_1'}}

        pending = order_mgr.get_pending_orders()

        assert isinstance(pending, dict)
        assert 'order_1' in pending

    def test_validate_signal_valid(self, mock_broker):
        """Test signal validation with valid signal."""
        order_mgr = OrderManager(mock_broker)

        valid_signal = {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 0.1
        }

        assert order_mgr._validate_signal(valid_signal) is True

    def test_validate_signal_invalid(self, mock_broker):
        """Test signal validation with invalid signal."""
        order_mgr = OrderManager(mock_broker)

        invalid_signal = {
            'action': 'invalid_action'
        }

        assert order_mgr._validate_signal(invalid_signal) is False


@pytest.mark.unit
class TestAlpacaBroker:
    """Test AlpacaBroker class (with mocking)."""

    @patch('src.execution.broker.tradeapi.REST')
    def test_initialization_paper(self, mock_rest):
        """Test Alpaca broker initialization in paper mode."""
        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        assert broker.paper is True
        assert broker.api_key == 'test_key'

    @patch('src.execution.broker.tradeapi.REST')
    def test_connect_success(self, mock_rest):
        """Test successful connection."""
        mock_api = Mock()
        mock_api.get_account.return_value = Mock(status='ACTIVE')
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        assert broker.connect() is True

    @patch('src.execution.broker.tradeapi.REST')
    def test_connect_failure(self, mock_rest):
        """Test failed connection."""
        mock_api = Mock()
        mock_api.get_account.side_effect = Exception("Connection failed")
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        assert broker.connect() is False

    @patch('src.execution.broker.tradeapi.REST')
    def test_place_order_market(self, mock_rest):
        """Test placing market order."""
        mock_api = Mock()
        mock_order = Mock(id='order_123')
        mock_api.submit_order.return_value = mock_order
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        order_id = broker.place_order(
            symbol='AAPL',
            quantity=10,
            side='buy',
            order_type='market'
        )

        assert order_id == 'order_123'

    @patch('src.execution.broker.tradeapi.REST')
    def test_place_order_invalid_quantity(self, mock_rest):
        """Test placing order with invalid quantity."""
        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        order_id = broker.place_order(
            symbol='AAPL',
            quantity=0,
            side='buy'
        )

        assert order_id is None

    @patch('src.execution.broker.tradeapi.REST')
    def test_cancel_order(self, mock_rest):
        """Test cancelling order."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        result = broker.cancel_order('order_123')

        assert result is True
        mock_api.cancel_order.assert_called_once_with('order_123')

    @patch('src.execution.broker.tradeapi.REST')
    def test_get_positions(self, mock_rest, sample_positions):
        """Test getting positions."""
        mock_api = Mock()

        # Create mock position objects
        mock_pos1 = Mock()
        mock_pos1.symbol = 'AAPL'
        mock_pos1.qty = '100'
        mock_pos1.market_value = '15000'
        mock_pos1.avg_entry_price = '145.00'
        mock_pos1.current_price = '150.00'
        mock_pos1.unrealized_pl = '500'
        mock_pos1.unrealized_plpc = '0.0345'
        mock_pos1.side = 'long'

        mock_api.list_positions.return_value = [mock_pos1]
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        positions = broker.get_positions()

        assert isinstance(positions, dict)
        assert 'AAPL' in positions

    @patch('src.execution.broker.tradeapi.REST')
    def test_get_account_info(self, mock_rest):
        """Test getting account information."""
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = 'TEST123'
        mock_account.status = 'ACTIVE'
        mock_account.currency = 'USD'
        mock_account.cash = '50000'
        mock_account.portfolio_value = '100000'
        mock_account.buying_power = '80000'
        mock_account.equity = '100000'
        mock_account.last_equity = '98000'
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked = False
        mock_account.transfers_blocked = False
        mock_account.account_blocked = False

        mock_api.get_account.return_value = mock_account
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        account_info = broker.get_account_info()

        assert isinstance(account_info, dict)
        assert account_info['status'] == 'ACTIVE'
        assert account_info['cash'] == 50000.0

    @patch('src.execution.broker.tradeapi.REST')
    def test_get_current_price(self, mock_rest):
        """Test getting current price."""
        mock_api = Mock()
        mock_trade = Mock()
        mock_trade.price = 150.00
        mock_api.get_latest_trade.return_value = mock_trade
        mock_rest.return_value = mock_api

        broker = AlpacaBroker('test_key', 'test_secret', paper=True)

        price = broker.get_current_price('AAPL')

        assert price == 150.00
