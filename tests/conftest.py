"""
Shared pytest fixtures and configuration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

    data = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.5,
        'high': close_prices + abs(np.random.randn(100) * 1),
        'low': close_prices - abs(np.random.randn(100) * 1),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


@pytest.fixture
def sample_multi_symbol_data():
    """Generate sample multi-symbol price data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    np.random.seed(42)

    data = pd.DataFrame({
        'AAPL': 150 + np.cumsum(np.random.randn(100) * 2),
        'GOOGL': 120 + np.cumsum(np.random.randn(100) * 1.5),
        'MSFT': 300 + np.cumsum(np.random.randn(100) * 3),
    }, index=dates)

    return data


@pytest.fixture
def sample_news_articles():
    """Generate sample news articles."""
    return [
        {
            'symbol': 'AAPL',
            'title': 'Apple announces record quarterly earnings',
            'description': 'Apple Inc. reported better than expected results.',
            'published_at': datetime.now().isoformat(),
            'source': 'TechNews'
        },
        {
            'symbol': 'AAPL',
            'title': 'Supply chain concerns for Apple',
            'description': 'Analysts worry about potential disruptions.',
            'published_at': datetime.now().isoformat(),
            'source': 'MarketWatch'
        },
        {
            'symbol': 'GOOGL',
            'title': 'Google launches new AI features',
            'description': 'Google announces major AI improvements.',
            'published_at': datetime.now().isoformat(),
            'source': 'TechCrunch'
        }
    ]


@pytest.fixture
def sample_predictions():
    """Generate sample price predictions."""
    return {
        'AAPL': {
            'predicted_price': 152.50,
            'current_price': 150.00,
            'confidence_interval': (148.00, 157.00),
            'probability_up': 0.65,
            'model_contributions': {'xgboost': 152.30, 'prophet': 152.70}
        },
        'GOOGL': {
            'predicted_price': 122.00,
            'current_price': 120.00,
            'confidence_interval': (118.00, 126.00),
            'probability_up': 0.58,
            'model_contributions': {'xgboost': 121.80, 'prophet': 122.20}
        }
    }


@pytest.fixture
def sample_portfolio():
    """Generate sample portfolio."""
    return {
        'AAPL': 0.40,
        'GOOGL': 0.35,
        'MSFT': 0.25
    }


@pytest.fixture
def sample_positions():
    """Generate sample broker positions."""
    return {
        'AAPL': {
            'symbol': 'AAPL',
            'qty': 100,
            'market_value': 15000.00,
            'avg_entry_price': 145.00,
            'current_price': 150.00,
            'unrealized_pl': 500.00,
            'unrealized_plpc': 0.0345,
            'side': 'long'
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'qty': 50,
            'market_value': 6000.00,
            'avg_entry_price': 115.00,
            'current_price': 120.00,
            'unrealized_pl': 250.00,
            'unrealized_plpc': 0.0435,
            'side': 'long'
        }
    }


@pytest.fixture
def sample_account_info():
    """Generate sample account information."""
    return {
        'account_number': 'TEST123',
        'status': 'ACTIVE',
        'currency': 'USD',
        'cash': 50000.00,
        'portfolio_value': 100000.00,
        'buying_power': 80000.00,
        'equity': 100000.00,
        'last_equity': 98000.00,
        'pattern_day_trader': False,
        'trading_blocked': False,
        'transfers_blocked': False,
        'account_blocked': False
    }


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config_content = """
system:
  mode: paper
  initial_capital: 100000
  execution_interval: 300
  log_level: INFO

assets:
  symbols:
    - AAPL
    - GOOGL
  timeframes:
    - 1h
    - 1d

brokers:
  default: alpaca
  alpaca:
    paper: true

strategies:
  llm:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.7
  rl:
    algorithm: PPO

portfolio:
  optimization_method: max_sharpe
  max_position_size: 0.2

risk:
  max_drawdown: 0.15
  stop_loss_pct: 0.05
  var_confidence: 0.95
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def temp_env_file(tmp_path, monkeypatch):
    """Create temporary environment variables."""
    env_vars = {
        'ALPACA_PAPER_KEY': 'test_paper_key',
        'ALPACA_PAPER_SECRET': 'test_paper_secret',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'NEWS_API_KEY': 'test_news_key'
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def mock_llm_response():
    """Generate mock LLM response."""
    return """
```python
class TestStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "TestStrategy"

    def generate_signal(self, data, predictions=None, sentiment=None):
        if len(data) < 20:
            return {'action': 'hold'}

        rsi = data.get('rsi', pd.Series([50]))

        if rsi.iloc[-1] < 30:
            return {
                'action': 'buy',
                'confidence': 0.7,
                'quantity': 0.1,
                'reason': 'RSI oversold'
            }
        elif rsi.iloc[-1] > 70:
            return {
                'action': 'sell',
                'confidence': 0.7,
                'quantity': 1.0,
                'reason': 'RSI overbought'
            }

        return {'action': 'hold'}
```
"""


@pytest.fixture
def sample_trades():
    """Generate sample trade history."""
    return [
        {
            'step': 10,
            'symbol': 'AAPL',
            'action': 'buy',
            'shares': 100,
            'price': 145.00,
            'cost': 14500.00
        },
        {
            'step': 50,
            'symbol': 'AAPL',
            'action': 'sell',
            'shares': 100,
            'price': 150.00,
            'proceeds': 14985.00,
            'profit': 485.00
        },
        {
            'step': 60,
            'symbol': 'GOOGL',
            'action': 'buy',
            'shares': 50,
            'price': 115.00,
            'cost': 5750.00
        }
    ]


@pytest.fixture
def sample_backtest_results():
    """Generate sample backtest results."""
    portfolio_values = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
        'value': 10000 + np.cumsum(np.random.randn(100) * 100),
        'cash': 5000 + np.random.randn(100) * 500,
        'positions_value': 5000 + np.random.randn(100) * 500
    })

    portfolio_values['returns'] = portfolio_values['value'].pct_change()

    return {
        'initial_value': 10000,
        'final_value': 12000,
        'total_return': 0.20,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.0,
        'max_drawdown': -0.10,
        'num_trades': 20,
        'win_rate': 0.60,
        'avg_profit': 100,
        'avg_loss': -50,
        'profit_factor': 2.0,
        'portfolio_history': portfolio_values,
        'trades_history': pd.DataFrame()
    }


# Mock classes for testing

class MockBroker:
    """Mock broker for testing."""

    def __init__(self):
        self.connected = True
        self.orders = {}
        self.positions = {}

    def connect(self):
        return self.connected

    def place_order(self, symbol, quantity, side, order_type='market',
                   limit_price=None, stop_price=None):
        order_id = f"order_{len(self.orders) + 1}"
        self.orders[order_id] = {
            'id': order_id,
            'symbol': symbol,
            'qty': quantity,
            'side': side,
            'type': order_type,
            'status': 'filled'
        }
        return order_id

    def cancel_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            return True
        return False

    def get_order_status(self, order_id):
        return self.orders.get(order_id, {})

    def get_positions(self):
        return self.positions

    def get_account_info(self):
        return {
            'portfolio_value': 100000.00,
            'cash': 50000.00,
            'buying_power': 80000.00
        }

    def get_current_price(self, symbol):
        prices = {'AAPL': 150.00, 'GOOGL': 120.00, 'MSFT': 300.00}
        return prices.get(symbol, 100.00)


@pytest.fixture
def mock_broker():
    """Provide mock broker instance."""
    return MockBroker()


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response_text=None):
        self.response_text = response_text or "Mock response"

    def messages_create(self, **kwargs):
        class Response:
            def __init__(self, text):
                self.content = [type('obj', (object,), {'text': text})]

        return Response(self.response_text)


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Provide mock LLM client."""
    return MockLLMClient(mock_llm_response)


# Utility functions for testing

def assert_valid_dataframe(df, required_columns=None):
    """Assert that a DataFrame is valid."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert not df.empty

    if required_columns:
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"


def assert_valid_signal(signal):
    """Assert that a trading signal is valid."""
    assert isinstance(signal, dict)
    assert 'action' in signal
    assert signal['action'] in ['buy', 'sell', 'hold']

    if signal['action'] != 'hold':
        assert 'quantity' in signal
        assert 'confidence' in signal
        assert 0 <= signal['confidence'] <= 1
        assert 0 <= signal['quantity'] <= 1


def assert_valid_metrics(metrics, required_metrics=None):
    """Assert that performance metrics are valid."""
    assert isinstance(metrics, dict)

    if required_metrics:
        for metric in required_metrics:
            assert metric in metrics, f"Missing required metric: {metric}"

    # Check for NaN values
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert not np.isnan(value), f"Metric {key} is NaN"
