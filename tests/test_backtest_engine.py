"""
Tests for backtesting engine and performance analyzer.
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.backtest_engine import BacktestEngine, PerformanceAnalyzer
from tests.conftest import assert_valid_metrics


@pytest.mark.unit
class TestBacktestEngine:
    """Test BacktestEngine class."""

    def test_initialization(self):
        """Test backtest engine initialization."""
        engine = BacktestEngine(
            initial_cash=10000,
            commission=0.001,
            slippage=0.0005
        )

        assert engine.initial_cash == 10000
        assert engine.commission == 0.001
        assert engine.slippage == 0.0005
        assert engine.cash == 10000

    def test_run_simple_strategy(self, sample_ohlcv_data):
        """Test running a simple strategy."""
        engine = BacktestEngine(initial_cash=10000)

        # Simple buy and hold strategy
        def buy_hold_strategy(data):
            if len(data) == 20:
                return {
                    'symbol': 'AAPL',
                    'action': 'buy',
                    'quantity': 1.0
                }
            return {'action': 'hold'}

        # Prepare data
        data = sample_ohlcv_data[['close']].copy()
        data.columns = ['AAPL']

        results = engine.run(data, buy_hold_strategy)

        assert isinstance(results, dict)
        assert 'final_value' in results
        assert 'total_return' in results
        assert 'num_trades' in results

    def test_execute_buy_trade(self, sample_ohlcv_data):
        """Test executing buy trade."""
        engine = BacktestEngine(initial_cash=10000)

        signal = {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 0.5  # Use 50% of cash
        }

        initial_cash = engine.cash
        prices = sample_ohlcv_data['close'].iloc[0]

        engine._execute_trade(signal, pd.Series({'AAPL': prices}))

        # Cash should decrease
        assert engine.cash < initial_cash

        # Should have position
        assert 'AAPL' in engine.positions

    def test_execute_sell_trade(self, sample_ohlcv_data):
        """Test executing sell trade."""
        engine = BacktestEngine(initial_cash=10000)

        # First buy
        engine.positions['AAPL'] = {
            'shares': 100,
            'avg_price': 100.00
        }

        signal = {
            'symbol': 'AAPL',
            'action': 'sell',
            'quantity': 1.0  # Sell all
        }

        initial_cash = engine.cash
        engine._execute_trade(signal, pd.Series({'AAPL': 110.00}))

        # Cash should increase
        assert engine.cash > initial_cash

        # Position should be closed
        assert 'AAPL' not in engine.positions

    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation."""
        engine = BacktestEngine(initial_cash=10000)

        engine.cash = 5000
        engine.positions = {
            'AAPL': {'shares': 10, 'avg_price': 100},
            'GOOGL': {'shares': 5, 'avg_price': 200}
        }

        current_prices = pd.Series({
            'AAPL': 150.00,
            'GOOGL': 250.00
        })

        portfolio_value = engine._calculate_portfolio_value(current_prices)

        # 5000 cash + 10*150 + 5*250 = 8250
        expected = 5000 + 1500 + 1250
        assert portfolio_value == expected

    def test_generate_results(self, sample_ohlcv_data):
        """Test results generation."""
        engine = BacktestEngine(initial_cash=10000)

        # Simulate some portfolio values
        for i in range(10):
            engine.portfolio_values.append({
                'timestamp': i,
                'value': 10000 + i * 100,
                'cash': 5000,
                'positions_value': 5000 + i * 100
            })

        # Add some trades
        engine.trades = [
            {'step': 1, 'symbol': 'AAPL', 'action': 'buy', 'shares': 10, 'price': 100},
            {'step': 5, 'symbol': 'AAPL', 'action': 'sell', 'shares': 10,
             'price': 110, 'profit': 100}
        ]

        data = sample_ohlcv_data[['close']].copy()
        results = engine._generate_results(data)

        assert isinstance(results, dict)
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'win_rate' in results

    def test_commission_and_slippage(self):
        """Test that commission and slippage are applied."""
        engine = BacktestEngine(
            initial_cash=10000,
            commission=0.01,  # 1% commission
            slippage=0.01     # 1% slippage
        )

        signal = {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 1.0
        }

        initial_cash = engine.cash
        engine._execute_trade(signal, pd.Series({'AAPL': 100.00}))

        # Cost should include commission and slippage
        # Effective price = 100 * (1 + 0.01) = 101
        # Cost = shares * 101 * (1 + 0.01) ≈ more than 100 per share
        assert engine.cash < initial_cash - 10000 * 0.98


@pytest.mark.unit
class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""

    def test_initialization(self):
        """Test performance analyzer initialization."""
        analyzer = PerformanceAnalyzer()

        assert analyzer is not None

    def test_calculate_metrics(self, sample_backtest_results):
        """Test calculating performance metrics."""
        analyzer = PerformanceAnalyzer()

        portfolio_values = sample_backtest_results['portfolio_history']['value']
        trades = sample_backtest_results['trades_history']

        metrics = analyzer.calculate_metrics(portfolio_values, trades)

        required_metrics = [
            'total_return',
            'annual_return',
            'volatility',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'calmar_ratio'
        ]

        assert_valid_metrics(metrics, required_metrics)

    def test_calculate_metrics_with_benchmark(self):
        """Test calculating metrics with benchmark."""
        analyzer = PerformanceAnalyzer()

        # Create sample data
        portfolio_values = pd.Series([10000 + i * 100 for i in range(100)])
        benchmark = pd.Series([10000 + i * 80 for i in range(100)])
        trades = pd.DataFrame()

        metrics = analyzer.calculate_metrics(portfolio_values, trades, benchmark)

        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert isinstance(metrics['alpha'], float)
        assert isinstance(metrics['beta'], float)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer()

        # Portfolio with positive returns
        portfolio_values = pd.Series([10000 * (1.01 ** i) for i in range(252)])
        trades = pd.DataFrame()

        metrics = analyzer.calculate_metrics(portfolio_values, trades)

        assert metrics['sharpe_ratio'] > 0

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        analyzer = PerformanceAnalyzer()

        # Portfolio with a drawdown
        values = [10000, 11000, 12000, 9000, 10000, 11000]
        portfolio_values = pd.Series(values)
        trades = pd.DataFrame()

        metrics = analyzer.calculate_metrics(portfolio_values, trades)

        assert metrics['max_drawdown'] < 0
        # Should be approximately -25% (from 12000 to 9000)
        assert metrics['max_drawdown'] <= -0.20

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        analyzer = PerformanceAnalyzer()

        # Generate returns with some volatility
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        portfolio_values = pd.Series([10000 * np.prod(1 + returns[:i+1])
                                     for i in range(252)])
        trades = pd.DataFrame()

        metrics = analyzer.calculate_metrics(portfolio_values, trades)

        assert 'sortino_ratio' in metrics
        assert isinstance(metrics['sortino_ratio'], float)

    def test_compare_strategies(self):
        """Test comparing multiple strategies."""
        analyzer = PerformanceAnalyzer()

        results_list = [
            {
                'strategy_name': 'Strategy A',
                'total_return': 0.20,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.10,
                'win_rate': 0.60,
                'num_trades': 20
            },
            {
                'strategy_name': 'Strategy B',
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.08,
                'win_rate': 0.65,
                'num_trades': 15
            }
        ]

        comparison = analyzer.compare_strategies(results_list)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Strategy' in comparison.columns
        assert 'Sharpe Ratio' in comparison.columns

        # Should be sorted by Sharpe ratio
        assert comparison.iloc[0]['Sharpe Ratio'] >= comparison.iloc[1]['Sharpe Ratio']

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        analyzer = PerformanceAnalyzer()

        # Create trades with wins and losses
        trades = pd.DataFrame({
            'profit': [100, -50, 150, -30, 200]
        })

        portfolio_values = pd.Series([10000 + i * 100 for i in range(100)])

        metrics = analyzer.calculate_metrics(portfolio_values, trades)

        # 3 wins out of 5 trades = 60%
        assert metrics['win_rate'] == 0.60
        assert metrics['num_trades'] == 5

    def test_empty_trades(self):
        """Test metrics with no trades."""
        analyzer = PerformanceAnalyzer()

        portfolio_values = pd.Series([10000 + i * 100 for i in range(100)])
        trades = pd.DataFrame()

        metrics = analyzer.calculate_metrics(portfolio_values, trades)

        assert metrics['num_trades'] == 0
        assert metrics['win_rate'] == 0
