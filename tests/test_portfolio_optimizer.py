"""
Tests for portfolio optimizer and risk manager.
"""

import pytest
import pandas as pd
import numpy as np
from src.execution.portfolio_optimizer import PortfolioOptimizer, RiskManager
from tests.conftest import assert_valid_metrics


@pytest.mark.unit
class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = PortfolioOptimizer(optimization_method='max_sharpe')

        assert optimizer.method == 'max_sharpe'

    def test_optimize_allocation_max_sharpe(self, sample_multi_symbol_data):
        """Test max Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(optimization_method='max_sharpe')

        weights = optimizer.optimize_allocation(sample_multi_symbol_data)

        assert isinstance(weights, dict)
        assert len(weights) > 0

        # Check weights sum to approximately 1
        total_weight = sum(weights.values())
        assert 0.95 <= total_weight <= 1.05

        # Check all weights are non-negative
        assert all(w >= 0 for w in weights.values())

    def test_optimize_allocation_min_volatility(self, sample_multi_symbol_data):
        """Test minimum volatility optimization."""
        optimizer = PortfolioOptimizer(optimization_method='min_volatility')

        weights = optimizer.optimize_allocation(sample_multi_symbol_data)

        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_optimize_allocation_single_asset(self):
        """Test optimization with single asset."""
        df = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104]
        })

        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize_allocation(df)

        assert weights == {'AAPL': 1.0}

    def test_optimize_allocation_with_predictions(self, sample_multi_symbol_data):
        """Test optimization with price predictions."""
        optimizer = PortfolioOptimizer()

        predictions = {
            'AAPL': 155.0,
            'GOOGL': 125.0,
            'MSFT': 310.0
        }

        weights = optimizer.optimize_allocation(
            sample_multi_symbol_data,
            predictions=predictions
        )

        assert isinstance(weights, dict)

    def test_calculate_metrics(self, sample_multi_symbol_data, sample_portfolio):
        """Test portfolio metrics calculation."""
        optimizer = PortfolioOptimizer()

        metrics = optimizer.calculate_metrics(
            sample_portfolio,
            sample_multi_symbol_data
        )

        required_metrics = [
            'expected_return',
            'volatility',
            'sharpe_ratio',
            'max_drawdown'
        ]
        assert_valid_metrics(metrics, required_metrics)

        # Check metric ranges
        assert -1 <= metrics['max_drawdown'] <= 0
        assert metrics['volatility'] >= 0

    def test_rebalance_check_needed(self, sample_portfolio):
        """Test rebalancing check when rebalancing is needed."""
        optimizer = PortfolioOptimizer()

        current_weights = {'AAPL': 0.50, 'GOOGL': 0.30, 'MSFT': 0.20}
        optimal_weights = {'AAPL': 0.40, 'GOOGL': 0.35, 'MSFT': 0.25}

        needs_rebalancing, trades = optimizer.rebalance_check(
            current_weights,
            optimal_weights,
            threshold=0.05
        )

        assert needs_rebalancing is True
        assert isinstance(trades, dict)

    def test_rebalance_check_not_needed(self, sample_portfolio):
        """Test rebalancing check when rebalancing is not needed."""
        optimizer = PortfolioOptimizer()

        current_weights = {'AAPL': 0.40, 'GOOGL': 0.35, 'MSFT': 0.25}
        optimal_weights = {'AAPL': 0.41, 'GOOGL': 0.34, 'MSFT': 0.25}

        needs_rebalancing, trades = optimizer.rebalance_check(
            current_weights,
            optimal_weights,
            threshold=0.05
        )

        assert needs_rebalancing is False

    def test_risk_parity_optimization(self, sample_multi_symbol_data):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(optimization_method='risk_parity')

        weights = optimizer.optimize_allocation(sample_multi_symbol_data)

        assert isinstance(weights, dict)
        assert len(weights) > 0


@pytest.mark.unit
class TestRiskManager:
    """Test RiskManager class."""

    def test_initialization(self):
        """Test risk manager initialization."""
        risk_mgr = RiskManager(
            max_position_size=0.2,
            max_portfolio_var=0.05,
            stop_loss_pct=0.05
        )

        assert risk_mgr.max_position_size == 0.2
        assert risk_mgr.max_portfolio_var == 0.05
        assert risk_mgr.stop_loss_pct == 0.05

    def test_check_position_size_within_limit(self):
        """Test position size check when within limit."""
        risk_mgr = RiskManager(max_position_size=0.2)

        adjusted_size = risk_mgr.check_position_size(
            'AAPL',
            proposed_size=15000,
            portfolio_value=100000
        )

        assert adjusted_size == 15000

    def test_check_position_size_exceeds_limit(self):
        """Test position size check when exceeding limit."""
        risk_mgr = RiskManager(max_position_size=0.2)

        adjusted_size = risk_mgr.check_position_size(
            'AAPL',
            proposed_size=30000,
            portfolio_value=100000
        )

        assert adjusted_size == 20000  # Capped at 20%

    def test_calculate_var(self, sample_multi_symbol_data, sample_portfolio):
        """Test Value at Risk calculation."""
        risk_mgr = RiskManager()

        var = risk_mgr.calculate_var(
            sample_portfolio,
            sample_multi_symbol_data,
            confidence=0.95
        )

        assert isinstance(var, float)
        assert var < 0  # VaR should be negative

    def test_calculate_cvar(self, sample_multi_symbol_data, sample_portfolio):
        """Test Conditional Value at Risk calculation."""
        risk_mgr = RiskManager()

        cvar = risk_mgr.calculate_cvar(
            sample_portfolio,
            sample_multi_symbol_data,
            confidence=0.95
        )

        assert isinstance(cvar, float)
        assert cvar < 0  # CVaR should be negative

    def test_check_correlation(self, sample_multi_symbol_data, sample_portfolio):
        """Test correlation check."""
        risk_mgr = RiskManager(max_correlation=0.7)

        result = risk_mgr.check_correlation(
            sample_portfolio,
            sample_multi_symbol_data
        )

        assert isinstance(result, dict)
        assert 'diversified' in result
        assert 'high_correlations' in result
        assert isinstance(result['diversified'], bool)

    def test_apply_stop_loss_triggered(self):
        """Test stop loss when triggered."""
        risk_mgr = RiskManager(stop_loss_pct=0.05)

        positions = {
            'AAPL': {'shares': 100, 'entry_price': 150.00},
            'GOOGL': {'shares': 50, 'entry_price': 120.00}
        }

        current_prices = {
            'AAPL': 140.00,  # Down 6.67% - should trigger
            'GOOGL': 121.00  # Up - should not trigger
        }

        symbols_to_close = risk_mgr.apply_stop_loss(positions, current_prices)

        assert 'AAPL' in symbols_to_close
        assert 'GOOGL' not in symbols_to_close

    def test_apply_stop_loss_not_triggered(self):
        """Test stop loss when not triggered."""
        risk_mgr = RiskManager(stop_loss_pct=0.05)

        positions = {
            'AAPL': {'shares': 100, 'entry_price': 150.00}
        }

        current_prices = {
            'AAPL': 148.00  # Down 1.33% - should not trigger
        }

        symbols_to_close = risk_mgr.apply_stop_loss(positions, current_prices)

        assert len(symbols_to_close) == 0

    def test_apply_constraints(self, sample_portfolio):
        """Test applying risk constraints."""
        risk_mgr = RiskManager(max_position_size=0.2)

        proposed = {
            'AAPL': 0.50,  # Should be capped
            'GOOGL': 0.30,  # Should be capped
            'MSFT': 0.20   # OK
        }

        adjusted = risk_mgr.apply_constraints(proposed)

        assert isinstance(adjusted, dict)
        # All positions should be <= max_position_size
        assert all(w <= risk_mgr.max_position_size for w in adjusted.values())

        # Weights should still sum to approximately 1
        assert 0.95 <= sum(adjusted.values()) <= 1.05

    def test_apply_constraints_renormalization(self):
        """Test constraint application renormalizes weights."""
        risk_mgr = RiskManager(max_position_size=0.3)

        proposed = {
            'AAPL': 0.25,
            'GOOGL': 0.25,
            'MSFT': 0.25,
            'TSLA': 0.25
        }

        adjusted = risk_mgr.apply_constraints(proposed)

        # Should sum to 1.0
        assert abs(sum(adjusted.values()) - 1.0) < 0.01
