"""
Portfolio Optimization and Risk Management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Portfolio optimizer using Modern Portfolio Theory.
    Supports multiple optimization objectives.
    """

    def __init__(self, optimization_method: str = "max_sharpe"):
        """
        Initialize portfolio optimizer.

        Args:
            optimization_method: Optimization method
                - max_sharpe: Maximize Sharpe ratio
                - min_volatility: Minimize volatility
                - efficient_risk: Target specific risk level
                - risk_parity: Equal risk contribution
        """
        self.method = optimization_method
        logger.info(f"Initialized PortfolioOptimizer with method: {optimization_method}")

    def optimize_allocation(
        self,
        price_data: pd.DataFrame,
        predictions: Optional[Dict[str, float]] = None,
        risk_tolerance: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights.

        Args:
            price_data: DataFrame with price data (columns = symbols)
            predictions: Optional price predictions to incorporate
            risk_tolerance: Risk tolerance (0-2, default 1.0)
            constraints: Additional constraints

        Returns:
            Dictionary mapping symbols to weights
        """
        if len(price_data.columns) < 2:
            logger.warning("Need at least 2 assets for optimization")
            # Equal weight for single asset
            return {price_data.columns[0]: 1.0}

        try:
            # Calculate expected returns
            if predictions:
                # Use predictions for expected returns
                mu = pd.Series(predictions)
            else:
                # Use historical mean returns
                mu = expected_returns.mean_historical_return(price_data, frequency=252)

            # Calculate covariance matrix
            S = risk_models.sample_cov(price_data, frequency=252)

            # Create efficient frontier
            ef = EfficientFrontier(mu, S)

            # Add constraints
            if constraints:
                if 'max_weight' in constraints:
                    ef.add_constraint(lambda w: w <= constraints['max_weight'])
                if 'min_weight' in constraints:
                    ef.add_constraint(lambda w: w >= constraints['min_weight'])

            # Optimize based on method
            if self.method == "max_sharpe":
                weights = ef.max_sharpe(risk_free_rate=0.02)
            elif self.method == "min_volatility":
                weights = ef.min_volatility()
            elif self.method == "efficient_risk":
                target_volatility = 0.15 * risk_tolerance
                weights = ef.efficient_risk(target_volatility=target_volatility)
            elif self.method == "risk_parity":
                weights = self._risk_parity_optimization(price_data)
                return weights
            else:
                logger.error(f"Unknown optimization method: {self.method}")
                return self._equal_weight(price_data.columns)

            # Clean weights (remove negligible positions)
            cleaned_weights = ef.clean_weights(cutoff=0.01)

            logger.info(f"Optimized portfolio: {cleaned_weights}")
            return cleaned_weights

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._equal_weight(price_data.columns)

    def _risk_parity_optimization(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Risk parity portfolio optimization.
        Equal risk contribution from each asset.
        """
        # Calculate covariance matrix
        cov_matrix = price_data.pct_change().dropna().cov()

        n_assets = len(price_data.columns)

        # Objective function: minimize difference in risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Target is equal risk contribution
            target = portfolio_vol / n_assets
            return np.sum((risk_contrib - target) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess (equal weight)
        w0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = dict(zip(price_data.columns, result.x))
            # Clean small weights
            weights = {k: v for k, v in weights.items() if v > 0.01}
            # Renormalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            return weights
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            return self._equal_weight(price_data.columns)

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Equal weight portfolio."""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    def calculate_metrics(
        self,
        weights: Dict[str, float],
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio metrics.

        Args:
            weights: Portfolio weights
            price_data: Historical price data

        Returns:
            Dictionary of metrics
        """
        # Filter price data for assets in portfolio
        portfolio_prices = price_data[[col for col in price_data.columns if col in weights]]

        # Calculate returns
        returns = portfolio_prices.pct_change().dropna()

        # Portfolio returns
        weights_array = np.array([weights.get(col, 0) for col in portfolio_prices.columns])
        portfolio_returns = returns @ weights_array

        # Calculate metrics
        mean_return = portfolio_returns.mean() * 252  # Annualized
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (mean_return - 0.02) / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (mean_return - 0.02) / downside_std if downside_std > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics = {
            'expected_return': float(mean_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
        }

        logger.info(f"Portfolio metrics - Return: {mean_return:.2%}, "
                   f"Sharpe: {sharpe_ratio:.2f}, Drawdown: {max_drawdown:.2%}")

        return metrics

    def rebalance_check(
        self,
        current_weights: Dict[str, float],
        optimal_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current portfolio weights
            optimal_weights: Optimal weights
            threshold: Rebalancing threshold

        Returns:
            Tuple of (needs_rebalancing, trade_instructions)
        """
        # Get all symbols
        all_symbols = set(list(current_weights.keys()) + list(optimal_weights.keys()))

        max_deviation = 0
        trade_instructions = {}

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            optimal = optimal_weights.get(symbol, 0)
            deviation = abs(optimal - current)

            if deviation > max_deviation:
                max_deviation = deviation

            if deviation > threshold:
                trade_instructions[symbol] = optimal - current

        needs_rebalancing = max_deviation > threshold

        if needs_rebalancing:
            logger.info(f"Rebalancing needed - Max deviation: {max_deviation:.2%}")
        else:
            logger.info("Portfolio is balanced, no rebalancing needed")

        return needs_rebalancing, trade_instructions


class RiskManager:
    """
    Risk management system for trading.
    Enforces position limits, stop losses, and risk metrics.
    """

    def __init__(
        self,
        max_position_size: float = 0.2,
        max_portfolio_var: float = 0.05,
        stop_loss_pct: float = 0.05,
        max_correlation: float = 0.7
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_var: Maximum portfolio Value at Risk
            stop_loss_pct: Stop loss percentage
            max_correlation: Maximum correlation between positions
        """
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        self.stop_loss_pct = stop_loss_pct
        self.max_correlation = max_correlation

        logger.info(f"Initialized RiskManager with max position: {max_position_size:.1%}")

    def check_position_size(
        self,
        symbol: str,
        proposed_size: float,
        portfolio_value: float
    ) -> float:
        """
        Enforce position size limits.

        Args:
            symbol: Symbol to check
            proposed_size: Proposed position size (dollars)
            portfolio_value: Total portfolio value

        Returns:
            Adjusted position size
        """
        max_size = portfolio_value * self.max_position_size

        if proposed_size > max_size:
            logger.warning(f"Position size for {symbol} exceeds limit: "
                         f"${proposed_size:.2f} > ${max_size:.2f}")
            return max_size

        return proposed_size

    def calculate_var(
        self,
        portfolio: Dict[str, float],
        price_data: pd.DataFrame,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate portfolio Value at Risk.

        Args:
            portfolio: Portfolio weights
            price_data: Historical price data
            confidence: Confidence level (default 95%)

        Returns:
            VaR as fraction of portfolio
        """
        # Filter for portfolio assets
        portfolio_prices = price_data[[col for col in price_data.columns if col in portfolio]]

        # Calculate returns
        returns = portfolio_prices.pct_change().dropna()

        # Portfolio returns
        weights = np.array([portfolio.get(col, 0) for col in portfolio_prices.columns])
        portfolio_returns = returns @ weights

        # Calculate VaR using historical simulation
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)

        logger.info(f"Portfolio VaR ({confidence:.0%}): {var:.2%}")
        return float(var)

    def calculate_cvar(
        self,
        portfolio: Dict[str, float],
        price_data: pd.DataFrame,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            portfolio: Portfolio weights
            price_data: Historical price data
            confidence: Confidence level

        Returns:
            CVaR as fraction of portfolio
        """
        # Filter for portfolio assets
        portfolio_prices = price_data[[col for col in price_data.columns if col in portfolio]]

        # Calculate returns
        returns = portfolio_prices.pct_change().dropna()

        # Portfolio returns
        weights = np.array([portfolio.get(col, 0) for col in portfolio_prices.columns])
        portfolio_returns = returns @ weights

        # Calculate CVaR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        logger.info(f"Portfolio CVaR ({confidence:.0%}): {cvar:.2%}")
        return float(cvar)

    def check_correlation(
        self,
        portfolio: Dict[str, float],
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Check correlations between portfolio positions.

        Args:
            portfolio: Portfolio weights
            price_data: Historical price data

        Returns:
            Dictionary with correlation analysis
        """
        # Filter for portfolio assets
        portfolio_symbols = list(portfolio.keys())
        portfolio_prices = price_data[portfolio_symbols]

        # Calculate correlation matrix
        corr_matrix = portfolio_prices.pct_change().dropna().corr()

        # Find high correlations
        high_corr_pairs = []
        for i in range(len(portfolio_symbols)):
            for j in range(i + 1, len(portfolio_symbols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > self.max_correlation:
                    high_corr_pairs.append({
                        'asset1': portfolio_symbols[i],
                        'asset2': portfolio_symbols[j],
                        'correlation': float(corr)
                    })

        diversified = len(high_corr_pairs) == 0

        if not diversified:
            logger.warning(f"High correlations detected: {len(high_corr_pairs)} pairs")

        return {
            'diversified': diversified,
            'high_correlations': high_corr_pairs,
            'correlation_matrix': corr_matrix.to_dict()
        }

    def apply_stop_loss(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> List[str]:
        """
        Identify positions that hit stop loss.

        Args:
            positions: Dictionary of positions with entry prices
                      {symbol: {'shares': X, 'entry_price': Y}}
            current_prices: Current prices for each symbol

        Returns:
            List of symbols to close
        """
        symbols_to_close = []

        for symbol, position in positions.items():
            if symbol not in current_prices:
                continue

            entry_price = position.get('entry_price', 0)
            current_price = current_prices[symbol]

            if entry_price == 0:
                continue

            # Calculate loss percentage
            loss_pct = (current_price - entry_price) / entry_price

            if loss_pct <= -self.stop_loss_pct:
                logger.warning(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
                symbols_to_close.append(symbol)

        return symbols_to_close

    def apply_constraints(
        self,
        proposed_allocation: Dict[str, float],
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Apply risk constraints to proposed allocation.

        Args:
            proposed_allocation: Proposed portfolio weights
            current_portfolio: Current portfolio (for turnover limits)

        Returns:
            Adjusted allocation
        """
        adjusted = proposed_allocation.copy()

        # Enforce max position size
        for symbol, weight in adjusted.items():
            if weight > self.max_position_size:
                logger.info(f"Capping {symbol} position from {weight:.2%} to {self.max_position_size:.2%}")
                adjusted[symbol] = self.max_position_size

        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted


if __name__ == "__main__":
    # Test portfolio optimization
    from src.data.technical_indicators import TechnicalIndicatorEngine

    # Fetch data for multiple symbols
    engine = TechnicalIndicatorEngine(['AAPL', 'GOOGL', 'MSFT'], ['1d'])

    price_data = pd.DataFrame()
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        df = engine.fetch_data(symbol, '1d', lookback=200)
        price_data[symbol] = df['close']

    # Optimize
    optimizer = PortfolioOptimizer(optimization_method='max_sharpe')
    weights = optimizer.optimize_allocation(price_data)
    print(f"Optimal Weights: {weights}")

    # Calculate metrics
    metrics = optimizer.calculate_metrics(weights, price_data)
    print(f"Portfolio Metrics: {metrics}")

    # Risk management
    risk_manager = RiskManager()
    var = risk_manager.calculate_var(weights, price_data)
    print(f"Portfolio VaR: {var:.2%}")

    corr_analysis = risk_manager.check_correlation(weights, price_data)
    print(f"Diversified: {corr_analysis['diversified']}")
