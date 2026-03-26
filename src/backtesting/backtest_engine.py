"""
Backtesting Engine and Performance Analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy evaluation.
    """

    def __init__(
        self,
        initial_cash: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ):
        """
        Initialize backtest engine.

        Args:
            initial_cash: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 2%)
            annualization_factor: Trading periods per year (252 for daily, 252*6.5 for hourly)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

        # State
        self.cash = initial_cash
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
        self.current_step = 0

        logger.info(f"Initialized BacktestEngine with ${initial_cash:.2f}")

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run backtest.

        Args:
            data: Historical market data
            strategy_func: Strategy function that generates signals
            strategy_params: Parameters for strategy

        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")

        # Reset state
        self.cash = self.initial_cash
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
        self.current_step = 0

        # Run through historical data
        for i in range(len(data)):
            self.current_step = i

            # Get current market state
            current_data = data.iloc[:i+1]

            if len(current_data) < 20:  # Need minimum history
                continue

            # Generate signal from strategy
            try:
                signal = strategy_func(current_data, **(strategy_params or {}))
            except Exception as e:
                logger.error(f"Strategy error at step {i}: {e}")
                signal = {'action': 'hold'}

            # Execute signal
            if signal and signal.get('action') != 'hold':
                self._execute_trade(signal, data.iloc[i])

            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(data.iloc[i])
            self.portfolio_values.append({
                'timestamp': data.index[i] if hasattr(data, 'index') else i,
                'value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })

        # Generate results
        results = self._generate_results(data)

        logger.info(f"Backtest complete - Final value: ${results['final_value']:.2f}, "
                   f"Return: {results['total_return']:.2%}")

        return results

    def _execute_trade(self, signal: Dict, current_prices: pd.Series) -> None:
        """Execute a trade based on signal."""
        symbol = signal.get('symbol')
        action = signal.get('action')
        quantity = signal.get('quantity', 1.0)

        if not symbol or symbol not in current_prices.index:
            return

        price = current_prices[symbol]

        if action == 'buy':
            # Apply slippage
            effective_price = price * (1 + self.slippage)

            # Calculate shares to buy
            max_spend = self.cash * quantity
            shares = int(max_spend / (effective_price * (1 + self.commission)))

            if shares > 0:
                cost = shares * effective_price * (1 + self.commission)

                if cost <= self.cash:
                    self.cash -= cost

                    if symbol in self.positions:
                        # Update average cost
                        old_shares = self.positions[symbol]['shares']
                        old_cost = self.positions[symbol]['avg_price']
                        new_avg = ((old_shares * old_cost) + (shares * effective_price)) / (old_shares + shares)

                        self.positions[symbol]['shares'] += shares
                        self.positions[symbol]['avg_price'] = new_avg
                    else:
                        self.positions[symbol] = {
                            'shares': shares,
                            'avg_price': effective_price
                        }

                    self.trades.append({
                        'step': self.current_step,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': effective_price,
                        'cost': cost
                    })

        elif action == 'sell':
            if symbol not in self.positions:
                return

            # Apply slippage
            effective_price = price * (1 - self.slippage)

            # Sell shares
            shares_held = self.positions[symbol]['shares']
            shares_to_sell = int(shares_held * quantity)

            if shares_to_sell > 0:
                proceeds = shares_to_sell * effective_price * (1 - self.commission)
                self.cash += proceeds

                # Calculate profit
                avg_cost = self.positions[symbol]['avg_price']
                profit = shares_to_sell * (effective_price - avg_cost)

                # Update position
                self.positions[symbol]['shares'] -= shares_to_sell

                if self.positions[symbol]['shares'] <= 0:
                    del self.positions[symbol]

                self.trades.append({
                    'step': self.current_step,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': effective_price,
                    'proceeds': proceeds,
                    'profit': profit
                })

    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """Calculate current portfolio value."""
        positions_value = 0

        for symbol, position in self.positions.items():
            if symbol in current_prices.index:
                positions_value += position['shares'] * current_prices[symbol]

        return self.cash + positions_value

    def _generate_results(self, data: pd.DataFrame) -> Dict:
        """Generate backtest results and metrics."""
        if not self.portfolio_values:
            return {'error': 'No portfolio history'}

        portfolio_df = pd.DataFrame(self.portfolio_values)
        final_value = portfolio_df['value'].iloc[-1]

        # Calculate returns
        portfolio_df['returns'] = portfolio_df['value'].pct_change()

        # Calculate metrics
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # Sharpe ratio (annualized, with risk-free rate)
        returns = portfolio_df['returns'].dropna()
        rf_per_period = self.risk_free_rate / self.annualization_factor
        excess_returns = returns - rf_per_period
        sharpe_ratio = np.sqrt(self.annualization_factor) * excess_returns.mean() / (returns.std() + 1e-8)

        # Sortino ratio (annualized, with risk-free rate)
        downside_returns = returns[returns < rf_per_period]
        sortino_ratio = np.sqrt(self.annualization_factor) * excess_returns.mean() / (downside_returns.std() + 1e-8)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        trades_df = pd.DataFrame(self.trades)

        if not trades_df.empty and 'profit' in trades_df.columns:
            profit_trades = trades_df[trades_df['profit'] > 0]
            win_rate = len(profit_trades) / len(trades_df[trades_df['action'] == 'sell'])

            avg_profit = profit_trades['profit'].mean() if len(profit_trades) > 0 else 0
            loss_trades = trades_df[trades_df['profit'] < 0]
            avg_loss = loss_trades['profit'].mean() if len(loss_trades) > 0 else 0

            profit_factor = (profit_trades['profit'].sum() /
                           abs(loss_trades['profit'].sum())) if len(loss_trades) > 0 else float('inf')
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0

        results = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'portfolio_history': portfolio_df,
            'trades_history': trades_df
        }

        return results

    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Optional[Dict] = None,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 21,
        retrain: bool = True
    ) -> Dict:
        """
        Perform walk-forward analysis for out-of-sample validation.

        This method trains on a rolling window and tests on the next period,
        providing realistic out-of-sample performance estimates.

        Args:
            data: Historical market data
            strategy_func: Strategy function that can be trained
            strategy_params: Base parameters for strategy
            train_period: Number of days for training window (default: 252 = 1 year)
            test_period: Number of days for testing window (default: 63 = 3 months)
            step_size: Days to step forward each iteration (default: 21 = 1 month)
            retrain: Whether to retrain on each window

        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Starting walk-forward analysis: train={train_period}, "
                   f"test={test_period}, step={step_size}")

        all_results = []
        window_results = []

        # Calculate total windows
        total_length = len(data)
        start_idx = 0
        window_num = 0

        while start_idx + train_period + test_period <= total_length:
            window_num += 1

            # Define train and test periods
            train_start = start_idx
            train_end = start_idx + train_period
            test_start = train_end
            test_end = min(test_start + test_period, total_length)

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            logger.info(f"\nWindow {window_num}: "
                       f"Train[{train_start}:{train_end}] "
                       f"Test[{test_start}:{test_end}]")

            # Train strategy on training data if needed
            trained_params = strategy_params.copy() if strategy_params else {}
            if retrain and hasattr(strategy_func, 'train'):
                try:
                    trained_params = strategy_func.train(train_data, trained_params)
                except Exception as e:
                    logger.warning(f"Training failed for window {window_num}: {e}")

            # Test on out-of-sample data
            # Reset state for this test window
            self.cash = self.initial_cash
            self.positions = {}
            self.portfolio_values = []
            self.trades = []
            self.current_step = 0

            # Run backtest on test period
            for i in range(len(test_data)):
                self.current_step = i

                # Get historical data up to current point
                # Include all train data + test data up to now
                historical_data = pd.concat([
                    train_data,
                    test_data.iloc[:i+1]
                ])

                if len(historical_data) < 20:
                    continue

                # Generate signal
                try:
                    signal = strategy_func(historical_data, **trained_params)
                except Exception as e:
                    logger.error(f"Strategy error at step {i}: {e}")
                    signal = {'action': 'hold'}

                # Execute signal
                if signal and signal.get('action') != 'hold':
                    self._execute_trade(signal, test_data.iloc[i])

                # Calculate portfolio value
                portfolio_value = self._calculate_portfolio_value(test_data.iloc[i])
                self.portfolio_values.append({
                    'timestamp': test_data.index[i] if hasattr(test_data, 'index') else i,
                    'value': portfolio_value,
                    'cash': self.cash,
                    'positions_value': portfolio_value - self.cash
                })

            # Calculate window metrics
            window_result = self._generate_results(test_data)
            window_result['window_num'] = window_num
            window_result['train_start'] = train_start
            window_result['train_end'] = train_end
            window_result['test_start'] = test_start
            window_result['test_end'] = test_end

            window_results.append(window_result)

            logger.info(f"Window {window_num} Results: "
                       f"Return={window_result['total_return']:.2%}, "
                       f"Sharpe={window_result['sharpe_ratio']:.2f}")

            # Move to next window
            start_idx += step_size

        # Aggregate results across all windows
        returns = [r['total_return'] for r in window_results]
        sharpes = [r['sharpe_ratio'] for r in window_results]
        drawdowns = [r['max_drawdown'] for r in window_results]

        aggregate_results = {
            'num_windows': len(window_results),
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'avg_sharpe': float(np.mean(sharpes)),
            'avg_drawdown': float(np.mean(drawdowns)),
            'win_rate_windows': float(sum(1 for r in returns if r > 0) / len(returns)),
            'best_return': float(max(returns)),
            'worst_return': float(min(returns)),
            'window_results': window_results,
            'returns_by_window': returns,
            'sharpes_by_window': sharpes
        }

        logger.info(f"\n{'='*60}")
        logger.info("Walk-Forward Analysis Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Windows: {aggregate_results['num_windows']}")
        logger.info(f"Avg Return: {aggregate_results['avg_return']:.2%} ± {aggregate_results['std_return']:.2%}")
        logger.info(f"Avg Sharpe: {aggregate_results['avg_sharpe']:.2f}")
        logger.info(f"Win Rate: {aggregate_results['win_rate_windows']:.1%}")
        logger.info(f"{'='*60}")

        return aggregate_results

    def calculate_market_impact(
        self,
        order_size: float,
        avg_daily_volume: float,
        volatility: float = 0.02
    ) -> float:
        """
        Calculate market impact using square-root model.

        Based on the Almgren-Chriss market impact model.

        Args:
            order_size: Size of order in shares
            avg_daily_volume: Average daily volume
            volatility: Daily volatility (default 2%)

        Returns:
            Market impact as percentage of price
        """
        if avg_daily_volume == 0:
            return 0.0

        # Participation rate (what fraction of daily volume)
        participation_rate = order_size / avg_daily_volume

        # Square-root impact model
        # impact = volatility * sqrt(participation_rate)
        impact = volatility * np.sqrt(participation_rate)

        # Cap impact at reasonable level
        impact = min(impact, 0.1)  # Max 10% impact

        return impact

    def _execute_trade_with_market_impact(
        self,
        signal: Dict,
        current_prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> None:
        """
        Execute trade with realistic market impact modeling.

        Args:
            signal: Trade signal
            current_prices: Current prices
            volumes: Current volumes (for impact calculation)
        """
        symbol = signal.get('symbol')
        action = signal.get('action')
        quantity = signal.get('quantity', 1.0)

        if not symbol or symbol not in current_prices.index:
            return

        price = current_prices[symbol]

        if action == 'buy':
            # Calculate order size
            max_spend = self.cash * quantity
            # Estimate shares before calculating impact
            estimated_shares = max_spend / (price * (1 + self.commission))

            # Calculate market impact if volume data available
            if volumes is not None and symbol in volumes.index:
                avg_volume = volumes[symbol]
                market_impact = self.calculate_market_impact(
                    estimated_shares,
                    avg_volume
                )
            else:
                market_impact = 0.0

            # Total slippage = base slippage + market impact
            total_slippage = self.slippage + market_impact
            effective_price = price * (1 + total_slippage)

            # Recalculate shares with impact
            shares = int(max_spend / (effective_price * (1 + self.commission)))

            if shares > 0:
                cost = shares * effective_price * (1 + self.commission)

                if cost <= self.cash:
                    self.cash -= cost

                    if symbol in self.positions:
                        old_shares = self.positions[symbol]['shares']
                        old_cost = self.positions[symbol]['avg_price']
                        new_avg = ((old_shares * old_cost) + (shares * effective_price)) / (old_shares + shares)

                        self.positions[symbol]['shares'] += shares
                        self.positions[symbol]['avg_price'] = new_avg
                    else:
                        self.positions[symbol] = {
                            'shares': shares,
                            'avg_price': effective_price
                        }

                    self.trades.append({
                        'step': self.current_step,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': effective_price,
                        'cost': cost,
                        'market_impact': market_impact
                    })

        elif action == 'sell':
            if symbol not in self.positions:
                return

            shares_held = self.positions[symbol]['shares']
            shares_to_sell = int(shares_held * quantity)

            # Calculate market impact
            if volumes is not None and symbol in volumes.index:
                avg_volume = volumes[symbol]
                market_impact = self.calculate_market_impact(
                    shares_to_sell,
                    avg_volume
                )
            else:
                market_impact = 0.0

            # Total slippage
            total_slippage = self.slippage + market_impact
            effective_price = price * (1 - total_slippage)

            if shares_to_sell > 0:
                proceeds = shares_to_sell * effective_price * (1 - self.commission)
                self.cash += proceeds

                avg_cost = self.positions[symbol]['avg_price']
                profit = shares_to_sell * (effective_price - avg_cost)

                self.positions[symbol]['shares'] -= shares_to_sell

                if self.positions[symbol]['shares'] <= 0:
                    del self.positions[symbol]

                self.trades.append({
                    'step': self.current_step,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': effective_price,
                    'proceeds': proceeds,
                    'profit': profit,
                    'market_impact': market_impact
                })

    def plot_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.

        Args:
            results: Backtest results dictionary
            save_path: Optional path to save plot
        """
        portfolio_df = results['portfolio_history']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value
        axes[0, 0].plot(portfolio_df['timestamp'], portfolio_df['value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value ($)')
        axes[0, 0].grid(True)

        # Drawdown
        returns = portfolio_df['returns'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)

        # Returns distribution
        axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)

        # Trade analysis
        trades_df = results['trades_history']
        if not trades_df.empty and 'profit' in trades_df.columns:
            profits = trades_df['profit'].dropna()
            axes[1, 1].bar(range(len(profits)), profits,
                          color=['green' if x > 0 else 'red' for x in profits])
            axes[1, 1].set_title('Trade Profits')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Profit ($)')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")

        plt.show()


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    """

    def __init__(self):
        logger.info("Initialized PerformanceAnalyzer")

    def calculate_metrics(
        self,
        portfolio_values: pd.Series,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio_values: Time series of portfolio values
            trades: DataFrame of trades
            benchmark: Optional benchmark returns for comparison

        Returns:
            Dictionary of metrics
        """
        # Returns
        returns = portfolio_values.pct_change().dropna()

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - 0.02) / downside_deviation if downside_deviation > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade metrics
        if not trades.empty:
            num_trades = len(trades)

            # Win rate
            if 'profit' in trades.columns:
                winning_trades = trades[trades['profit'] > 0]
                win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

                avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
                losing_trades = trades[trades['profit'] < 0]
                avg_loss = abs(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 0

                profit_factor = (winning_trades['profit'].sum() /
                               abs(losing_trades['profit'].sum())) if len(losing_trades) > 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            num_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Benchmark comparison
        if benchmark is not None:
            benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
            alpha = total_return - benchmark_return

            # Beta
            covariance = returns.cov(benchmark.pct_change())
            benchmark_variance = benchmark.pct_change().var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        else:
            alpha = 0
            beta = 0

        metrics = {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'alpha': float(alpha),
            'beta': float(beta)
        }

        return metrics

    def compare_strategies(self, results_list: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            results_list: List of backtest results

        Returns:
            Comparison DataFrame
        """
        comparison = []

        for result in results_list:
            strategy_name = result.get('strategy_name', 'Unknown')

            comparison.append({
                'Strategy': strategy_name,
                'Total Return': result.get('total_return', 0),
                'Sharpe Ratio': result.get('sharpe_ratio', 0),
                'Max Drawdown': result.get('max_drawdown', 0),
                'Win Rate': result.get('win_rate', 0),
                'Num Trades': result.get('num_trades', 0)
            })

        df = pd.DataFrame(comparison)
        return df.sort_values('Sharpe Ratio', ascending=False)


if __name__ == "__main__":
    # Test backtesting
    from src.data.technical_indicators import TechnicalIndicatorEngine

    # Fetch data
    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=252)
    df = engine.calculate_indicators(df)

    # Simple momentum strategy
    def momentum_strategy(data, lookback=20):
        if len(data) < lookback + 1:
            return {'action': 'hold'}

        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-lookback]

        momentum = (current_price - past_price) / past_price

        if momentum > 0.05:  # 5% gain
            return {
                'symbol': 'AAPL',
                'action': 'buy',
                'quantity': 0.1
            }
        elif momentum < -0.05:  # 5% loss
            return {
                'symbol': 'AAPL',
                'action': 'sell',
                'quantity': 1.0
            }

        return {'action': 'hold'}

    # Run backtest
    engine_bt = BacktestEngine(initial_cash=10000)

    # Prepare data
    data_bt = df[['close']].copy()
    data_bt.columns = ['AAPL']

    results = engine_bt.run(data_bt, momentum_strategy, {'lookback': 20})

    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")

    # Plot results
    # engine_bt.plot_results(results)
