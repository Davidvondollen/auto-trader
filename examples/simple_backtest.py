"""
Simple Backtest Example
Demonstrates how to backtest a basic trading strategy.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.technical_indicators import TechnicalIndicatorEngine
from src.backtesting.backtest_engine import BacktestEngine
import pandas as pd


def simple_moving_average_strategy(data, fast_period=20, slow_period=50):
    """
    Simple moving average crossover strategy.

    Args:
        data: Market data with indicators
        fast_period: Fast moving average period
        slow_period: Slow moving average period

    Returns:
        Trading signal
    """
    if len(data) < slow_period + 1:
        return {'action': 'hold'}

    # Calculate moving averages
    fast_ma = data['close'].rolling(fast_period).mean().iloc[-1]
    slow_ma = data['close'].rolling(slow_period).mean().iloc[-1]

    prev_fast_ma = data['close'].rolling(fast_period).mean().iloc[-2]
    prev_slow_ma = data['close'].rolling(slow_period).mean().iloc[-2]

    # Bullish crossover
    if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
        return {
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 0.5  # Use 50% of available capital
        }

    # Bearish crossover
    if fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
        return {
            'symbol': 'AAPL',
            'action': 'sell',
            'quantity': 1.0  # Sell entire position
        }

    return {'action': 'hold'}


def main():
    print("=" * 80)
    print("Simple Backtest Example")
    print("=" * 80)

    # Fetch historical data
    print("\n1. Fetching historical data...")
    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=252)
    df = engine.calculate_indicators(df)
    print(f"   Loaded {len(df)} days of data")

    # Prepare data for backtest
    data = df[['close']].copy()
    data.columns = ['AAPL']

    # Initialize backtest engine
    print("\n2. Initializing backtest engine...")
    backtest = BacktestEngine(
        initial_cash=10000,
        commission=0.001,
        slippage=0.0005
    )

    # Run backtest
    print("\n3. Running backtest...")
    results = backtest.run(
        data=data,
        strategy_func=simple_moving_average_strategy,
        strategy_params={'fast_period': 20, 'slow_period': 50}
    )

    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Capital:    ${results['initial_value']:,.2f}")
    print(f"Final Value:        ${results['final_value']:,.2f}")
    print(f"Total Return:       {results['total_return']:.2%}")
    print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:      {results['sortino_ratio']:.2f}")
    print(f"Max Drawdown:       {results['max_drawdown']:.2%}")
    print(f"Number of Trades:   {results['num_trades']}")
    print(f"Win Rate:           {results['win_rate']:.2%}")
    print(f"Profit Factor:      {results['profit_factor']:.2f}")
    print("=" * 80)

    # Plot results (optional - requires display)
    # backtest.plot_results(results)


if __name__ == "__main__":
    main()
