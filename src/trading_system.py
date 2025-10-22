"""
Agentic Trading System - Main Orchestrator
Integrates all components for autonomous trading.
"""

import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.data.technical_indicators import TechnicalIndicatorEngine
from src.data.news_sentiment import NewsSentimentAnalyzer
from src.models.price_predictor import PricePredictionEngine
from src.strategies.llm_strategy_generator import LLMStrategyGenerator, BaseStrategy
from src.models.rl_trainer import RLTrainingSystem
from src.execution.portfolio_optimizer import PortfolioOptimizer, RiskManager
from src.execution.broker import AlpacaBroker, OrderManager
from src.backtesting.backtest_engine import BacktestEngine, PerformanceAnalyzer
import pandas as pd


class AgenticTradingSystem:
    """
    Main orchestrator for the autonomous agentic trading system.
    Coordinates all components for data collection, strategy generation,
    prediction, optimization, and execution.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading system.

        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 80)
        logger.info("Initializing Agentic Trading System")
        logger.info("=" * 80)

        # Load configuration
        self.config = ConfigLoader(config_path)

        if not self.config.validate_config():
            raise ValueError("Invalid configuration")

        # Initialize components
        self._initialize_components()

        # State
        self.active_strategies = []
        self.portfolio = {}
        self.last_rebalance = None
        self.is_running = False

        mode = "PAPER TRADING" if self.config.is_paper_trading() else "LIVE TRADING"
        logger.info(f"System initialized in {mode} mode")

    def _initialize_components(self) -> None:
        """Initialize all system components."""
        logger.info("Initializing components...")

        # Data acquisition
        symbols = self.config.get_symbols()
        timeframes = self.config.get_timeframes()

        self.data_engine = TechnicalIndicatorEngine(symbols, timeframes)
        logger.info(f"✓ Data engine initialized ({len(symbols)} symbols)")

        # News and sentiment
        llm_api_key = self.config.get_api_key('anthropic')
        self.news_analyzer = NewsSentimentAnalyzer(
            use_llm=llm_api_key is not None,
            llm_client=None  # Can add LLM client here
        )
        logger.info("✓ News analyzer initialized")

        # Price prediction
        prediction_models = self.config.get('prediction.models', ['xgboost', 'prophet'])
        self.price_predictor = PricePredictionEngine(models=prediction_models)
        logger.info(f"✓ Price predictor initialized ({len(prediction_models)} models)")

        # LLM strategy generator
        if llm_api_key:
            llm_provider = self.config.get('strategies.llm.provider', 'anthropic')
            llm_model = self.config.get('strategies.llm.model', 'claude-sonnet-4-5')
            temperature = self.config.get('strategies.llm.temperature', 0.7)

            self.strategy_generator = LLMStrategyGenerator(
                api_key=llm_api_key,
                provider=llm_provider,
                model=llm_model,
                temperature=temperature
            )
            logger.info("✓ LLM strategy generator initialized")
        else:
            self.strategy_generator = None
            logger.warning("⚠ LLM strategy generator not initialized (no API key)")

        # RL system
        rl_algorithm = self.config.get('strategies.rl.algorithm', 'PPO')
        self.rl_system = RLTrainingSystem(algorithm=rl_algorithm)
        logger.info(f"✓ RL system initialized ({rl_algorithm})")

        # Portfolio optimization
        optimization_method = self.config.get('portfolio.optimization_method', 'max_sharpe')
        self.portfolio_optimizer = PortfolioOptimizer(optimization_method=optimization_method)
        logger.info(f"✓ Portfolio optimizer initialized ({optimization_method})")

        # Risk management
        max_position = self.config.get('portfolio.max_position_size', 0.2)
        max_var = self.config.get('risk.max_portfolio_var', 0.05)
        stop_loss = self.config.get('risk.stop_loss_pct', 0.05)

        self.risk_manager = RiskManager(
            max_position_size=max_position,
            max_portfolio_var=max_var,
            stop_loss_pct=stop_loss
        )
        logger.info("✓ Risk manager initialized")

        # Broker and order management
        broker_type = self.config.get('brokers.default', 'alpaca')

        if broker_type == 'alpaca':
            paper_mode = self.config.is_paper_trading()
            api_key = self.config.get_api_key('alpaca_paper_key' if paper_mode else 'alpaca_live_key')
            secret_key = self.config.get_api_key('alpaca_paper_secret' if paper_mode else 'alpaca_live_secret')

            if api_key and secret_key:
                self.broker = AlpacaBroker(api_key, secret_key, paper=paper_mode)
                self.order_manager = OrderManager(self.broker)

                if self.broker.connect():
                    logger.info("✓ Broker connected (Alpaca)")
                else:
                    logger.error("✗ Failed to connect to broker")
                    self.broker = None
                    self.order_manager = None
            else:
                logger.warning("⚠ Broker not initialized (no API keys)")
                self.broker = None
                self.order_manager = None
        else:
            logger.warning(f"⚠ Unknown broker type: {broker_type}")
            self.broker = None
            self.order_manager = None

        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer()
        logger.info("✓ Performance analyzer initialized")

        logger.info("All components initialized successfully\n")

    def generate_initial_strategies(self, n_strategies: int = 3) -> List[str]:
        """
        Generate initial trading strategies using LLM.

        Args:
            n_strategies: Number of strategies to generate

        Returns:
            List of strategy codes
        """
        if not self.strategy_generator:
            logger.error("Strategy generator not available")
            return []

        logger.info(f"Generating {n_strategies} initial strategies...")

        strategies = []
        symbols = self.config.get_symbols()

        for i in range(n_strategies):
            # Get market context
            symbol = symbols[i % len(symbols)]
            regime = self.data_engine.get_market_regime(symbol)

            context = {
                'symbol': symbol,
                'market_regime': regime.get('trend', 'unknown'),
                'trend': regime.get('trend', 'unknown'),
                'volatility': regime.get('volatility', 'medium'),
                'available_indicators': 'RSI, MACD, Bollinger Bands, SMA, EMA, ATR',
                'sentiment_score': 'neutral',
                'performance_summary': 'Initial strategy generation'
            }

            strategy_code = self.strategy_generator.generate_strategy(
                context,
                strategy_name=f"Strategy_{i+1}"
            )

            if strategy_code:
                strategies.append(strategy_code)
                logger.info(f"✓ Generated Strategy_{i+1}")

        logger.info(f"Generated {len(strategies)} strategies\n")
        return strategies

    def backtest_strategy(
        self,
        strategy_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Backtest a strategy on historical data.

        Args:
            strategy_code: Strategy code to test
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Backtest results
        """
        logger.info("Running backtest...")

        # Fetch historical data
        symbols = self.config.get_symbols()
        all_data = {}

        for symbol in symbols:
            df = self.data_engine.fetch_data(symbol, '1d', lookback=252)
            df = self.data_engine.calculate_indicators(df)
            all_data[symbol] = df

        # Create backtest engine
        initial_cash = self.config.get_initial_capital()
        commission = self.config.get('backtesting.commission', 0.001)

        backtest_engine = BacktestEngine(
            initial_cash=initial_cash,
            commission=commission
        )

        # TODO: Execute strategy and run backtest
        # This requires instantiating the strategy class from code string
        # For now, return placeholder results

        results = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0
        }

        logger.info("Backtest complete\n")
        return results

    def run(self) -> None:
        """
        Main execution loop.
        Continuously monitors markets and executes strategies.
        """
        logger.info("=" * 80)
        logger.info("Starting Trading System")
        logger.info("=" * 80)

        if not self.broker:
            logger.error("Cannot run without broker connection")
            return

        self.is_running = True
        execution_interval = self.config.get('system.execution_interval', 300)

        logger.info(f"Execution interval: {execution_interval}s")
        logger.info("Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while self.is_running:
                iteration += 1
                logger.info(f"--- Iteration {iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

                try:
                    self._execute_iteration()
                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {e}")
                    logger.exception(e)

                # Sleep until next iteration
                logger.info(f"Sleeping for {execution_interval}s...\n")
                time.sleep(execution_interval)

        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")
            self.stop()

    def _execute_iteration(self) -> None:
        """Execute one iteration of the trading loop."""

        # 1. Data Collection
        logger.info("1. Collecting market data...")
        market_data = self.data_engine.fetch_all_data()

        if not market_data:
            logger.warning("No market data available")
            return

        # Get news and sentiment
        news_api_key = self.config.get_api_key('newsapi')
        if news_api_key:
            symbols = self.config.get_symbols()
            articles = self.news_analyzer.fetch_news(symbols, api_key=news_api_key)
            sentiment_df = self.news_analyzer.analyze_sentiment(articles)
            sentiment = self.news_analyzer.aggregate_sentiment(sentiment_df)
        else:
            sentiment = {}

        logger.info(f"   Fetched data for {len(market_data)} symbol-timeframe combinations")

        # 2. Price Predictions
        logger.info("2. Generating price predictions...")
        predictions = {}

        for key, df in market_data.items():
            symbol = key.split('_')[0]

            if symbol not in predictions:
                try:
                    # Train predictor on this data
                    self.price_predictor.train_models(df)

                    # Generate prediction
                    pred = self.price_predictor.predict(symbol, df, horizon='1h')
                    predictions[symbol] = pred

                    logger.info(f"   {symbol}: ${pred['predicted_price']:.2f} "
                              f"(current: ${pred['current_price']:.2f})")
                except Exception as e:
                    logger.error(f"   Prediction failed for {symbol}: {e}")

        # 3. Generate Trading Signals
        logger.info("3. Generating trading signals...")
        signals = []

        # For now, use a simple signal based on predictions
        for symbol, pred in predictions.items():
            current_price = pred['current_price']
            predicted_price = pred['predicted_price']
            probability_up = pred['probability_up']

            # Simple signal logic
            if probability_up > 0.6 and predicted_price > current_price * 1.02:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': 0.1,  # 10% of buying power
                    'confidence': probability_up,
                    'reason': f'Price predicted to rise to ${predicted_price:.2f}'
                })
                logger.info(f"   BUY signal for {symbol} (confidence: {probability_up:.2f})")

        # 4. Portfolio Optimization
        if signals:
            logger.info("4. Optimizing portfolio allocation...")

            # Get current positions
            positions = self.broker.get_positions()

            # Build price data for optimization
            price_data = pd.DataFrame()
            for symbol in predictions.keys():
                key = f"{symbol}_1d"
                if key in market_data:
                    price_data[symbol] = market_data[key]['close']

            if len(price_data.columns) >= 2:
                try:
                    # Optimize allocation
                    optimal_weights = self.portfolio_optimizer.optimize_allocation(
                        price_data,
                        predictions={s: p['predicted_price'] for s, p in predictions.items()}
                    )

                    # Apply risk constraints
                    safe_weights = self.risk_manager.apply_constraints(optimal_weights)

                    logger.info(f"   Optimal allocation: {safe_weights}")

                    # Check if rebalancing needed
                    current_weights = self._get_current_weights(positions)
                    needs_rebalancing, trades = self.portfolio_optimizer.rebalance_check(
                        current_weights,
                        safe_weights,
                        threshold=0.05
                    )

                    if needs_rebalancing:
                        logger.info("   Rebalancing needed")
                        # Convert to signals
                        # signals = self._convert_weights_to_signals(trades)
                except Exception as e:
                    logger.error(f"   Portfolio optimization failed: {e}")
        else:
            logger.info("4. No signals to optimize")

        # 5. Risk Checks
        logger.info("5. Performing risk checks...")

        # Check stop losses
        if self.order_manager:
            positions = self.broker.get_positions()
            current_prices = {s: self.broker.get_current_price(s) for s in positions.keys()}

            positions_dict = {
                symbol: {
                    'shares': pos['qty'],
                    'entry_price': pos['avg_entry_price']
                }
                for symbol, pos in positions.items()
            }

            stop_loss_symbols = self.risk_manager.apply_stop_loss(positions_dict, current_prices)

            if stop_loss_symbols:
                logger.warning(f"   Stop loss triggered for: {stop_loss_symbols}")
                for symbol in stop_loss_symbols:
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': 1.0,
                        'reason': 'Stop loss triggered'
                    })

        # 6. Execute Trades
        if signals and self.order_manager:
            logger.info(f"6. Executing {len(signals)} signals...")

            order_ids = self.order_manager.execute_strategy_signals(signals)
            logger.info(f"   Placed {len(order_ids)} orders")

            # Monitor orders
            time.sleep(5)  # Wait a bit for fills
            self.order_manager.monitor_orders()
        else:
            logger.info("6. No signals to execute")

        # 7. Performance Tracking
        logger.info("7. Tracking performance...")

        if self.broker:
            account_info = self.broker.get_account_info()
            portfolio_value = account_info.get('portfolio_value', 0)
            equity = account_info.get('equity', 0)

            initial_capital = self.config.get_initial_capital()
            return_pct = ((portfolio_value - initial_capital) / initial_capital) * 100

            logger.info(f"   Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"   Return: {return_pct:+.2f}%")

    def _get_current_weights(self, positions: Dict) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        if not positions:
            return {}

        total_value = sum(pos['market_value'] for pos in positions.values())

        if total_value == 0:
            return {}

        weights = {
            symbol: pos['market_value'] / total_value
            for symbol, pos in positions.items()
        }

        return weights

    def stop(self) -> None:
        """Stop the trading system gracefully."""
        logger.info("Stopping trading system...")

        self.is_running = False

        # Cancel pending orders
        if self.order_manager:
            cancelled = self.order_manager.cancel_all_pending()
            logger.info(f"Cancelled {cancelled} pending orders")

        # Get final statistics
        if self.broker:
            account_info = self.broker.get_account_info()
            logger.info(f"Final Portfolio Value: ${account_info.get('portfolio_value', 0):.2f}")

        logger.info("Trading system stopped")

    def get_status(self) -> Dict:
        """
        Get current system status.

        Returns:
            Status dictionary
        """
        status = {
            'is_running': self.is_running,
            'mode': 'paper' if self.config.is_paper_trading() else 'live',
            'symbols': self.config.get_symbols(),
            'broker_connected': self.broker is not None and self.broker.connect(),
        }

        if self.broker:
            account_info = self.broker.get_account_info()
            status['account_info'] = account_info

            positions = self.broker.get_positions()
            status['num_positions'] = len(positions)
            status['positions'] = positions

        if self.order_manager:
            status['pending_orders'] = len(self.order_manager.get_pending_orders())
            status['filled_orders'] = len(self.order_manager.get_filled_orders())

        return status


if __name__ == "__main__":
    # Initialize system
    system = AgenticTradingSystem()

    # Get status
    status = system.get_status()
    print("\nSystem Status:")
    print(f"Mode: {status['mode']}")
    print(f"Broker Connected: {status['broker_connected']}")
    print(f"Symbols: {status['symbols']}")

    # Run system (uncomment to start trading)
    # system.run()
