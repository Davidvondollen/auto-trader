# Agentic Trading System

An autonomous trading system that leverages Large Language Models (LLMs) and Reinforcement Learning (RL) to evolve and optimize trading strategies across multiple asset classes.

## Features

- **Autonomous Strategy Generation**: Uses Claude/GPT to generate and refine trading strategies
- **Adaptive Learning**: Employs RL (PPO, SAC, TD3) to optimize parameters
- **Multi-Asset Support**: Trade equities, cryptocurrencies, forex, and more
- **Portfolio Optimization**: Modern Portfolio Theory with risk management
- **Safety First**: Defaults to paper trading with explicit opt-in for live trading
- **Comprehensive Backtesting**: Test strategies before deployment
- **Real-time Monitoring**: Track performance and risk metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Management Layer                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ LLM Strategy│  │ RL Optimizer │  │ Strategy         │  │
│  │ Generator   │  │              │  │ Evaluator        │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                    Decision Engine Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Price        │  │ Signal       │  │ Portfolio       │  │
│  │ Predictor    │  │ Generator    │  │ Optimizer       │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-trader.git
cd auto-trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```
# Alpaca API Keys (for paper trading)
ALPACA_PAPER_KEY=your_paper_key_here
ALPACA_PAPER_SECRET=your_paper_secret_here

# Anthropic API Key (for LLM strategy generation)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: News API Key
NEWS_API_KEY=your_newsapi_key_here
```

Edit `config/config.yaml` to customize:

```yaml
system:
  mode: paper  # ALWAYS start with paper trading!
  initial_capital: 100000
  execution_interval: 300

assets:
  symbols:
    - AAPL
    - GOOGL
    - MSFT
```

### 3. Run Examples

#### Simple Backtest

```bash
python examples/simple_backtest.py
```

Output:
```
================================================================================
BACKTEST RESULTS
================================================================================
Initial Capital:    $10,000.00
Final Value:        $12,345.67
Total Return:       23.46%
Sharpe Ratio:       1.85
Max Drawdown:       -8.23%
Number of Trades:   42
Win Rate:           58.33%
================================================================================
```

#### Generate Trading Strategy with LLM

```bash
python examples/generate_strategy.py
```

This will:
1. Analyze current market conditions
2. Generate a custom trading strategy using Claude
3. Explain the strategy in plain English
4. Save the strategy code for backtesting

### 4. Run the Trading System

```python
from src.trading_system import AgenticTradingSystem

# Initialize system (defaults to paper trading)
system = AgenticTradingSystem()

# Check status
status = system.get_status()
print(f"Mode: {status['mode']}")  # Should be 'paper'
print(f"Broker Connected: {status['broker_connected']}")

# Run the system (will execute continuously)
system.run()
```

Press `Ctrl+C` to stop the system gracefully.

### 5. Launch Web Dashboard (NEW in Phase 2)

```bash
# Start the Streamlit dashboard
python run_dashboard.py

# Or directly with Streamlit
streamlit run src/visualization/dashboard.py
```

The dashboard will open at `http://localhost:8501` and provides:
- Real-time portfolio monitoring
- Performance analytics with charts
- Strategy management interface
- News & sentiment analysis
- Risk monitoring tools

## Project Structure

```
auto-trader/
├── src/
│   ├── data/
│   │   ├── technical_indicators.py  # Data acquisition and indicators
│   │   └── news_sentiment.py        # News and sentiment analysis
│   ├── models/
│   │   ├── price_predictor.py       # Ensemble price prediction
│   │   ├── rl_environment.py        # RL trading environment
│   │   └── rl_trainer.py            # RL training system
│   ├── strategies/
│   │   └── llm_strategy_generator.py # LLM-based strategy generation
│   ├── execution/
│   │   ├── broker.py                # Broker integrations
│   │   ├── portfolio_optimizer.py   # Portfolio optimization & risk
│   │   └── safety.py                # Live trading safety monitor (NEW)
│   ├── backtesting/
│   │   └── backtest_engine.py       # Backtesting framework
│   ├── visualization/
│   │   └── dashboard.py             # Streamlit web dashboard (NEW)
│   ├── utils/
│   │   └── config_loader.py         # Configuration management
│   └── trading_system.py            # Main orchestrator
├── config/
│   └── config.yaml                  # System configuration
├── examples/
│   ├── simple_backtest.py           # Backtest example
│   └── generate_strategy.py         # Strategy generation example
├── tests/                           # Unit tests
├── run_dashboard.py                 # Dashboard launcher (NEW)
├── PHASE_2_SUMMARY.md              # Phase 2 documentation (NEW)
├── .env.example                     # Environment variables template
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Core Components

### 1. Data Acquisition

**Technical Indicators Engine**
- Fetches OHLCV data from Yahoo Finance, CCXT, Alpha Vantage
- Calculates 20+ technical indicators
- Identifies market regimes (trending, ranging, volatile)

```python
from src.data.technical_indicators import TechnicalIndicatorEngine

engine = TechnicalIndicatorEngine(['AAPL', 'GOOGL'], ['1h', '1d'])
df = engine.fetch_data('AAPL', '1d', lookback=200)
df = engine.calculate_indicators(df, ['rsi', 'macd', 'bbands'])

regime = engine.get_market_regime('AAPL')
print(regime)  # {'trend': 'uptrend', 'volatility': 'medium', ...}
```

**News & Sentiment Analysis**
- Aggregates news from multiple sources
- FinBERT sentiment analysis
- Event extraction using LLMs

```python
from src.data.news_sentiment import NewsSentimentAnalyzer

analyzer = NewsSentimentAnalyzer()
articles = analyzer.fetch_news(['AAPL'], api_key=news_api_key)
sentiment = analyzer.analyze_sentiment(articles)
aggregated = analyzer.aggregate_sentiment(sentiment)
```

### 2. Price Prediction

Ensemble of multiple models:
- **XGBoost**: Gradient boosting on features
- **Prophet**: Time series decomposition
- **LSTM**: Deep learning (optional)

```python
from src.models.price_predictor import PricePredictionEngine

predictor = PricePredictionEngine(models=['xgboost', 'prophet'])
predictor.train_models(market_data)

prediction = predictor.predict('AAPL', market_data, horizon='1h')
print(f"Predicted: ${prediction['predicted_price']:.2f}")
print(f"Confidence: {prediction['confidence_interval']}")
```

### 3. LLM Strategy Generation

Generate trading strategies using Claude or GPT:

```python
from src.strategies.llm_strategy_generator import LLMStrategyGenerator

generator = LLMStrategyGenerator(api_key=anthropic_key)

context = {
    'symbol': 'AAPL',
    'market_regime': 'trending',
    'volatility': 'medium',
    'available_indicators': 'RSI, MACD, BB'
}

strategy_code = generator.generate_strategy(context)
explanation = generator.explain_strategy(strategy_code)
```

### 4. Reinforcement Learning

Train RL agents on trading environment:

```python
from src.models.rl_trainer import RLTrainingSystem

trainer = RLTrainingSystem(algorithm='PPO')
env = trainer.create_environment(market_data, initial_balance=10000)

trainer.train(env, total_timesteps=100000)
metrics = trainer.evaluate(env, n_episodes=10)
```

### 5. Portfolio Optimization

Modern Portfolio Theory optimization:

```python
from src.execution.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(optimization_method='max_sharpe')
weights = optimizer.optimize_allocation(price_data, predictions)
metrics = optimizer.calculate_metrics(weights, price_data)

print(f"Expected Return: {metrics['expected_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### 6. Risk Management

Comprehensive risk controls:

```python
from src.execution.portfolio_optimizer import RiskManager

risk_mgr = RiskManager(
    max_position_size=0.2,    # Max 20% per position
    stop_loss_pct=0.05,       # 5% stop loss
    max_portfolio_var=0.05    # Max 5% VaR
)

# Position sizing
adjusted_size = risk_mgr.check_position_size('AAPL', proposed_size, portfolio_value)

# Value at Risk
var = risk_mgr.calculate_var(portfolio, price_data, confidence=0.95)

# Stop loss checks
symbols_to_close = risk_mgr.apply_stop_loss(positions, current_prices)
```

### 7. Backtesting

Event-driven backtesting with realistic execution:

```python
from src.backtesting.backtest_engine import BacktestEngine

backtest = BacktestEngine(
    initial_cash=10000,
    commission=0.001,
    slippage=0.0005
)

results = backtest.run(data, strategy_func, strategy_params)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Safety Features

### Paper Trading Default

The system **defaults to paper trading** for safety:

```yaml
# config/config.yaml
system:
  mode: paper  # NEVER change without thorough testing!
```

### Live Trading Opt-in

To enable live trading (NOT RECOMMENDED without extensive testing):

1. Change `config/config.yaml`:
   ```yaml
   system:
     mode: live  # ⚠️ WARNING: Real money!
   ```

2. Set live API credentials in `.env`:
   ```
   ALPACA_LIVE_KEY=your_live_key
   ALPACA_LIVE_SECRET=your_live_secret
   ```

3. The system will show prominent warnings

### Risk Controls

- **Position Limits**: Max 20% per position (configurable)
- **Stop Losses**: Automatic 5% stop loss (configurable)
- **Drawdown Limits**: System pauses on excessive drawdown
- **Portfolio VaR**: Monitors Value at Risk
- **Correlation Checks**: Ensures diversification

## Configuration

### System Settings

```yaml
system:
  mode: paper                    # paper or live
  initial_capital: 100000        # Starting capital
  execution_interval: 300        # Seconds between iterations
  log_level: INFO                # Logging level
```

### Asset Selection

```yaml
assets:
  symbols:
    - AAPL
    - GOOGL
    - MSFT
  crypto:
    - BTC/USDT
    - ETH/USDT
  timeframes:
    - 1h
    - 4h
    - 1d
```

### Strategy Configuration

```yaml
strategies:
  llm:
    provider: anthropic           # or openai
    model: claude-sonnet-4-5
    temperature: 0.7
    max_strategies: 5
  rl:
    algorithm: PPO                # PPO, SAC, TD3, A2C
    training_timesteps: 1000000
```

### Risk Parameters

```yaml
risk:
  max_drawdown: 0.15              # Max 15% drawdown
  stop_loss_pct: 0.05             # 5% stop loss
  var_confidence: 0.95            # 95% VaR confidence
  max_correlation: 0.7            # Max correlation between positions
```

## API Requirements

### Required

- **Alpaca** (for trading): Free paper trading account at [alpaca.markets](https://alpaca.markets)
- **Anthropic** (for LLM strategies): API key from [anthropic.com](https://anthropic.com)

### Optional

- **NewsAPI** (for sentiment): Free tier at [newsapi.org](https://newsapi.org)
- **Alpha Vantage** (for data): Free tier at [alphavantage.co](https://alphavantage.co)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

Each module is independently testable:

```bash
# Test technical indicators
python src/data/technical_indicators.py

# Test strategy generation
python src/strategies/llm_strategy_generator.py

# Test backtesting
python src/backtesting/backtest_engine.py
```

## Monitoring & Logging

Logs are written to `logs/` directory with rotation:

```
logs/
├── trading_2025-10-22.log
├── trading_2025-10-23.log
└── ...
```

Monitor real-time with:

```bash
tail -f logs/trading_$(date +%Y-%m-%d).log
```

## Performance Metrics

The system tracks:

- **Returns**: Total, annualized, risk-adjusted
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown
- **Trading**: Win rate, profit factor, avg win/loss
- **Portfolio**: Allocation, diversification, VaR

## Disclaimer

**⚠️ IMPORTANT DISCLAIMER ⚠️**

This software is for **EDUCATIONAL PURPOSES ONLY**.

- Trading involves **SUBSTANTIAL RISK** of loss
- Past performance does **NOT** guarantee future results
- **NEVER** trade with money you cannot afford to lose
- Always thoroughly **BACKTEST** strategies before live trading
- Start with **PAPER TRADING** and only progress to live trading after extensive testing
- The authors are **NOT** responsible for any financial losses

**USE AT YOUR OWN RISK.**

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/auto-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/auto-trader/discussions)

## Roadmap

### Phase 1 (Complete)
- ✅ Core architecture
- ✅ Data acquisition
- ✅ Strategy generation (LLM)
- ✅ Backtesting framework
- ✅ Paper trading

### Phase 2 (Complete)
- ✅ Live trading integration
  - Safety monitoring with circuit breakers
  - Manual confirmation for live trades
  - Position and daily loss limits
- ✅ Advanced RL strategies
  - PPO, SAC, TD3, A2C algorithms
  - Gym-compatible trading environment
  - Model training and evaluation
- ✅ Multi-timeframe analysis
  - Technical indicators across timeframes
  - Ensemble price prediction (XGBoost, Prophet)
  - News sentiment analysis with FinBERT
- ✅ Web dashboard
  - Real-time portfolio monitoring
  - Performance analytics
  - Strategy management
  - Risk visualization

See `PHASE_2_SUMMARY.md` for detailed documentation.

### Phase 3 (Planned)
- 📋 Options trading
- 📋 Pairs trading / stat arb
- 📋 Social trading features
- 📋 Mobile app

## Acknowledgments

Built with:
- [Anthropic Claude](https://anthropic.com) - LLM strategy generation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL framework
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) - Portfolio optimization
- [Alpaca](https://alpaca.markets) - Trading platform
- And many other open-source libraries

---

**Happy Trading! 📈**

Remember: *Start with paper trading, test thoroughly, and never risk more than you can afford to lose.*
