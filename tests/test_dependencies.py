"""
Test that all required dependencies can be imported successfully.
"""

import pytest


class TestCoreDependencies:
    """Test core Python dependencies."""

    def test_import_numpy(self):
        """Test numpy import."""
        import numpy as np
        assert np.__version__

    def test_import_pandas(self):
        """Test pandas import."""
        import pandas as pd
        assert pd.__version__

    def test_import_scipy(self):
        """Test scipy import."""
        import scipy
        assert scipy.__version__

    def test_import_sklearn(self):
        """Test scikit-learn import."""
        import sklearn
        assert sklearn.__version__


class TestMLDependencies:
    """Test machine learning dependencies."""

    def test_import_xgboost(self):
        """Test XGBoost import."""
        import xgboost
        assert xgboost.__version__

    def test_import_gymnasium(self):
        """Test gymnasium import."""
        import gymnasium
        assert gymnasium.__version__


class TestDataDependencies:
    """Test data fetching dependencies."""

    def test_import_yfinance(self):
        """Test yfinance import."""
        import yfinance
        assert yfinance.__version__

    def test_import_ccxt(self):
        """Test ccxt import."""
        import ccxt
        assert ccxt.__version__

    def test_import_beautifulsoup4(self):
        """Test BeautifulSoup4 import."""
        from bs4 import BeautifulSoup
        assert BeautifulSoup


class TestVisualizationDependencies:
    """Test visualization dependencies."""

    def test_import_matplotlib(self):
        """Test matplotlib import."""
        import matplotlib
        assert matplotlib.__version__

    def test_import_seaborn(self):
        """Test seaborn import."""
        import seaborn
        assert seaborn.__version__

    def test_import_plotly(self):
        """Test plotly import."""
        import plotly
        assert plotly.__version__


class TestOptimizationDependencies:
    """Test optimization dependencies."""

    def test_import_cvxpy(self):
        """Test cvxpy import."""
        import cvxpy
        assert cvxpy.__version__

    def test_import_pypfopt(self):
        """Test PyPortfolioOpt import."""
        from pypfopt import EfficientFrontier
        assert EfficientFrontier


class TestUtilityDependencies:
    """Test utility dependencies."""

    def test_import_loguru(self):
        """Test loguru import."""
        from loguru import logger
        assert logger

    def test_import_dotenv(self):
        """Test python-dotenv import."""
        from dotenv import load_dotenv
        assert load_dotenv

    def test_import_pytest(self):
        """Test pytest import."""
        import pytest
        assert pytest.__version__

    def test_import_hypothesis(self):
        """Test hypothesis import."""
        import hypothesis
        assert hypothesis.__version__


class TestBrokerDependencies:
    """Test broker-related dependencies."""

    def test_import_alpaca(self):
        """Test alpaca-trade-api import."""
        import alpaca_trade_api
        # Version check might fail due to msgpack version conflict
        # but import should work

    def test_import_aiohttp(self):
        """Test aiohttp import."""
        import aiohttp
        assert aiohttp.__version__

    def test_import_websockets(self):
        """Test websockets import."""
        import websockets
        assert websockets.__version__


class TestProjectImports:
    """Test that all project modules can be imported."""

    def test_import_backtesting(self):
        """Test backtesting module import."""
        from src.backtesting.backtest_engine import BacktestEngine, PerformanceAnalyzer
        assert BacktestEngine
        assert PerformanceAnalyzer

    def test_import_models(self):
        """Test models module import."""
        from src.models.price_predictor import PricePredictionEngine
        from src.models.rl_environment import TradingEnvironment
        assert PricePredictionEngine
        assert TradingEnvironment

    def test_import_execution(self):
        """Test execution module import."""
        from src.execution.broker import AlpacaBroker, OrderManager
        from src.execution.portfolio_optimizer import PortfolioOptimizer, RiskManager
        assert AlpacaBroker
        assert OrderManager
        assert PortfolioOptimizer
        assert RiskManager

    def test_import_data(self):
        """Test data module import."""
        from src.data.technical_indicators import TechnicalIndicatorEngine
        assert TechnicalIndicatorEngine

    def test_import_utils(self):
        """Test utils module import."""
        from src.utils.config_loader import ConfigLoader
        assert ConfigLoader


class TestCompatibilityChecks:
    """Test version compatibility and conflicts."""

    def test_numpy_version(self):
        """Test numpy version is compatible."""
        import numpy as np
        major, minor = map(int, np.__version__.split('.')[:2])
        # numpy >= 1.20 or numpy 2.x
        assert (major == 1 and minor >= 20) or major >= 2

    def test_pandas_version(self):
        """Test pandas version is compatible."""
        import pandas as pd
        major, minor = map(int, pd.__version__.split('.')[:2])
        assert major >= 2 or (major == 1 and minor >= 2)  # pandas >= 1.2

    def test_sklearn_version(self):
        """Test scikit-learn version is compatible."""
        import sklearn
        major, minor = map(int, sklearn.__version__.split('.')[:2])
        assert major >= 1  # sklearn >= 1.0


@pytest.mark.slow
class TestDependencyFunctionality:
    """Test that dependencies work correctly."""

    def test_numpy_operations(self):
        """Test basic numpy operations."""
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.std() > 0

    def test_pandas_operations(self):
        """Test basic pandas operations."""
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        assert df['a'].sum() == 6

    def test_xgboost_model(self):
        """Test XGBoost can create a model."""
        import xgboost as xgb
        import numpy as np

        model = xgb.XGBRegressor(n_estimators=10)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == 3

    def test_matplotlib_plot(self):
        """Test matplotlib can create plots."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.close(fig)

    def test_gymnasium_environment(self):
        """Test gymnasium can create environments."""
        import gymnasium as gym

        # Create a simple environment
        env = gym.make('CartPole-v1')
        obs, info = env.reset()
        assert obs is not None
        env.close()
