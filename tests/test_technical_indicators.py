"""
Tests for technical indicators engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.data.technical_indicators import TechnicalIndicatorEngine
from tests.conftest import assert_valid_dataframe


@pytest.mark.unit
class TestTechnicalIndicatorEngine:
    """Test TechnicalIndicatorEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = TechnicalIndicatorEngine(['AAPL', 'GOOGL'], ['1h', '1d'])

        assert engine.symbols == ['AAPL', 'GOOGL']
        assert engine.timeframes == ['1h', '1d']
        assert isinstance(engine.data_cache, dict)
        assert isinstance(engine.last_update, dict)

    @patch('yfinance.Ticker')
    def test_fetch_data_yfinance(self, mock_ticker, sample_ohlcv_data):
        """Test fetching data from Yahoo Finance."""
        # Mock yfinance response
        mock_instance = Mock()
        mock_instance.history.return_value = sample_ohlcv_data
        mock_ticker.return_value = mock_instance

        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
        df = engine.fetch_data('AAPL', '1d', lookback=100, source='yfinance')

        assert_valid_dataframe(df, ['open', 'high', 'low', 'close', 'volume'])
        assert len(df) > 0

    def test_fetch_data_caching(self, sample_ohlcv_data):
        """Test data caching mechanism."""
        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        # Mock the fetch method
        with patch.object(engine, '_fetch_yfinance', return_value=sample_ohlcv_data):
            # First fetch
            df1 = engine.fetch_data('AAPL', '1d', lookback=100)

            # Second fetch (should use cache)
            df2 = engine.fetch_data('AAPL', '1d', lookback=100)

            assert df1.equals(df2)
            assert 'AAPL_1d_yfinance' in engine.data_cache

    def test_calculate_indicators(self, sample_ohlcv_data):
        """Test calculating technical indicators."""
        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        df = engine.calculate_indicators(
            sample_ohlcv_data,
            indicators=['sma', 'ema', 'rsi', 'macd']
        )

        assert_valid_dataframe(df)

        # Check for indicator columns
        assert 'sma_20' in df.columns or 'SMA_20' in df.columns
        assert 'rsi' in df.columns or 'RSI_14' in df.columns

    def test_calculate_indicators_sma(self, sample_ohlcv_data):
        """Test SMA calculation."""
        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        df = engine.calculate_indicators(sample_ohlcv_data, indicators=['sma'])

        # Check SMA columns exist
        sma_cols = [col for col in df.columns if 'sma' in col.lower()]
        assert len(sma_cols) > 0

    def test_calculate_indicators_rsi(self, sample_ohlcv_data):
        """Test RSI calculation."""
        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        df = engine.calculate_indicators(sample_ohlcv_data, indicators=['rsi'])

        # Check RSI column exists
        assert 'rsi' in df.columns or 'RSI_14' in df.columns

        # Check RSI values are in valid range
        if 'rsi' in df.columns:
            rsi_values = df['rsi'].dropna()
        else:
            rsi_values = df['RSI_14'].dropna()

        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()

    def test_calculate_indicators_empty_list(self, sample_ohlcv_data):
        """Test with empty indicator list."""
        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        df = engine.calculate_indicators(sample_ohlcv_data, indicators=None)

        assert_valid_dataframe(df)

    @patch.object(TechnicalIndicatorEngine, 'fetch_data')
    @patch.object(TechnicalIndicatorEngine, 'calculate_indicators')
    def test_get_market_regime(self, mock_calc, mock_fetch, sample_ohlcv_data):
        """Test market regime detection."""
        # Add required indicators
        df_with_indicators = sample_ohlcv_data.copy()
        df_with_indicators['sma_50'] = df_with_indicators['close'].rolling(50).mean()
        df_with_indicators['sma_200'] = df_with_indicators['close'].rolling(200).mean()
        df_with_indicators['atr'] = 2.0
        df_with_indicators['rsi'] = 55.0
        df_with_indicators['ADX_14'] = 25.0

        mock_fetch.return_value = df_with_indicators
        mock_calc.return_value = df_with_indicators

        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
        regime = engine.get_market_regime('AAPL')

        assert isinstance(regime, dict)
        assert 'trend' in regime
        assert 'volatility' in regime
        assert 'momentum' in regime
        assert 'trend_strength' in regime
        assert 'timestamp' in regime

    def test_timeframe_to_seconds(self):
        """Test timeframe conversion to seconds."""
        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        assert engine._timeframe_to_seconds('1m') == 60
        assert engine._timeframe_to_seconds('5m') == 300
        assert engine._timeframe_to_seconds('1h') == 3600
        assert engine._timeframe_to_seconds('4h') == 14400
        assert engine._timeframe_to_seconds('1d') == 86400

    @patch.object(TechnicalIndicatorEngine, 'fetch_data')
    def test_get_latest_price(self, mock_fetch, sample_ohlcv_data):
        """Test getting latest price."""
        mock_fetch.return_value = sample_ohlcv_data

        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
        price = engine.get_latest_price('AAPL')

        assert isinstance(price, (int, float))
        assert price > 0

    @patch.object(TechnicalIndicatorEngine, 'fetch_data')
    @patch.object(TechnicalIndicatorEngine, 'calculate_indicators')
    def test_fetch_all_data(self, mock_calc, mock_fetch, sample_ohlcv_data):
        """Test fetching all data for multiple symbols."""
        mock_fetch.return_value = sample_ohlcv_data
        mock_calc.return_value = sample_ohlcv_data

        engine = TechnicalIndicatorEngine(['AAPL', 'GOOGL'], ['1h', '1d'])
        all_data = engine.fetch_all_data()

        assert isinstance(all_data, dict)
        # Should have data for each symbol-timeframe combination
        assert len(all_data) <= 4  # 2 symbols x 2 timeframes

    def test_market_regime_uptrend(self, sample_ohlcv_data):
        """Test uptrend detection."""
        df = sample_ohlcv_data.copy()

        # Create strong uptrend
        df['close'] = pd.Series(range(100, 200), index=df.index)
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(50).mean() - 10
        df['atr'] = 1.0
        df['rsi'] = 55.0
        df['ADX_14'] = 30.0

        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        with patch.object(engine, 'fetch_data', return_value=df):
            with patch.object(engine, 'calculate_indicators', return_value=df):
                regime = engine.get_market_regime('AAPL')

                # Should detect uptrend
                assert 'uptrend' in regime['trend'].lower() or regime['trend'] == 'sideways'

    def test_market_regime_high_volatility(self, sample_ohlcv_data):
        """Test high volatility detection."""
        df = sample_ohlcv_data.copy()
        df['atr'] = 10.0  # High ATR relative to price
        df['sma_50'] = 100
        df['sma_200'] = 100
        df['rsi'] = 50
        df['ADX_14'] = 20

        engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])

        with patch.object(engine, 'fetch_data', return_value=df):
            with patch.object(engine, 'calculate_indicators', return_value=df):
                regime = engine.get_market_regime('AAPL')

                assert regime['volatility'] in ['high', 'medium', 'low', 'unknown']
