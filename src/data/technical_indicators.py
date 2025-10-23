"""
Technical Indicators Engine
Handles data acquisition and technical indicator calculation across multiple timeframes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import ccxt


class TechnicalIndicatorEngine:
    """
    Fetches market data and calculates technical indicators across multiple timeframes.
    Supports stocks, crypto, and forex.
    """

    def __init__(self, symbols: List[str], timeframes: List[str]):
        """
        Initialize the engine.

        Args:
            symbols: List of symbols to track (e.g., ['AAPL', 'GOOGL'])
            timeframes: List of timeframes (e.g., ['1h', '4h', '1d'])
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_cache = {}
        self.last_update = {}

        # Initialize crypto exchange for crypto symbols
        self.crypto_exchange = None
        if any('/' in symbol for symbol in symbols):
            try:
                self.crypto_exchange = ccxt.binance()
            except Exception as e:
                logger.warning(f"Could not initialize crypto exchange: {e}")

    def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 500,
        source: str = 'yfinance'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol and timeframe.

        Args:
            symbol: Symbol to fetch
            timeframe: Timeframe (1h, 4h, 1d, etc.)
            lookback: Number of bars to fetch
            source: Data source ('yfinance' for stocks, 'crypto' for crypto)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{source}"

        # Check cache (refresh if older than timeframe interval)
        if cache_key in self.data_cache:
            last_fetch = self.last_update.get(cache_key, datetime.min)
            interval_seconds = self._timeframe_to_seconds(timeframe)

            if (datetime.now() - last_fetch).total_seconds() < interval_seconds / 2:
                logger.debug(f"Using cached data for {cache_key}")
                return self.data_cache[cache_key].copy()

        try:
            # Determine if crypto or stock based on source parameter or symbol format
            if source == 'crypto' or '/' in symbol:
                df = self._fetch_crypto(symbol, timeframe, lookback)
            else:
                df = self._fetch_stock(symbol, timeframe, lookback)

            if df is not None and not df.empty:
                self.data_cache[cache_key] = df
                self.last_update[cache_key] = datetime.now()
                logger.info(f"Fetched {len(df)} bars for {cache_key}")
                return df.copy()
            else:
                logger.warning(f"No data fetched for {cache_key}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {cache_key}: {e}")
            return pd.DataFrame()

    def _fetch_stock(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Fetch stock data using yfinance."""
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h',  # 4h will be resampled
            '1d': '1d', '1w': '1wk', '1M': '1mo'
        }

        interval = interval_map.get(timeframe, '1d')

        # Calculate period
        period_map = {
            '1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d',
            '1h': '730d', '4h': '730d', '1d': 'max'
        }
        period = period_map.get(timeframe, '2y')

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        # Standardize column names
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'volume': 'volume'})

        # Resample 4h if needed
        if timeframe == '4h':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        # Limit to lookback
        df = df.tail(lookback)

        return df

    def _fetch_crypto(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Fetch crypto data using ccxt."""
        if not self.crypto_exchange:
            logger.warning("Crypto exchange not initialized")
            return pd.DataFrame()

        try:
            # CCXT timeframe format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }
            ccxt_timeframe = timeframe_map.get(timeframe, '1d')

            # Fetch OHLCV
            since = None  # Fetch recent data
            ohlcv = self.crypto_exchange.fetch_ohlcv(
                symbol,
                timeframe=ccxt_timeframe,
                limit=lookback
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate technical indicators on OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to calculate (None = all)

        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df

        df = df.copy()

        # Default to all indicators
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'adx']

        try:
            if 'sma' in indicators:
                df = self._calculate_sma(df)

            if 'ema' in indicators:
                df = self._calculate_ema(df)

            if 'rsi' in indicators:
                df = self._calculate_rsi(df)

            if 'macd' in indicators:
                df = self._calculate_macd(df)

            if 'bbands' in indicators:
                df = self._calculate_bollinger_bands(df)

            if 'atr' in indicators:
                df = self._calculate_atr(df)

            if 'adx' in indicators:
                df = self._calculate_adx(df)

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return df

    def _calculate_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    def _calculate_ema(self, df: pd.DataFrame, periods: List[int] = [12, 26, 50]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        for period in periods:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        if len(df) < period:
            return df

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        if len(df) < slow:
            return df

        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        if len(df) < period:
            return df

        df['bb_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        df['bb_upper'] = df['bb_middle'] + (std_dev * std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        if len(df) < period:
            return df

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        if len(df) < period * 2:
            return df

        # Calculate directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate ATR for normalization
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate directional indicators
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        # Calculate ADX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['di_plus'] = pos_di
        df['di_minus'] = neg_di

        return df

    def get_market_regime(self, symbol: str, timeframe: str = '1d') -> Dict:
        """
        Determine current market regime for a symbol.

        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to use

        Returns:
            Dictionary with regime information
        """
        df = self.fetch_data(symbol, timeframe, lookback=100)

        if df.empty:
            return {
                'trend': 'unknown',
                'volatility': 'unknown',
                'momentum': 'unknown',
                'trend_strength': 0.0,
                'timestamp': datetime.now()
            }

        df = self.calculate_indicators(df)

        # Determine trend
        trend = 'sideways'
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]

            if current_price > sma_20 > sma_50:
                trend = 'uptrend'
            elif current_price < sma_20 < sma_50:
                trend = 'downtrend'

        # Determine volatility
        volatility = 'medium'
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            atr_pct = (atr / df['close'].iloc[-1]) * 100

            if atr_pct > 3:
                volatility = 'high'
            elif atr_pct < 1:
                volatility = 'low'

        # Determine momentum
        momentum = 'neutral'
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]

            if rsi > 60:
                momentum = 'positive'
            elif rsi < 40:
                momentum = 'negative'

        # Trend strength
        trend_strength = 0.0
        if 'adx' in df.columns:
            trend_strength = df['adx'].iloc[-1] / 100.0

        return {
            'trend': trend,
            'volatility': volatility,
            'momentum': momentum,
            'trend_strength': trend_strength,
            'timestamp': datetime.now()
        }

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols and timeframes.

        Returns:
            Dictionary mapping 'symbol_timeframe' to DataFrames
        """
        all_data = {}

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                df = self.fetch_data(symbol, timeframe)

                if not df.empty:
                    df = self.calculate_indicators(df)
                    key = f"{symbol}_{timeframe}"
                    all_data[key] = df

        return all_data

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Latest price or None
        """
        df = self.fetch_data(symbol, '1d', lookback=1)

        if df.empty:
            return None

        return float(df['close'].iloc[-1])

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        multiplier = int(timeframe[:-1]) if timeframe[:-1] else 1
        unit = timeframe[-1]

        units = {
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'M': 2592000
        }

        return multiplier * units.get(unit, 86400)


if __name__ == "__main__":
    # Test the engine
    engine = TechnicalIndicatorEngine(['AAPL', 'GOOGL'], ['1h', '1d'])

    # Fetch data
    df = engine.fetch_data('AAPL', '1d', lookback=100)
    print(f"\nFetched {len(df)} bars for AAPL")
    print(df.tail())

    # Calculate indicators
    df = engine.calculate_indicators(df)
    print(f"\nCalculated indicators:")
    print(df[['close', 'rsi', 'macd', 'bb_upper', 'bb_lower']].tail())

    # Get market regime
    regime = engine.get_market_regime('AAPL')
    print(f"\nMarket Regime for AAPL:")
    print(regime)
