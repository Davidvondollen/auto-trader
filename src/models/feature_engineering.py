"""
Advanced Feature Engineering for Trading
Includes microstructure, order flow, and market regime features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for price prediction and trading strategies.

    Includes:
    - Microstructure features (bid-ask spread proxies, tick analysis)
    - Order flow features (volume imbalance, trade intensity)
    - Market regime features (volatility regimes, trend strength)
    - Time-based features (intraday patterns, day-of-week effects)
    """

    def __init__(self):
        """Initialize feature engineer."""
        logger.info("Initialized AdvancedFeatureEngineer")

    def create_all_features(
        self,
        df: pd.DataFrame,
        include_microstructure: bool = True,
        include_order_flow: bool = True,
        include_regime: bool = True,
        include_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Create all available features.

        Args:
            df: DataFrame with OHLCV data
            include_microstructure: Include microstructure features
            include_order_flow: Include order flow features
            include_regime: Include market regime features
            include_temporal: Include temporal features

        Returns:
            DataFrame with added features
        """
        features = df.copy()

        if include_microstructure:
            features = self.add_microstructure_features(features)

        if include_order_flow:
            features = self.add_order_flow_features(features)

        if include_regime:
            features = self.add_regime_features(features)

        if include_temporal:
            features = self.add_temporal_features(features)

        return features

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure features.

        These approximate bid-ask spreads and market liquidity using OHLC data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with microstructure features
        """
        features = df.copy()

        # High-Low spread (proxy for intraday volatility and liquidity)
        features['hl_spread'] = (features['high'] - features['low']) / features['close']

        # Range position (where did price close in the day's range?)
        # 0 = closed at low, 1 = closed at high
        range_size = features['high'] - features['low']
        range_size = range_size.replace(0, np.nan)  # Avoid division by zero
        features['range_position'] = (features['close'] - features['low']) / range_size

        # Open-Close spread
        features['oc_spread'] = np.abs(features['close'] - features['open']) / features['close']

        # Price efficiency ratio (trending vs random walk)
        # High ratio = strong trend, low ratio = choppy/random
        lookback = 10
        price_change = np.abs(features['close'] - features['close'].shift(lookback))
        path_length = features['close'].diff().abs().rolling(window=lookback).sum()
        features['efficiency_ratio'] = price_change / (path_length + 1e-8)

        # Garman-Klass volatility estimator (more efficient than close-to-close)
        # Uses OHLC information
        hl_ratio = np.log(features['high'] / features['low']) ** 2
        oc_ratio = np.log(features['close'] / features['open']) ** 2
        features['gk_volatility'] = np.sqrt(0.5 * hl_ratio - (2 * np.log(2) - 1) * oc_ratio)

        # Parkinson volatility (another OHLC volatility estimator)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(features['high'] / features['low']) ** 2)
        )

        # Rolling volatility ratio (current vs historical)
        historical_vol = features['gk_volatility'].rolling(window=20, min_periods=1).mean()
        features['vol_ratio'] = features['gk_volatility'] / (historical_vol + 1e-8)

        logger.debug("Added microstructure features")
        return features

    def add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add order flow features using volume and price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with order flow features
        """
        features = df.copy()

        if 'volume' not in features.columns:
            logger.warning("Volume data not available for order flow features")
            return features

        # On-Balance Volume (OBV) - cumulative buying/selling pressure
        price_direction = np.sign(features['close'].diff())
        features['obv'] = (price_direction * features['volume']).cumsum()

        # Volume-weighted price change
        features['vw_price_change'] = (
            features['close'].pct_change() * features['volume']
        ).rolling(window=20, min_periods=1).sum()

        # Volume momentum (acceleration/deceleration)
        features['volume_momentum'] = features['volume'].pct_change()

        # Relative volume (current vs average)
        vol_ma = features['volume'].rolling(window=20, min_periods=1).mean()
        features['relative_volume'] = features['volume'] / (vol_ma + 1e-8)

        # Volume price trend (VPT) - similar to OBV but considers magnitude
        features['vpt'] = (
            (features['close'].pct_change() * features['volume']).cumsum()
        )

        # Money Flow Index (MFI) - volume-weighted RSI
        typical_price = (features['high'] + features['low'] + features['close']) / 3
        raw_money_flow = typical_price * features['volume']

        # Positive and negative money flow
        money_flow_pos = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        money_flow_neg = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        money_flow_ratio = (
            money_flow_pos.rolling(window=14, min_periods=1).sum() /
            (money_flow_neg.rolling(window=14, min_periods=1).sum() + 1e-8)
        )
        features['mfi'] = 100 - (100 / (1 + money_flow_ratio))

        # Volume imbalance (buy vs sell pressure proxy)
        # Assume: close > open = buying pressure
        buy_volume = features['volume'].where(features['close'] > features['open'], 0)
        sell_volume = features['volume'].where(features['close'] < features['open'], 0)

        features['volume_imbalance'] = (
            (buy_volume - sell_volume) /
            (buy_volume + sell_volume + 1e-8)
        )

        # Trade intensity (volume per price move)
        price_range = features['high'] - features['low']
        features['trade_intensity'] = features['volume'] / (price_range + 1e-8)

        logger.debug("Added order flow features")
        return features

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features.

        Identifies different market conditions: trending, ranging, volatile, calm.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with regime features
        """
        features = df.copy()

        # Trend strength using ADX-like calculation
        # High values = strong trend, low values = ranging
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift(1))
        low_close = np.abs(features['low'] - features['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        plus_dm = features['high'].diff()
        minus_dm = -features['low'].diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = true_range.rolling(window=14, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / (atr + 1e-8))

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        features['trend_strength'] = dx.rolling(window=14, min_periods=1).mean()

        # Volatility regime
        returns = features['close'].pct_change()
        vol_short = returns.rolling(window=10, min_periods=1).std()
        vol_long = returns.rolling(window=50, min_periods=1).std()

        features['vol_regime'] = vol_short / (vol_long + 1e-8)
        # >1 = increasing volatility, <1 = decreasing volatility

        # Hurst exponent (mean reversion vs trending)
        # ~0.5 = random walk, >0.5 = trending, <0.5 = mean reverting
        def calculate_hurst(prices, lags=20):
            """Calculate Hurst exponent."""
            if len(prices) < lags + 2:
                return 0.5

            lags_range = range(2, lags + 1)
            tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags_range]

            if not tau or min(tau) == 0:
                return 0.5

            poly = np.polyfit(np.log(lags_range), np.log(tau), 1)
            return poly[0] * 2.0

        # Calculate rolling Hurst
        hurst_window = 50
        hurst_values = []

        for i in range(len(features)):
            if i < hurst_window:
                hurst_values.append(0.5)
            else:
                price_window = features['close'].iloc[i-hurst_window:i].values
                hurst = calculate_hurst(price_window, lags=min(20, hurst_window // 2))
                hurst_values.append(hurst)

        features['hurst_exponent'] = hurst_values

        # Market regime classification
        # Combine trend strength and volatility
        features['regime_trending'] = (features['trend_strength'] > 25).astype(int)
        features['regime_high_vol'] = (features['vol_regime'] > 1.2).astype(int)
        features['regime_mean_reverting'] = (features['hurst_exponent'] < 0.45).astype(int)

        logger.debug("Added market regime features")
        return features

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Captures day-of-week effects, hour-of-day patterns, etc.

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame with temporal features
        """
        features = df.copy()

        if not isinstance(features.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping temporal features")
            return features

        # Day of week (Monday = 0, Friday = 4)
        features['day_of_week'] = features.index.dayofweek

        # Month of year
        features['month'] = features.index.month

        # Day of month
        features['day_of_month'] = features.index.day

        # Quarter
        features['quarter'] = features.index.quarter

        # Is month end
        features['is_month_end'] = features.index.is_month_end.astype(int)

        # Is quarter end
        features['is_quarter_end'] = features.index.is_quarter_end.astype(int)

        # Time-based cyclical encoding (for continuous representation)
        # This is better than raw numbers for ML models
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 5)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 5)

        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        # If intraday data, add hour features
        if features.index.hour.nunique() > 1:
            features['hour'] = features.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

            # Market session (US market hours)
            features['is_market_hours'] = (
                (features['hour'] >= 9) & (features['hour'] < 16)
            ).astype(int)

        # Days since epoch (for long-term trends)
        features['days_since_start'] = (
            (features.index - features.index[0]).days
        )

        logger.debug("Added temporal features")
        return features

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        correlation_threshold: float = 0.95
    ) -> List[str]:
        """
        Select most relevant features, removing highly correlated ones.

        Args:
            df: DataFrame with features
            target_col: Target column
            correlation_threshold: Remove features with correlation above this

        Returns:
            List of selected feature names
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and non-feature columns
        exclude_cols = [target_col, 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not feature_cols:
            return []

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()

        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Remove one from each highly correlated pair
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > correlation_threshold)
        ]

        selected_features = [col for col in feature_cols if col not in to_drop]

        logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")

        return selected_features


if __name__ == "__main__":
    # Test feature engineering
    from src.data.technical_indicators import TechnicalIndicatorEngine

    logger.info("Testing AdvancedFeatureEngineer")

    # Fetch data
    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=200)
    df = engine.calculate_indicators(df)

    # Create features
    fe = AdvancedFeatureEngineer()
    df_with_features = fe.create_all_features(df)

    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Total features: {len(df_with_features.columns)}")
    print(f"\nNew features added:")
    new_features = set(df_with_features.columns) - set(df.columns)
    for feat in sorted(new_features):
        print(f"  - {feat}")

    # Feature selection
    selected = fe.select_features(df_with_features)
    print(f"\nSelected {len(selected)} features after correlation filtering")
