"""
Price Prediction Engine
Ensemble of multiple models for price forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not available")


class PricePredictionEngine:
    """
    Ensemble price prediction using multiple models.
    Supports XGBoost, Prophet, and gradient boosting.
    """

    def __init__(self, models: List[str] = ['xgboost', 'prophet']):
        """
        Initialize prediction engine.

        Args:
            models: List of models to use ('xgboost', 'prophet', 'gbm')
        """
        self.models = models
        self.trained_models = {}
        self.scalers = {}
        self.feature_columns = []

        logger.info(f"Initialized PricePredictionEngine with models: {models}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with features
        """
        features = df.copy()

        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))

        # Volatility features
        features['volatility'] = features['returns'].rolling(window=20).std()

        # Volume features
        if 'volume' in features.columns:
            features['volume_change'] = features['volume'].pct_change()
            features['volume_ma'] = features['volume'].rolling(window=20).mean()

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = features['close'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)

        # Technical indicator features (if available)
        indicator_cols = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'atr', 'adx'
        ]

        # Drop NaN values
        features = features.dropna()

        return features

    def train_models(self, df: pd.DataFrame) -> None:
        """
        Train all models on historical data.

        Args:
            df: DataFrame with OHLCV and indicators
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for training")
            return

        logger.info("Training prediction models...")

        # Prepare features
        features = self.prepare_features(df)

        if features.empty:
            logger.warning("No features available after preparation")
            return

        # Define feature columns
        self.feature_columns = [col for col in features.columns if col not in
                                ['open', 'high', 'low', 'close', 'volume']]

        X = features[self.feature_columns]
        y = features['close']

        # Train each model
        if 'xgboost' in self.models and HAS_XGBOOST:
            self._train_xgboost(X, y)

        if 'prophet' in self.models and HAS_PROPHET:
            self._train_prophet(df)

        if 'gbm' in self.models:
            self._train_gbm(X, y)

        logger.info(f"Trained {len(self.trained_models)} models")

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost model."""
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_scaled, y)

            self.trained_models['xgboost'] = model
            self.scalers['xgboost'] = scaler

            logger.info("✓ XGBoost model trained")

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    def _train_prophet(self, df: pd.DataFrame) -> None:
        """Train Prophet model."""
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['close'].values
            })

            # Train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)

            self.trained_models['prophet'] = model

            logger.info("✓ Prophet model trained")

        except Exception as e:
            logger.error(f"Prophet training failed: {e}")

    def _train_gbm(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Gradient Boosting model."""
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_scaled, y)

            self.trained_models['gbm'] = model
            self.scalers['gbm'] = scaler

            logger.info("✓ Gradient Boosting model trained")

        except Exception as e:
            logger.error(f"GBM training failed: {e}")

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame,
        horizon: str = '1h'
    ) -> Dict:
        """
        Generate price prediction using ensemble of models.

        Args:
            symbol: Symbol being predicted
            df: DataFrame with recent data
            horizon: Prediction horizon

        Returns:
            Dictionary with prediction results
        """
        if not self.trained_models:
            logger.warning("No trained models available")
            return self._default_prediction(df)

        predictions = []
        weights = []

        # Get predictions from each model
        if 'xgboost' in self.trained_models:
            pred = self._predict_xgboost(df)
            if pred is not None:
                predictions.append(pred)
                weights.append(0.4)

        if 'prophet' in self.trained_models:
            pred = self._predict_prophet(df)
            if pred is not None:
                predictions.append(pred)
                weights.append(0.3)

        if 'gbm' in self.trained_models:
            pred = self._predict_gbm(df)
            if pred is not None:
                predictions.append(pred)
                weights.append(0.3)

        if not predictions:
            return self._default_prediction(df)

        # Ensemble prediction (weighted average)
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(predictions, weights=weights)

        # Calculate confidence metrics
        current_price = float(df['close'].iloc[-1])
        pred_change = (ensemble_pred - current_price) / current_price

        # Calculate probability of price increase
        if pred_change > 0:
            probability_up = 0.5 + min(abs(pred_change) * 10, 0.4)
        else:
            probability_up = 0.5 - min(abs(pred_change) * 10, 0.4)

        # Confidence interval (using std of predictions)
        if len(predictions) > 1:
            pred_std = np.std(predictions)
            ci_lower = ensemble_pred - 1.96 * pred_std
            ci_upper = ensemble_pred + 1.96 * pred_std
        else:
            volatility = df['close'].pct_change().std()
            ci_lower = ensemble_pred * (1 - 2 * volatility)
            ci_upper = ensemble_pred * (1 + 2 * volatility)

        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': float(ensemble_pred),
            'price_change': float(pred_change),
            'probability_up': float(probability_up),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'horizon': horizon,
            'timestamp': datetime.now(),
            'models_used': list(self.trained_models.keys())
        }

    def _predict_xgboost(self, df: pd.DataFrame) -> Optional[float]:
        """Get prediction from XGBoost model."""
        try:
            features = self.prepare_features(df)

            if features.empty:
                return None

            X = features[self.feature_columns].iloc[-1:].values
            X_scaled = self.scalers['xgboost'].transform(X)

            pred = self.trained_models['xgboost'].predict(X_scaled)[0]
            return float(pred)

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return None

    def _predict_prophet(self, df: pd.DataFrame) -> Optional[float]:
        """Get prediction from Prophet model."""
        try:
            # Create future dataframe
            future = pd.DataFrame({
                'ds': [df.index[-1] + timedelta(hours=1)]
            })

            forecast = self.trained_models['prophet'].predict(future)
            pred = forecast['yhat'].iloc[0]

            return float(pred)

        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return None

    def _predict_gbm(self, df: pd.DataFrame) -> Optional[float]:
        """Get prediction from GBM model."""
        try:
            features = self.prepare_features(df)

            if features.empty:
                return None

            X = features[self.feature_columns].iloc[-1:].values
            X_scaled = self.scalers['gbm'].transform(X)

            pred = self.trained_models['gbm'].predict(X_scaled)[0]
            return float(pred)

        except Exception as e:
            logger.error(f"GBM prediction failed: {e}")
            return None

    def _default_prediction(self, df: pd.DataFrame) -> Dict:
        """Return default prediction when models are not available."""
        current_price = float(df['close'].iloc[-1])

        # Simple momentum-based prediction
        recent_change = df['close'].pct_change().iloc[-5:].mean()
        predicted_price = current_price * (1 + recent_change)

        return {
            'symbol': 'UNKNOWN',
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': recent_change,
            'probability_up': 0.5 + (recent_change * 5),
            'confidence_interval': (current_price * 0.98, current_price * 1.02),
            'horizon': '1h',
            'timestamp': datetime.now(),
            'models_used': ['momentum']
        }

    def evaluate_predictions(
        self,
        predictions: List[Dict],
        actuals: List[float]
    ) -> Dict:
        """
        Evaluate prediction accuracy.

        Args:
            predictions: List of prediction dictionaries
            actuals: List of actual prices

        Returns:
            Evaluation metrics
        """
        if not predictions or not actuals:
            return {}

        pred_prices = [p['predicted_price'] for p in predictions]

        # Calculate metrics
        errors = np.array(actuals) - np.array(pred_prices)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        mape = np.mean(np.abs(errors / np.array(actuals))) * 100

        # Direction accuracy
        pred_directions = [1 if p['probability_up'] > 0.5 else -1 for p in predictions]
        actual_directions = [1 if actuals[i] > predictions[i]['current_price'] else -1
                            for i in range(len(predictions))]
        direction_accuracy = np.mean([1 if pred_directions[i] == actual_directions[i] else 0
                                     for i in range(len(predictions))])

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'n_predictions': len(predictions)
        }


if __name__ == "__main__":
    # Test the predictor
    from src.data.technical_indicators import TechnicalIndicatorEngine

    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=200)
    df = engine.calculate_indicators(df)

    predictor = PricePredictionEngine(models=['gbm', 'xgboost'])
    predictor.train_models(df)

    prediction = predictor.predict('AAPL', df, horizon='1h')

    print("\nPrice Prediction:")
    print(f"Symbol: {prediction['symbol']}")
    print(f"Current Price: ${prediction['current_price']:.2f}")
    print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
    print(f"Price Change: {prediction['price_change']*100:+.2f}%")
    print(f"Probability Up: {prediction['probability_up']:.2f}")
    print(f"Confidence Interval: ${prediction['confidence_interval'][0]:.2f} - ${prediction['confidence_interval'][1]:.2f}")
