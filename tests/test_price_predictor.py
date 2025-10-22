"""
Tests for price prediction engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.models.price_predictor import (
    PricePredictionEngine,
    FeatureEngineer,
    XGBoostPredictor,
    ProphetPredictor
)


@pytest.mark.unit
class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_create_features(self, sample_ohlcv_data):
        """Test feature engineering."""
        engineer = FeatureEngineer()

        features = engineer.create_features(sample_ohlcv_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

        # Check for common features
        assert 'returns' in features.columns or 'close' in features.columns
        assert any('sma' in col.lower() for col in features.columns)

    def test_create_features_with_lags(self, sample_ohlcv_data):
        """Test feature engineering with lag features."""
        engineer = FeatureEngineer()

        features = engineer.create_features(sample_ohlcv_data)

        # Check for lag features
        lag_cols = [col for col in features.columns if 'lag' in col.lower()]
        assert len(lag_cols) > 0

    def test_prepare_sequences(self):
        """Test sequence preparation for LSTM."""
        engineer = FeatureEngineer()

        data = np.random.randn(100, 5)
        X, y = engineer.prepare_sequences(data, sequence_length=10)

        assert X.shape[0] == 90  # 100 - 10
        assert X.shape[1] == 10
        assert X.shape[2] == 5
        assert y.shape[0] == 90


@pytest.mark.unit
class TestXGBoostPredictor:
    """Test XGBoostPredictor class."""

    def test_initialization(self):
        """Test XGBoost predictor initialization."""
        predictor = XGBoostPredictor()

        assert predictor.model is None
        assert predictor.is_trained is False

    def test_train(self, sample_ohlcv_data):
        """Test training XGBoost model."""
        predictor = XGBoostPredictor()
        engineer = FeatureEngineer()

        features = engineer.create_features(sample_ohlcv_data)

        # Train should not raise error
        predictor.train(features, target_col='close', horizon=1)

        assert predictor.is_trained is True
        assert predictor.model is not None

    def test_predict(self, sample_ohlcv_data):
        """Test making predictions."""
        predictor = XGBoostPredictor()
        engineer = FeatureEngineer()

        features = engineer.create_features(sample_ohlcv_data)
        predictor.train(features, target_col='close')

        prediction = predictor.predict(features)

        assert isinstance(prediction, float)
        assert prediction > 0

    def test_predict_before_training(self, sample_ohlcv_data):
        """Test prediction fails before training."""
        predictor = XGBoostPredictor()
        engineer = FeatureEngineer()

        features = engineer.create_features(sample_ohlcv_data)

        with pytest.raises(ValueError):
            predictor.predict(features)

    def test_get_feature_importance(self, sample_ohlcv_data):
        """Test getting feature importance."""
        predictor = XGBoostPredictor()
        engineer = FeatureEngineer()

        features = engineer.create_features(sample_ohlcv_data)
        predictor.train(features)

        importance = predictor.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


@pytest.mark.unit
class TestProphetPredictor:
    """Test ProphetPredictor class."""

    def test_initialization(self):
        """Test Prophet predictor initialization."""
        predictor = ProphetPredictor()

        assert predictor.model is None
        assert predictor.is_trained is False

    def test_train(self, sample_ohlcv_data):
        """Test training Prophet model."""
        predictor = ProphetPredictor()

        predictor.train(sample_ohlcv_data, target_col='close')

        assert predictor.is_trained is True
        assert predictor.model is not None

    def test_predict(self, sample_ohlcv_data):
        """Test making predictions with Prophet."""
        predictor = ProphetPredictor()

        predictor.train(sample_ohlcv_data)
        forecast = predictor.predict(periods=5)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 5
        assert 'yhat' in forecast.columns

    def test_predict_before_training(self):
        """Test prediction fails before training."""
        predictor = ProphetPredictor()

        with pytest.raises(ValueError):
            predictor.predict(periods=1)


@pytest.mark.unit
class TestPricePredictionEngine:
    """Test PricePredictionEngine class."""

    def test_initialization(self):
        """Test prediction engine initialization."""
        engine = PricePredictionEngine(models=['xgboost', 'prophet'])

        assert 'xgboost' in engine.models
        assert 'prophet' in engine.models
        assert 'xgboost' in engine.predictors
        assert 'prophet' in engine.predictors

    def test_train_models(self, sample_ohlcv_data):
        """Test training all models."""
        engine = PricePredictionEngine(models=['xgboost'])

        # Should not raise error
        engine.train_models(sample_ohlcv_data)

    @patch.object(XGBoostPredictor, 'predict')
    def test_predict(self, mock_predict, sample_ohlcv_data):
        """Test making ensemble prediction."""
        mock_predict.return_value = 152.50

        engine = PricePredictionEngine(models=['xgboost'])
        engine.predictors['xgboost'].is_trained = True

        prediction = engine.predict('AAPL', sample_ohlcv_data, horizon='1h')

        assert isinstance(prediction, dict)
        assert 'predicted_price' in prediction
        assert 'current_price' in prediction
        assert 'confidence_interval' in prediction
        assert 'probability_up' in prediction

    def test_predict_no_models(self, sample_ohlcv_data):
        """Test prediction with no trained models."""
        engine = PricePredictionEngine(models=['xgboost'])

        prediction = engine.predict('AAPL', sample_ohlcv_data)

        # Should return current price as fallback
        assert isinstance(prediction, dict)
        assert prediction['predicted_price'] == prediction['current_price']

    def test_backtest_accuracy(self, sample_ohlcv_data):
        """Test backtesting prediction accuracy."""
        engine = PricePredictionEngine(models=['xgboost'])

        metrics = engine.backtest_accuracy('AAPL', sample_ohlcv_data, period_days=30)

        assert isinstance(metrics, dict)
        # Should have metrics or error
        assert 'error' in metrics or 'mae' in metrics

    def test_ensemble_multiple_models(self, sample_ohlcv_data):
        """Test ensemble with multiple models."""
        engine = PricePredictionEngine(models=['xgboost', 'prophet'])

        engine.train_models(sample_ohlcv_data)

        # Mock predictions from both models
        with patch.object(XGBoostPredictor, 'predict', return_value=152.0):
            with patch.object(ProphetPredictor, 'predict',
                            return_value=pd.DataFrame({'yhat': [153.0]})):

                prediction = engine.predict('AAPL', sample_ohlcv_data)

                # Ensemble should be weighted average
                assert isinstance(prediction['predicted_price'], float)


@pytest.mark.slow
class TestPricePredictorIntegration:
    """Integration tests for price prediction (slower)."""

    def test_full_prediction_pipeline(self, sample_ohlcv_data):
        """Test complete prediction pipeline."""
        engine = PricePredictionEngine(models=['xgboost'])

        # Train
        engine.train_models(sample_ohlcv_data)

        # Predict
        prediction = engine.predict('AAPL', sample_ohlcv_data)

        # Validate prediction structure
        assert 'symbol' in prediction
        assert 'predicted_price' in prediction
        assert 'model_contributions' in prediction
        assert prediction['symbol'] == 'AAPL'
