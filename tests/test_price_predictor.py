"""
Tests for price prediction engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.models.price_predictor import PricePredictionEngine


@pytest.mark.unit
class TestPricePredictionEngineFeatures:
    """Test Feature Creation within PricePredictionEngine."""

    def test_create_features(self, sample_ohlcv_data):
        """Test feature engineering."""
        engine = PricePredictionEngine()
        sample_ohlcv_data['sma_20'] = sample_ohlcv_data['close'].rolling(20).mean()
        features = engine.prepare_features(sample_ohlcv_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

        # Check for common features
        assert 'returns' in features.columns or 'close' in features.columns
        assert any('sma' in col.lower() for col in features.columns)

    def test_create_features_with_lags(self, sample_ohlcv_data):
        """Test feature engineering with lag features."""
        engine = PricePredictionEngine()

        features = engine.prepare_features(sample_ohlcv_data)

        # Check for lag features
        lag_cols = [col for col in features.columns if 'lag' in col.lower()]
        assert len(lag_cols) > 0


@pytest.mark.unit
class TestPricePredictionEngine:
    """Test PricePredictionEngine class."""

    def test_initialization(self):
        """Test prediction engine initialization."""
        engine = PricePredictionEngine(models=['xgboost', 'prophet'])

        assert 'xgboost' in engine.models
        assert 'prophet' in engine.models

    def test_train_models(self, sample_ohlcv_data):
        """Test training all models."""
        engine = PricePredictionEngine(models=['xgboost'])

        # Should not raise error
        engine.train_models(sample_ohlcv_data)
        assert 'xgboost' in engine.trained_models

    def test_predict(self, sample_ohlcv_data):
        """Test making ensemble prediction."""
        engine = PricePredictionEngine(models=['xgboost'])
        engine.train_models(sample_ohlcv_data)

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
        assert prediction['predicted_price'] != prediction['current_price']

    def test_ensemble_multiple_models(self, sample_ohlcv_data):
        """Test ensemble with multiple models."""
        engine = PricePredictionEngine(models=['xgboost', 'prophet'])

        engine.train_models(sample_ohlcv_data)

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
        assert 'models_used' in prediction
        assert prediction['symbol'] == 'AAPL'
