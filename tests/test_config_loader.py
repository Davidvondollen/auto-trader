"""
Tests for configuration loader.
"""

import pytest
import os
from pathlib import Path
from src.utils.config_loader import ConfigLoader, load_config


@pytest.mark.unit
class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_load_config_success(self, temp_config_file, temp_env_file):
        """Test successful configuration loading."""
        config = ConfigLoader(temp_config_file)

        assert config.config is not None
        assert isinstance(config.config, dict)

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader('/nonexistent/config.yaml')

    def test_get_value_dot_notation(self, temp_config_file):
        """Test getting values using dot notation."""
        config = ConfigLoader(temp_config_file)

        assert config.get('system.mode') == 'paper'
        assert config.get('system.initial_capital') == 100000
        assert config.get('assets.symbols') == ['AAPL', 'GOOGL']

    def test_get_value_default(self, temp_config_file):
        """Test getting values with default."""
        config = ConfigLoader(temp_config_file)

        assert config.get('nonexistent.key', 'default') == 'default'
        assert config.get('system.nonexistent', 42) == 42

    def test_get_env(self, temp_config_file, temp_env_file):
        """Test getting environment variables."""
        config = ConfigLoader(temp_config_file)

        assert config.get_env('ALPACA_PAPER_KEY') == 'test_paper_key'
        assert config.get_env('NONEXISTENT_KEY', 'default') == 'default'

    def test_get_api_key(self, temp_config_file, temp_env_file):
        """Test getting API keys."""
        config = ConfigLoader(temp_config_file)

        assert config.get_api_key('alpaca_paper_key') == 'test_paper_key'
        assert config.get_api_key('anthropic') == 'test_anthropic_key'

    def test_get_api_key_unknown_service(self, temp_config_file):
        """Test getting API key for unknown service."""
        config = ConfigLoader(temp_config_file)

        with pytest.raises(ValueError):
            config.get_api_key('unknown_service')

    def test_is_paper_trading(self, temp_config_file):
        """Test paper trading check."""
        config = ConfigLoader(temp_config_file)

        assert config.is_paper_trading() is True

    def test_get_symbols(self, temp_config_file):
        """Test getting trading symbols."""
        config = ConfigLoader(temp_config_file)

        symbols = config.get_symbols()
        assert symbols == ['AAPL', 'GOOGL']
        assert len(symbols) == 2

    def test_get_timeframes(self, temp_config_file):
        """Test getting timeframes."""
        config = ConfigLoader(temp_config_file)

        timeframes = config.get_timeframes()
        assert '1h' in timeframes
        assert '1d' in timeframes

    def test_get_initial_capital(self, temp_config_file):
        """Test getting initial capital."""
        config = ConfigLoader(temp_config_file)

        capital = config.get_initial_capital()
        assert capital == 100000.0
        assert isinstance(capital, float)

    def test_validate_config_success(self, temp_config_file):
        """Test configuration validation success."""
        config = ConfigLoader(temp_config_file)

        assert config.validate_config() is True

    def test_validate_config_missing_field(self, tmp_path):
        """Test configuration validation with missing field."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("system:\n  mode: paper")

        config = ConfigLoader(str(bad_config))
        assert config.validate_config() is False

    def test_validate_config_invalid_mode(self, tmp_path):
        """Test configuration validation with invalid mode."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("""
system:
  mode: invalid
  initial_capital: 100000
assets:
  symbols:
    - AAPL
brokers:
  default: alpaca
""")

        config = ConfigLoader(str(bad_config))
        assert config.validate_config() is False

    def test_to_dict(self, temp_config_file):
        """Test converting config to dictionary."""
        config = ConfigLoader(temp_config_file)

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'system' in config_dict
        assert 'assets' in config_dict

    def test_load_config_convenience_function(self, temp_config_file):
        """Test convenience function for loading config."""
        config = load_config(temp_config_file)

        assert isinstance(config, ConfigLoader)
        assert config.get('system.mode') == 'paper'
