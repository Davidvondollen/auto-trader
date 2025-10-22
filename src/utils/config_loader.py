"""
Configuration loader for the agentic trading system.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


class ConfigLoader:
    """Load and manage configuration from YAML file and environment variables."""

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'system.mode')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_env(self, env_var: str, default: str = None) -> str:
        """
        Get environment variable value.

        Args:
            env_var: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value
        """
        return os.getenv(env_var, default)

    def get_api_key(self, service: str) -> str:
        """
        Get API key for a service from environment variables.

        Args:
            service: Service name (e.g., 'anthropic', 'alpaca')

        Returns:
            API key

        Raises:
            ValueError: If API key not found
        """
        # Map service names to env var names
        env_var_map = {
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'alpaca_paper_key': 'ALPACA_PAPER_KEY',
            'alpaca_paper_secret': 'ALPACA_PAPER_SECRET',
            'alpaca_live_key': 'ALPACA_LIVE_KEY',
            'alpaca_live_secret': 'ALPACA_LIVE_SECRET',
            'newsapi': 'NEWS_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_KEY',
            'polygon': 'POLYGON_API_KEY',
            'finnhub': 'FINNHUB_API_KEY',
        }

        env_var = env_var_map.get(service.lower())
        if not env_var:
            raise ValueError(f"Unknown service: {service}")

        api_key = os.getenv(env_var)
        if not api_key:
            logger.warning(f"API key for {service} not found in environment variables")
            return None

        return api_key

    def is_paper_trading(self) -> bool:
        """Check if system is in paper trading mode."""
        mode = self.get('system.mode', 'paper')
        return mode == 'paper'

    def get_symbols(self) -> list:
        """Get list of symbols to trade."""
        return self.get('assets.symbols', [])

    def get_timeframes(self) -> list:
        """Get list of timeframes to analyze."""
        return self.get('assets.timeframes', ['1h', '4h', '1d'])

    def get_initial_capital(self) -> float:
        """Get initial capital for trading."""
        return float(self.get('system.initial_capital', 100000))

    def validate_config(self) -> bool:
        """
        Validate configuration for required fields.

        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'system.mode',
            'system.initial_capital',
            'assets.symbols',
            'brokers.default',
        ]

        for field in required_fields:
            if self.get(field) is None:
                logger.error(f"Required configuration field missing: {field}")
                return False

        # Validate mode
        mode = self.get('system.mode')
        if mode not in ['paper', 'live']:
            logger.error(f"Invalid system mode: {mode}. Must be 'paper' or 'live'")
            return False

        # Validate symbols
        symbols = self.get_symbols()
        if not symbols:
            logger.error("No trading symbols configured")
            return False

        logger.info("Configuration validation passed")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


def load_config(config_path: str = None) -> ConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config.yaml file

    Returns:
        ConfigLoader instance
    """
    config = ConfigLoader(config_path)

    if not config.validate_config():
        raise ValueError("Configuration validation failed")

    return config


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print(f"Mode: {config.get('system.mode')}")
    print(f"Symbols: {config.get_symbols()}")
    print(f"Initial Capital: {config.get_initial_capital()}")
    print(f"Is Paper Trading: {config.is_paper_trading()}")
