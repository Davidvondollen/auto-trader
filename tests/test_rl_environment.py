"""
Tests for RL trading environment.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.rl_environment import TradingEnvironment


@pytest.fixture
def sample_rl_data():
    """Fixture for sample RL environment data."""
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=100))
    df = pd.DataFrame({
        'open': np.random.uniform(98, 102, 100),
        'high': np.random.uniform(100, 105, 100),
        'low': np.random.uniform(95, 100, 100),
        'close': np.random.uniform(99, 104, 100),
        'volume': np.random.uniform(1e6, 5e6, 100)
    }, index=dates)
    return df


@pytest.mark.unit
class TestTradingEnvironment:
    """Test TradingEnvironment class."""

    def test_initialization(self, sample_rl_data):
        """Test environment initialization."""
        env = TradingEnvironment(sample_rl_data)
        assert env.initial_balance == 10000
        assert env.current_step == 0

    def test_reset(self, sample_rl_data):
        """Test environment reset."""
        env = TradingEnvironment(sample_rl_data)
        obs, info = env.reset()
        assert env.current_step == env.window_size
        assert isinstance(obs, np.ndarray)

    def test_step_buy_action(self, sample_rl_data):
        """Test step with buy action."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        initial_balance = env.balance
        obs, reward, term, trunc, info = env.step(np.array([0.5]))
        assert env.balance < initial_balance
        assert env.shares_held > 0

    def test_step_sell_action(self, sample_rl_data):
        """Test step with sell action."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        env.shares_held = 10
        initial_balance = env.balance
        obs, reward, term, trunc, info = env.step(np.array([-0.5]))
        assert env.balance > initial_balance

    def test_step_hold_action(self, sample_rl_data):
        """Test step with hold action."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        initial_balance = env.balance
        initial_shares = env.shares_held
        obs, reward, term, trunc, info = env.step(np.array([0.0]))
        assert env.balance == initial_balance
        assert env.shares_held == initial_shares

    def test_get_observation(self, sample_rl_data):
        """Test observation space."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        obs = env._get_observation()
        assert obs.shape == env.observation_space.shape

    def test_calculate_reward(self, sample_rl_data):
        """Test reward calculation."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        env.balance = 11000
        reward = env._execute_trade(0.5, 100)
        assert reward != 0

    def test_episode_termination(self, sample_rl_data):
        """Test episode termination."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        env.current_step = len(sample_rl_data) - 2
        obs, reward, term, trunc, info = env.step(np.array([0.0]))
        assert term

    def test_portfolio_history(self, sample_rl_data):
        """Test portfolio history tracking."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        env.step(np.array([0.5]))
        history = env.get_portfolio_history()
        assert isinstance(history, pd.DataFrame)
        assert not history.empty

    def test_trades_history(self, sample_rl_data):
        """Test trade history tracking."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        env.step(np.array([0.5]))
        history = env.get_trade_history()
        assert isinstance(history, pd.DataFrame)
        assert not history.empty

    def test_commission_applied(self, sample_rl_data):
        """Test commission is applied correctly."""
        env_comm = TradingEnvironment(sample_rl_data, commission=0.01)
        env_no_comm = TradingEnvironment(sample_rl_data, commission=0.0)
        env_comm.reset()
        env_no_comm.reset()
        env_comm.step(np.array([0.5]))
        env_no_comm.step(np.array([0.5]))
        assert env_comm.balance < env_no_comm.balance

    def test_slippage_applied(self, sample_rl_data):
        """Test slippage is applied correctly."""
        env_slippage = TradingEnvironment(sample_rl_data, slippage=0.01)
        env_no_slippage = TradingEnvironment(sample_rl_data, slippage=0.0)
        env_slippage.reset()
        env_no_slippage.reset()
        env_slippage.step(np.array([0.01]))
        env_no_slippage.step(np.array([0.01]))
        assert env_slippage.balance < env_no_slippage.balance

    def test_info_dict(self, sample_rl_data):
        """Test info dictionary content."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        obs, reward, term, trunc, info = env.step(np.array([0.0]))
        assert 'portfolio_value' in info
        assert 'balance' in info
        assert 'shares_held' in info

    def test_render(self, sample_rl_data):
        """Test rendering."""
        env = TradingEnvironment(sample_rl_data)
        env.reset()
        env.render()  # Should not raise error


@pytest.mark.slow
class TestTradingEnvironmentIntegration:
    """Integration tests for TradingEnvironment."""

    def test_full_episode(self, sample_rl_data):
        """Test running a full episode."""
        env = TradingEnvironment(sample_rl_data)
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        assert info['portfolio_value'] > 0

    def test_deterministic_episode(self, sample_rl_data):
        """Test that two episodes with same seed are deterministic."""
        env1 = TradingEnvironment(sample_rl_data)
        env2 = TradingEnvironment(sample_rl_data)
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)
        assert np.array_equal(obs1, obs2)

        for _ in range(5):
            action = env1.action_space.sample()
            obs1, _, _, _, _ = env1.step(action)
            obs2, _, _, _, _ = env2.step(action)
            assert np.array_equal(obs1, obs2)

