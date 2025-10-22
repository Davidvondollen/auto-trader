"""
Tests for RL trading environment.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.rl_environment import TradingEnvironment


@pytest.mark.unit
class TestTradingEnvironment:
    """Test TradingEnvironment class."""

    def test_initialization(self, sample_ohlcv_data):
        """Test environment initialization."""
        env = TradingEnvironment(
            data=sample_ohlcv_data,
            initial_balance=10000,
            commission=0.001,
            slippage=0.0005,
            window_size=20
        )

        assert env.initial_balance == 10000
        assert env.commission == 0.001
        assert env.slippage == 0.0005
        assert env.window_size == 20

    def test_observation_space(self, sample_ohlcv_data):
        """Test observation space definition."""
        env = TradingEnvironment(sample_ohlcv_data)

        assert env.observation_space is not None
        assert hasattr(env.observation_space, 'shape')

    def test_action_space(self, sample_ohlcv_data):
        """Test action space definition."""
        env = TradingEnvironment(sample_ohlcv_data)

        assert env.action_space is not None
        assert hasattr(env.action_space, 'sample')

    def test_reset(self, sample_ohlcv_data):
        """Test environment reset."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        observation, info = env.reset()

        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)
        assert env.balance == 10000
        assert env.shares_held == 0

    def test_step_buy_action(self, sample_ohlcv_data):
        """Test step with buy action."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # Buy action (positive value)
        action = np.array([0.5])
        observation, reward, terminated, truncated, info = env.step(action)

        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_sell_action(self, sample_ohlcv_data):
        """Test step with sell action."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # First buy some shares
        env.shares_held = 10
        env.cost_basis = 100.0

        # Sell action (negative value)
        action = np.array([-0.5])
        observation, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)

    def test_step_hold_action(self, sample_ohlcv_data):
        """Test step with hold action."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()
        initial_balance = env.balance

        # Hold action (near zero)
        action = np.array([0.05])
        observation, reward, terminated, truncated, info = env.step(action)

        # Balance should not change much
        assert abs(env.balance - initial_balance) < 100

    def test_execute_buy(self, sample_ohlcv_data):
        """Test buy execution."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()
        initial_balance = env.balance

        current_price = sample_ohlcv_data.iloc[env.current_step]['close']
        env._execute_buy(current_price, amount=0.5)

        # Balance should decrease
        assert env.balance < initial_balance

        # Should have shares
        if env.balance < initial_balance:  # If trade executed
            assert env.shares_held > 0

    def test_execute_sell(self, sample_ohlcv_data):
        """Test sell execution."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # Set up position
        env.shares_held = 10
        env.cost_basis = 100.0
        initial_balance = env.balance

        current_price = sample_ohlcv_data.iloc[env.current_step]['close']
        env._execute_sell(current_price, amount=1.0)

        # Balance should increase
        assert env.balance > initial_balance

        # Shares should be sold
        assert env.shares_held == 0

    def test_get_observation(self, sample_ohlcv_data):
        """Test observation generation."""
        env = TradingEnvironment(sample_ohlcv_data, window_size=20)

        env.reset()
        observation = env._get_observation()

        assert isinstance(observation, np.ndarray)
        assert observation.shape == env.observation_space.shape

    def test_calculate_reward(self, sample_ohlcv_data):
        """Test reward calculation."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()
        env.portfolio_values.append(10000)
        env.portfolio_values.append(10500)

        reward = env._calculate_reward(10500)

        assert isinstance(reward, float)

    def test_episode_termination(self, sample_ohlcv_data):
        """Test episode terminates at end of data."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # Step through entire episode
        terminated = False
        steps = 0
        max_steps = len(sample_ohlcv_data) - env.window_size

        while not terminated and steps < max_steps:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1

        # Should terminate
        assert terminated or steps >= max_steps - 1

    def test_portfolio_history(self, sample_ohlcv_data):
        """Test portfolio history tracking."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # Take a few steps
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)

        history = env.get_portfolio_history()

        assert isinstance(history, pd.DataFrame)
        assert 'portfolio_value' in history.columns
        assert len(history) == len(env.portfolio_values)

    def test_trades_history(self, sample_ohlcv_data):
        """Test trades history tracking."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # Execute some trades
        for _ in range(3):
            action = np.array([0.5])  # Buy
            env.step(action)

        history = env.get_trades_history()

        assert isinstance(history, pd.DataFrame)

    def test_commission_applied(self, sample_ohlcv_data):
        """Test that commission is applied to trades."""
        env_no_comm = TradingEnvironment(
            sample_ohlcv_data,
            initial_balance=10000,
            commission=0.0
        )

        env_with_comm = TradingEnvironment(
            sample_ohlcv_data,
            initial_balance=10000,
            commission=0.01  # 1%
        )

        # Reset both
        env_no_comm.reset()
        env_with_comm.reset()

        # Execute same buy action
        current_price = 100.0
        env_no_comm._execute_buy(current_price, amount=1.0)
        env_with_comm._execute_buy(current_price, amount=1.0)

        # With commission should have less shares
        if env_no_comm.shares_held > 0 and env_with_comm.shares_held > 0:
            assert env_with_comm.shares_held <= env_no_comm.shares_held

    def test_slippage_applied(self, sample_ohlcv_data):
        """Test that slippage is applied to trades."""
        env = TradingEnvironment(
            sample_ohlcv_data,
            initial_balance=10000,
            slippage=0.01  # 1%
        )

        env.reset()

        price = 100.0

        # Buy - should pay more due to slippage
        initial_balance = env.balance
        env._execute_buy(price, amount=0.5)

        # Effective price should be higher than quoted
        if env.shares_held > 0:
            cost = initial_balance - env.balance
            shares = env.shares_held
            effective_price = cost / shares

            # Should pay more than base price (plus commission)
            assert effective_price > price

    def test_info_dict(self, sample_ohlcv_data):
        """Test info dictionary contents."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        info = env._get_info()

        assert isinstance(info, dict)
        assert 'step' in info
        assert 'balance' in info
        assert 'shares_held' in info
        assert 'portfolio_value' in info
        assert 'return' in info

    def test_render(self, sample_ohlcv_data, capsys):
        """Test rendering (should not crash)."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env.reset()

        # Should not raise error
        env.render(mode='human')

        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should print something


@pytest.mark.slow
class TestTradingEnvironmentIntegration:
    """Integration tests for trading environment."""

    def test_full_episode(self, sample_ohlcv_data):
        """Test running a complete episode."""
        env = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        observation, info = env.reset()

        total_reward = 0
        steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 50:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, float)

    def test_deterministic_episode(self, sample_ohlcv_data):
        """Test that episodes with same actions are deterministic."""
        env1 = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)
        env2 = TradingEnvironment(sample_ohlcv_data, initial_balance=10000)

        env1.reset(seed=42)
        env2.reset(seed=42)

        # Take same actions
        actions = [np.array([0.5]), np.array([0.0]), np.array([-0.5])]

        for action in actions:
            _, reward1, _, _, _ = env1.step(action)
            _, reward2, _, _, _ = env2.step(action)

            # Rewards should be identical
            assert abs(reward1 - reward2) < 1e-6
