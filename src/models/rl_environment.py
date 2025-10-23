"""
Reinforcement Learning Trading Environment
Gymnasium-compatible environment for training RL agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment for RL agents.

    Observation space: Market data window (OHLCV + indicators)
    Action space: Continuous [-1, 1] where:
        - Negative values = sell
        - Positive values = buy
        - 0 = hold
        Magnitude indicates position size
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        window_size: int = 20,
        max_position: float = 1.0
    ):
        """
        Initialize trading environment.

        Args:
            data: DataFrame with OHLCV and indicators
            initial_balance: Starting cash balance
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
            window_size: Number of bars in observation window
            max_position: Maximum position size (1.0 = 100% of balance)
        """
        super(TradingEnvironment, self).__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.window_size = window_size
        self.max_position = max_position

        # Feature columns (exclude non-feature columns)
        exclude_cols = []
        self.feature_columns = [col for col in data.columns
                               if col not in exclude_cols and data[col].dtype in [np.float64, np.float32, np.int64]]

        # Define observation space
        # Shape: (window_size, n_features + 3)
        # +3 for: balance, shares_held, cost_basis
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features + 3),
            dtype=np.float32
        )

        # Define action space: continuous [-1, 1]
        # -1 = sell all, 0 = hold, 1 = buy max
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0.0
        self.cost_basis = 0.0
        self.total_trades = 0
        self.portfolio_history = []
        self.trade_history = []

        logger.info(f"TradingEnvironment initialized with {len(data)} steps, "
                   f"window_size={window_size}, features={n_features}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.cost_basis = 0.0
        self.total_trades = 0
        self.portfolio_history = []
        self.trade_history = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (continuous value in [-1, 1])

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Ensure action is scalar
        if isinstance(action, np.ndarray):
            action = action[0]

        action = float(np.clip(action, -1.0, 1.0))

        # Get current price
        current_price = self.data.loc[self.current_step, 'close']

        # Execute trade
        reward = self._execute_trade(action, current_price)

        # Move to next step
        self.current_step += 1

        # Calculate portfolio value
        portfolio_value = self.balance + (self.shares_held * current_price)
        self.portfolio_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': portfolio_value,
            'price': current_price
        })

        # Check if terminated (end of data)
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # We don't truncate episodes

        # Get observation
        observation = self._get_observation()

        # Info dict
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'current_price': current_price
        }

        return observation, reward, terminated, truncated, info

    def _execute_trade(self, action: float, current_price: float) -> float:
        """
        Execute trade based on action.

        Args:
            action: Action value in [-1, 1]
            current_price: Current asset price

        Returns:
            Reward for this action
        """
        prev_portfolio_value = self.balance + (self.shares_held * current_price)

        # Apply slippage
        if action > 0:
            # Buy - price goes up
            execution_price = current_price * (1 + self.slippage)
        elif action < 0:
            # Sell - price goes down
            execution_price = current_price * (1 - self.slippage)
        else:
            # Hold
            return 0.0

        if action > 0:
            # Buy
            max_shares = (self.balance * self.max_position) / execution_price
            shares_to_buy = max_shares * abs(action)

            if shares_to_buy > 0:
                cost = shares_to_buy * execution_price
                commission_cost = cost * self.commission

                if self.balance >= cost + commission_cost:
                    self.shares_held += shares_to_buy
                    self.balance -= (cost + commission_cost)
                    self.cost_basis = (self.cost_basis * (self.shares_held - shares_to_buy) +
                                      cost) / self.shares_held if self.shares_held > 0 else execution_price

                    self.total_trades += 1
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': execution_price,
                        'cost': cost + commission_cost
                    })

        elif action < 0:
            # Sell
            shares_to_sell = self.shares_held * abs(action)

            if shares_to_sell > 0 and self.shares_held >= shares_to_sell:
                proceeds = shares_to_sell * execution_price
                commission_cost = proceeds * self.commission

                self.shares_held -= shares_to_sell
                self.balance += (proceeds - commission_cost)

                self.total_trades += 1
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'proceeds': proceeds - commission_cost
                })

        # Calculate reward (change in portfolio value)
        new_portfolio_value = self.balance + (self.shares_held * current_price)
        reward = new_portfolio_value - prev_portfolio_value

        # Normalize reward
        reward = reward / prev_portfolio_value if prev_portfolio_value > 0 else 0.0

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array of shape (window_size, n_features + 3)
        """
        # Get window of market data
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step

        # Use iloc for half-open interval [start_idx, end_idx) to get exactly window_size rows
        window_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values

        # Pad if necessary
        if len(window_data) < self.window_size:
            padding = np.zeros((self.window_size - len(window_data), len(self.feature_columns)))
            window_data = np.vstack([padding, window_data])

        # Normalize market data
        window_data = self._normalize(window_data)

        # Add portfolio state to each row
        current_price = self.data.loc[self.current_step, 'close']
        portfolio_value = self.balance + (self.shares_held * current_price)

        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance if portfolio_value > 0 else 0,
            self.cost_basis / current_price if current_price > 0 and self.shares_held > 0 else 1.0
        ])

        # Repeat portfolio state for each window row
        portfolio_state_repeated = np.tile(portfolio_state, (self.window_size, 1))

        # Concatenate
        observation = np.hstack([window_data, portfolio_state_repeated])

        return observation.astype(np.float32)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using min-max scaling per feature.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        if len(data) == 0:
            return data

        # Calculate min and max for each feature
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # Normalize
        normalized = (data - min_vals) / range_vals

        return normalized

    def _get_info(self) -> Dict:
        """
        Get info dictionary with current state.

        Returns:
            Info dictionary
        """
        current_price = self.data.loc[self.current_step, 'close'] if self.current_step < len(self.data) else 0
        portfolio_value = self.balance + (self.shares_held * current_price)
        return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        return {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': portfolio_value,
            'return': return_pct,
            'current_price': current_price,
            'total_trades': self.total_trades
        }

    def render(self, mode: str = 'human') -> None:
        """
        Render environment state.

        Args:
            mode: Render mode
        """
        current_price = self.data.loc[self.current_step, 'close']
        portfolio_value = self.balance + (self.shares_held * current_price)
        profit = portfolio_value - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100

        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{len(self.data)-1}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares Held: {self.shares_held:.4f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Profit: ${profit:.2f} ({profit_pct:+.2f}%)")
        print(f"Total Trades: {self.total_trades}")
        print(f"{'='*60}")

    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio history as DataFrame.

        Returns:
            DataFrame with portfolio history
        """
        return pd.DataFrame(self.portfolio_history)

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame with trade history
        """
        return pd.DataFrame(self.trade_history)


if __name__ == "__main__":
    # Test environment
    from src.data.technical_indicators import TechnicalIndicatorEngine

    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=200)
    df = engine.calculate_indicators(df)

    env = TradingEnvironment(df, initial_balance=10000)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Run random episode
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 10 == 0:
            env.render()

        if terminated or truncated:
            break

    print(f"\nEpisode finished!")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
