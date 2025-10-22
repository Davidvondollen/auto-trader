"""
Reinforcement Learning Training System
Supports multiple RL algorithms: PPO, SAC, TD3, A2C
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from loguru import logger

from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.models.rl_environment import TradingEnvironment


class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    """

    def __init__(self, verbose: int = 0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                if 'portfolio_value' in info:
                    self.portfolio_values.append(info['portfolio_value'])

                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    logger.info(f"Episodes: {len(self.episode_rewards)}, "
                              f"Avg Reward: {avg_reward:.4f}, "
                              f"Avg Length: {avg_length:.1f}")

        return True


class RLTrainingSystem:
    """
    Reinforcement Learning training system with support for multiple algorithms.
    """

    ALGORITHMS = {
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C
    }

    def __init__(self, algorithm: str = 'PPO', learning_rate: float = 0.0003):
        """
        Initialize RL training system.

        Args:
            algorithm: RL algorithm to use (PPO, SAC, TD3, A2C)
            learning_rate: Learning rate for training
        """
        if algorithm.upper() not in self.ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not supported. "
                           f"Choose from: {list(self.ALGORITHMS.keys())}")

        self.algorithm = algorithm.upper()
        self.learning_rate = learning_rate
        self.model = None
        self.env = None
        self.training_history = []

        logger.info(f"Initialized RLTrainingSystem with {self.algorithm}")

    def create_environment(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        window_size: int = 20
    ) -> TradingEnvironment:
        """
        Create trading environment.

        Args:
            data: Market data DataFrame
            initial_balance: Starting balance
            commission: Commission rate
            slippage: Slippage rate
            window_size: Observation window size

        Returns:
            Trading environment
        """
        env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            commission=commission,
            slippage=slippage,
            window_size=window_size
        )

        self.env = env
        logger.info(f"Created trading environment with {len(data)} steps")

        return env

    def train(
        self,
        env: Optional[TradingEnvironment] = None,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        save_path: Optional[str] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train RL agent.

        Args:
            env: Trading environment (uses self.env if None)
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_path: Path to save model
            verbose: Verbosity level

        Returns:
            Training metrics
        """
        if env is None:
            env = self.env

        if env is None:
            raise ValueError("No environment available. Call create_environment first.")

        logger.info(f"Starting {self.algorithm} training for {total_timesteps} timesteps...")

        # Wrap environment
        monitor_env = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitor_env])

        # Create model
        model_class = self.ALGORITHMS[self.algorithm]

        # Algorithm-specific parameters
        if self.algorithm == 'PPO':
            self.model = model_class(
                'MlpPolicy',
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=verbose
            )
        elif self.algorithm == 'SAC':
            self.model = model_class(
                'MlpPolicy',
                vec_env,
                learning_rate=self.learning_rate,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                verbose=verbose
            )
        elif self.algorithm == 'TD3':
            self.model = model_class(
                'MlpPolicy',
                vec_env,
                learning_rate=self.learning_rate,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                verbose=verbose
            )
        elif self.algorithm == 'A2C':
            self.model = model_class(
                'MlpPolicy',
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=5,
                gamma=0.99,
                gae_lambda=0.95,
                verbose=verbose
            )

        # Setup callbacks
        callback = TradingCallback(verbose=verbose)

        # Train
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        training_time = (datetime.now() - start_time).total_seconds()

        # Save model
        if save_path:
            self.save_model(save_path)

        # Compile metrics
        metrics = {
            'algorithm': self.algorithm,
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'episode_rewards': callback.episode_rewards,
            'episode_lengths': callback.episode_lengths,
            'avg_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0,
            'std_reward': np.std(callback.episode_rewards) if callback.episode_rewards else 0,
            'max_reward': np.max(callback.episode_rewards) if callback.episode_rewards else 0,
            'min_reward': np.min(callback.episode_rewards) if callback.episode_rewards else 0
        }

        self.training_history.append(metrics)

        logger.info(f"Training complete in {training_time:.1f}s")
        logger.info(f"Avg Reward: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}")

        return metrics

    def evaluate(
        self,
        env: Optional[TradingEnvironment] = None,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate trained agent.

        Args:
            env: Trading environment
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No trained model available")

        if env is None:
            env = self.env

        if env is None:
            raise ValueError("No environment available")

        logger.info(f"Evaluating {self.algorithm} for {n_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        final_portfolio_values = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            final_portfolio_values.append(info['portfolio_value'])

            logger.info(f"Episode {episode+1}/{n_episodes}: "
                       f"Reward={episode_reward:.4f}, "
                       f"Portfolio=${info['portfolio_value']:.2f}")

        # Calculate metrics
        initial_balance = env.initial_balance
        returns = [(pv - initial_balance) / initial_balance for pv in final_portfolio_values]

        metrics = {
            'n_episodes': n_episodes,
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'avg_episode_length': float(np.mean(episode_lengths)),
            'avg_portfolio_value': float(np.mean(final_portfolio_values)),
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'max_return': float(np.max(returns)),
            'min_return': float(np.min(returns)),
            'win_rate': float(np.sum(np.array(returns) > 0) / n_episodes)
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Avg Return: {metrics['avg_return']*100:.2f}% ± {metrics['std_return']*100:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate']*100:.1f}%")

        return metrics

    def predict_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for given observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic action

        Returns:
            action, state
        """
        if self.model is None:
            raise ValueError("No trained model available")

        return self.model.predict(observation, deterministic=deterministic)

    def save_model(self, path: str) -> None:
        """
        Save trained model.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(save_path))
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str) -> None:
        """
        Load trained model.

        Args:
            path: Path to model file
        """
        model_class = self.ALGORITHMS[self.algorithm]
        self.model = model_class.load(path)
        logger.info(f"Model loaded from {path}")

    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame.

        Returns:
            DataFrame with training history
        """
        return pd.DataFrame(self.training_history)

    def compare_algorithms(
        self,
        data: pd.DataFrame,
        algorithms: List[str] = ['PPO', 'SAC', 'TD3', 'A2C'],
        total_timesteps: int = 50000,
        n_eval_episodes: int = 5
    ) -> pd.DataFrame:
        """
        Compare multiple RL algorithms.

        Args:
            data: Market data
            algorithms: List of algorithms to compare
            total_timesteps: Training timesteps per algorithm
            n_eval_episodes: Evaluation episodes per algorithm

        Returns:
            DataFrame with comparison results
        """
        results = []

        for algo in algorithms:
            if algo.upper() not in self.ALGORITHMS:
                logger.warning(f"Skipping unknown algorithm: {algo}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Training {algo}")
            logger.info(f"{'='*60}")

            # Create trainer
            trainer = RLTrainingSystem(algorithm=algo)

            # Create environment
            env = trainer.create_environment(data)

            # Train
            try:
                train_metrics = trainer.train(
                    env=env,
                    total_timesteps=total_timesteps,
                    verbose=0
                )

                # Evaluate
                eval_metrics = trainer.evaluate(
                    env=env,
                    n_episodes=n_eval_episodes
                )

                # Combine metrics
                result = {
                    'algorithm': algo,
                    'training_time': train_metrics['training_time'],
                    'avg_train_reward': train_metrics['avg_reward'],
                    'avg_eval_return': eval_metrics['avg_return'],
                    'std_eval_return': eval_metrics['std_return'],
                    'win_rate': eval_metrics['win_rate'],
                    'avg_portfolio_value': eval_metrics['avg_portfolio_value']
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error training {algo}: {e}")
                continue

        comparison_df = pd.DataFrame(results)

        logger.info(f"\n{'='*60}")
        logger.info("Algorithm Comparison")
        logger.info(f"{'='*60}")
        print(comparison_df)

        return comparison_df


if __name__ == "__main__":
    # Test RL trainer
    from src.data.technical_indicators import TechnicalIndicatorEngine

    logger.info("Testing RL Training System")

    # Fetch data
    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=500)
    df = engine.calculate_indicators(df)

    # Create trainer
    trainer = RLTrainingSystem(algorithm='PPO')

    # Create environment
    env = trainer.create_environment(df, initial_balance=10000)

    # Train
    metrics = trainer.train(
        env=env,
        total_timesteps=10000,
        verbose=1
    )

    print("\nTraining Metrics:")
    for key, value in metrics.items():
        if key not in ['episode_rewards', 'episode_lengths']:
            print(f"{key}: {value}")

    # Evaluate
    eval_metrics = trainer.evaluate(env=env, n_episodes=5)

    print("\nEvaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")
