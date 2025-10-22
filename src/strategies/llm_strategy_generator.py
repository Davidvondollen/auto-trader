"""
LLM-based Trading Strategy Generator.
Uses Claude or GPT to generate and evolve trading strategies.
"""

import re
import ast
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
from anthropic import Anthropic
import json


STRATEGY_GENERATION_PROMPT = """
You are an expert quantitative trader tasked with generating a Python trading strategy.

Market Context:
- Asset: {symbol}
- Market Regime: {market_regime}
- Current Trend: {trend}
- Volatility: {volatility}
- Recent Performance: {performance_summary}

Available Data Features:
- Technical Indicators: {available_indicators}
- Sentiment Score: {sentiment_score}
- Price Predictions: {prediction_summary}

Requirements:
1. Generate a trading strategy class that inherits from BaseStrategy
2. Implement generate_signal() method that returns a signal dictionary
3. Include clear entry and exit logic based on the data
4. Add risk management rules (stop loss, position sizing)
5. Be conservative with risk controls
6. Use only the provided indicators and data

Signal Format:
{{
    'action': 'buy' | 'sell' | 'hold',
    'confidence': 0.0 to 1.0,
    'quantity': position size (0.0 to 1.0),
    'reason': 'explanation',
    'stop_loss': optional stop loss price,
    'take_profit': optional take profit price
}}

Return ONLY the Python code for the strategy class, nothing else.
The class should be named {strategy_name}.

Example structure:
```python
class {strategy_name}(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.name = "{strategy_name}"
        # Initialize parameters

    def generate_signal(self, data, predictions=None, sentiment=None):
        # Strategy logic here
        # Return signal dictionary
        pass
```

Generate the strategy code now:
"""

STRATEGY_EVOLUTION_PROMPT = """
You are tasked with improving a trading strategy based on its performance.

Current Strategy Code:
```python
{strategy_code}
```

Performance Metrics:
- Total Return: {total_return:.2%}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown:.2%}
- Win Rate: {win_rate:.2%}
- Number of Trades: {num_trades}

Issues Identified:
{issues}

Instructions:
1. Analyze the strategy's weaknesses
2. Modify the strategy to address performance issues
3. Keep the same strategy class structure
4. Improve entry/exit logic or risk management
5. Maintain conservative risk controls

Return ONLY the improved Python code for the strategy class.
"""


class BaseStrategy:
    """Base class that all generated strategies must inherit from."""

    def __init__(self):
        self.name = "BaseStrategy"
        self.positions = {}
        self.last_signal = None

    def generate_signal(
        self,
        data: Dict,
        predictions: Optional[Dict] = None,
        sentiment: Optional[Dict] = None
    ) -> Dict:
        """
        Generate trading signal based on market data.

        Args:
            data: Market data with indicators
            predictions: Price predictions
            sentiment: Sentiment analysis results

        Returns:
            Signal dictionary with action, confidence, quantity, reason
        """
        raise NotImplementedError("Subclasses must implement generate_signal")

    def validate_signal(self, signal: Dict) -> bool:
        """Validate signal format and values."""
        required_keys = ['action', 'confidence', 'quantity', 'reason']

        if not all(key in signal for key in required_keys):
            return False

        if signal['action'] not in ['buy', 'sell', 'hold']:
            return False

        if not 0 <= signal['confidence'] <= 1:
            return False

        if not 0 <= signal['quantity'] <= 1:
            return False

        return True


class LLMStrategyGenerator:
    """Generate and evolve trading strategies using LLMs."""

    def __init__(
        self,
        api_key: str,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7
    ):
        """
        Initialize the strategy generator.

        Args:
            api_key: API key for LLM provider
            provider: 'anthropic' or 'openai'
            model: Model name
            temperature: Generation temperature
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Provider {provider} not yet implemented")

        self.generated_strategies = []

        logger.info(f"Initialized LLM Strategy Generator with {provider}/{model}")

    def generate_strategy(
        self,
        context: Dict,
        strategy_name: Optional[str] = None
    ) -> str:
        """
        Generate a new trading strategy using LLM.

        Args:
            context: Market context and requirements
            strategy_name: Name for the strategy

        Returns:
            Generated strategy code
        """
        if strategy_name is None:
            strategy_name = f"Strategy_{len(self.generated_strategies) + 1}"

        # Build prompt
        prompt = STRATEGY_GENERATION_PROMPT.format(
            symbol=context.get('symbol', 'UNKNOWN'),
            market_regime=context.get('market_regime', 'unknown'),
            trend=context.get('trend', 'unknown'),
            volatility=context.get('volatility', 'medium'),
            performance_summary=context.get('performance_summary', 'N/A'),
            available_indicators=context.get('available_indicators', 'SMA, EMA, RSI, MACD, BB'),
            sentiment_score=context.get('sentiment_score', 'neutral'),
            prediction_summary=context.get('prediction_summary', 'N/A'),
            strategy_name=strategy_name
        )

        try:
            logger.info(f"Generating strategy: {strategy_name}")

            # Call LLM
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=self.temperature,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                strategy_code = response.content[0].text

            # Extract code from response
            strategy_code = self._extract_code(strategy_code)

            # Validate strategy
            if not self._validate_strategy(strategy_code):
                logger.error("Generated strategy failed validation")
                return None

            self.generated_strategies.append({
                'name': strategy_name,
                'code': strategy_code,
                'timestamp': datetime.now().isoformat(),
                'context': context
            })

            logger.info(f"Successfully generated strategy: {strategy_name}")
            return strategy_code

        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return None

    def evolve_strategy(
        self,
        strategy_code: str,
        performance_metrics: Dict,
        issues: List[str] = None
    ) -> str:
        """
        Evolve/improve a strategy based on performance feedback.

        Args:
            strategy_code: Current strategy code
            performance_metrics: Performance metrics
            issues: List of identified issues

        Returns:
            Improved strategy code
        """
        if issues is None:
            issues = self._identify_issues(performance_metrics)

        issues_text = "\n".join(f"- {issue}" for issue in issues)

        prompt = STRATEGY_EVOLUTION_PROMPT.format(
            strategy_code=strategy_code,
            total_return=performance_metrics.get('total_return', 0),
            sharpe_ratio=performance_metrics.get('sharpe_ratio', 0),
            max_drawdown=performance_metrics.get('max_drawdown', 0),
            win_rate=performance_metrics.get('win_rate', 0),
            num_trades=performance_metrics.get('num_trades', 0),
            issues=issues_text
        )

        try:
            logger.info("Evolving strategy based on performance")

            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=self.temperature,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                improved_code = response.content[0].text

            improved_code = self._extract_code(improved_code)

            if not self._validate_strategy(improved_code):
                logger.error("Evolved strategy failed validation")
                return strategy_code  # Return original

            logger.info("Successfully evolved strategy")
            return improved_code

        except Exception as e:
            logger.error(f"Strategy evolution failed: {e}")
            return strategy_code

    def generate_strategy_variants(
        self,
        base_strategy: str,
        n_variants: int = 5,
        context: Dict = None
    ) -> List[str]:
        """
        Create variations of a base strategy for testing.

        Args:
            base_strategy: Base strategy code
            n_variants: Number of variants to create
            context: Market context

        Returns:
            List of strategy code variants
        """
        variants = []

        for i in range(n_variants):
            # Generate slight variations by modifying context
            variant_context = context.copy() if context else {}
            variant_context['temperature'] = 0.5 + i * 0.1

            # This is simplified - in practice you'd modify specific aspects
            variant_name = f"Variant_{i+1}"

            variant_code = self.generate_strategy(variant_context, variant_name)

            if variant_code:
                variants.append(variant_code)

        logger.info(f"Generated {len(variants)} strategy variants")
        return variants

    def explain_strategy(self, strategy_code: str) -> str:
        """
        Generate human-readable explanation of a strategy.

        Args:
            strategy_code: Strategy code to explain

        Returns:
            Natural language explanation
        """
        prompt = f"""
        Explain this trading strategy in simple terms that a non-programmer can understand:

        ```python
        {strategy_code}
        ```

        Describe:
        1. What market conditions it looks for
        2. When it enters positions
        3. When it exits positions
        4. How it manages risk
        5. What makes this strategy unique

        Keep the explanation concise (3-5 sentences).
        """

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                explanation = response.content[0].text

            return explanation

        except Exception as e:
            logger.error(f"Strategy explanation failed: {e}")
            return "Unable to generate explanation"

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, try to find class definition
        class_pattern = r"(class \w+\(BaseStrategy\):.*)"
        matches = re.findall(class_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Return as-is if nothing found
        return response.strip()

    def _validate_strategy(self, strategy_code: str) -> bool:
        """
        Validate generated strategy code for safety and correctness.

        Args:
            strategy_code: Strategy code to validate

        Returns:
            True if valid, False otherwise
        """
        # Check for dangerous operations
        dangerous_keywords = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',
            'file(',
            'input(',
            'raw_input('
        ]

        for keyword in dangerous_keywords:
            if keyword in strategy_code:
                logger.error(f"Dangerous keyword found: {keyword}")
                return False

        # Check for required elements
        required_elements = [
            'class ',
            'BaseStrategy',
            'def generate_signal',
            'return'
        ]

        for element in required_elements:
            if element not in strategy_code:
                logger.error(f"Required element missing: {element}")
                return False

        # Try to parse as valid Python
        try:
            ast.parse(strategy_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False

        logger.debug("Strategy validation passed")
        return True

    def _identify_issues(self, performance_metrics: Dict) -> List[str]:
        """Identify issues based on performance metrics."""
        issues = []

        if performance_metrics.get('sharpe_ratio', 0) < 0.5:
            issues.append("Low risk-adjusted returns (Sharpe ratio < 0.5)")

        if performance_metrics.get('max_drawdown', 0) < -0.2:
            issues.append("Excessive drawdown (> 20%)")

        if performance_metrics.get('win_rate', 0) < 0.4:
            issues.append("Low win rate (< 40%)")

        if performance_metrics.get('num_trades', 0) < 10:
            issues.append("Too few trades for statistical significance")

        if performance_metrics.get('num_trades', 0) > 1000:
            issues.append("Overtrading - too many trades")

        return issues

    def save_strategy(self, strategy_code: str, filename: str) -> None:
        """Save strategy to file."""
        try:
            with open(filename, 'w') as f:
                f.write(strategy_code)
            logger.info(f"Strategy saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save strategy: {e}")

    def load_strategy(self, filename: str) -> str:
        """Load strategy from file."""
        try:
            with open(filename, 'r') as f:
                strategy_code = f.read()
            logger.info(f"Strategy loaded from {filename}")
            return strategy_code
        except Exception as e:
            logger.error(f"Failed to load strategy: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        exit(1)

    generator = LLMStrategyGenerator(api_key=api_key)

    # Generate a strategy
    context = {
        'symbol': 'AAPL',
        'market_regime': 'trending',
        'trend': 'uptrend',
        'volatility': 'medium',
        'available_indicators': 'RSI, MACD, Bollinger Bands, SMA',
        'sentiment_score': 'positive',
        'prediction_summary': 'Price expected to rise 2% in next day'
    }

    strategy_code = generator.generate_strategy(context, "MomentumStrategy")

    if strategy_code:
        print("Generated Strategy:")
        print(strategy_code)

        # Explain the strategy
        explanation = generator.explain_strategy(strategy_code)
        print("\nStrategy Explanation:")
        print(explanation)
