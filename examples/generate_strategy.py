"""
Strategy Generation Example
Demonstrates how to generate trading strategies using LLM.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.llm_strategy_generator import LLMStrategyGenerator
from src.data.technical_indicators import TechnicalIndicatorEngine


def main():
    print("=" * 80)
    print("LLM Strategy Generation Example")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n❌ Error: ANTHROPIC_API_KEY not found in environment variables")
        print("   Please set it in your .env file or environment")
        return

    # Initialize strategy generator
    print("\n1. Initializing LLM Strategy Generator...")
    generator = LLMStrategyGenerator(
        api_key=api_key,
        provider='anthropic',
        model='claude-sonnet-4-5-20250929',
        temperature=0.7
    )
    print("   ✓ Generator initialized")

    # Get market context
    print("\n2. Analyzing market conditions...")
    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    regime = engine.get_market_regime('AAPL')
    print(f"   Market Regime: {regime}")

    # Define strategy requirements
    context = {
        'symbol': 'AAPL',
        'market_regime': regime.get('trend', 'unknown'),
        'trend': regime.get('trend', 'unknown'),
        'volatility': regime.get('volatility', 'medium'),
        'available_indicators': 'RSI, MACD, Bollinger Bands, SMA, EMA, ATR, OBV',
        'sentiment_score': 'neutral',
        'performance_summary': 'Generate a robust momentum strategy',
        'prediction_summary': 'N/A'
    }

    # Generate strategy
    print("\n3. Generating trading strategy...")
    strategy_code = generator.generate_strategy(
        context=context,
        strategy_name="MomentumStrategy"
    )

    if strategy_code:
        print("   ✓ Strategy generated successfully!")

        print("\n" + "=" * 80)
        print("GENERATED STRATEGY CODE")
        print("=" * 80)
        print(strategy_code)
        print("=" * 80)

        # Generate explanation
        print("\n4. Generating strategy explanation...")
        explanation = generator.explain_strategy(strategy_code)
        print("\n" + "=" * 80)
        print("STRATEGY EXPLANATION")
        print("=" * 80)
        print(explanation)
        print("=" * 80)

        # Save strategy
        output_path = Path(__file__).parent.parent / "strategies" / "generated_momentum_strategy.py"
        output_path.parent.mkdir(exist_ok=True)

        generator.save_strategy(strategy_code, str(output_path))
        print(f"\n✓ Strategy saved to: {output_path}")

    else:
        print("   ❌ Strategy generation failed")


if __name__ == "__main__":
    main()
