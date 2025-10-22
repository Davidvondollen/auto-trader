"""
News & Sentiment Analysis
Fetches news and performs sentiment analysis using FinBERT and LLMs.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment using FinBERT and optional LLM enhancement.
    """

    def __init__(self, use_llm: bool = False, llm_client=None):
        """
        Initialize sentiment analyzer.

        Args:
            use_llm: Whether to use LLM for event extraction
            llm_client: LLM client for advanced analysis
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.sentiment_model = None

        # Try to load FinBERT
        try:
            logger.info("Loading FinBERT sentiment model...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✓ FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}")
            self.sentiment_pipeline = None

    def fetch_news(
        self,
        symbols: List[str],
        api_key: str,
        lookback_days: int = 7,
        max_articles: int = 100
    ) -> pd.DataFrame:
        """
        Fetch news articles for symbols.

        Args:
            symbols: List of symbols
            api_key: NewsAPI key
            lookback_days: Days to look back
            max_articles: Maximum articles per symbol

        Returns:
            DataFrame with articles
        """
        all_articles = []

        for symbol in symbols:
            try:
                articles = self._fetch_newsapi(symbol, api_key, lookback_days, max_articles)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")

        if not all_articles:
            logger.warning("No articles fetched")
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        logger.info(f"Fetched {len(df)} articles for {len(symbols)} symbols")

        return df

    def _fetch_newsapi(
        self,
        symbol: str,
        api_key: str,
        lookback_days: int,
        max_articles: int
    ) -> List[Dict]:
        """Fetch news from NewsAPI."""
        url = "https://newsapi.org/v2/everything"

        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=lookback_days)

        params = {
            'q': symbol,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': min(max_articles, 100),
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d')
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'symbol': symbol,
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })

            return articles

        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

    def analyze_sentiment(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of articles using FinBERT.

        Args:
            articles_df: DataFrame with articles

        Returns:
            DataFrame with sentiment scores
        """
        if articles_df.empty:
            return articles_df

        if not self.sentiment_pipeline:
            logger.warning("Sentiment model not available, using simple heuristics")
            return self._simple_sentiment(articles_df)

        sentiments = []

        for idx, row in articles_df.iterrows():
            try:
                # Combine title and description
                text = f"{row.get('title', '')} {row.get('description', '')}"

                if not text.strip():
                    sentiments.append({
                        'sentiment': 'neutral',
                        'score': 0.0,
                        'confidence': 0.0
                    })
                    continue

                # Truncate to model max length
                text = text[:512]

                # Get sentiment
                result = self.sentiment_pipeline(text)[0]

                # Map FinBERT labels
                label_map = {
                    'positive': 'positive',
                    'negative': 'negative',
                    'neutral': 'neutral'
                }

                sentiment = label_map.get(result['label'].lower(), 'neutral')
                confidence = result['score']

                # Convert to numeric score (-1 to 1)
                if sentiment == 'positive':
                    score = confidence
                elif sentiment == 'negative':
                    score = -confidence
                else:
                    score = 0.0

                sentiments.append({
                    'sentiment': sentiment,
                    'score': score,
                    'confidence': confidence
                })

            except Exception as e:
                logger.error(f"Sentiment analysis failed for article {idx}: {e}")
                sentiments.append({
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0
                })

        # Add sentiment columns
        sentiment_df = pd.DataFrame(sentiments)
        result_df = pd.concat([articles_df.reset_index(drop=True), sentiment_df], axis=1)

        logger.info(f"Analyzed sentiment for {len(result_df)} articles")

        return result_df

    def _simple_sentiment(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """Simple keyword-based sentiment analysis."""
        positive_words = [
            'gain', 'rise', 'up', 'surge', 'bullish', 'profit', 'growth',
            'strong', 'beat', 'exceed', 'rally', 'boom', 'success'
        ]

        negative_words = [
            'loss', 'fall', 'down', 'drop', 'bearish', 'decline', 'weak',
            'miss', 'concern', 'risk', 'crash', 'slump', 'fail'
        ]

        sentiments = []

        for idx, row in articles_df.iterrows():
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count > neg_count:
                sentiment = 'positive'
                score = min(pos_count / 10, 1.0)
            elif neg_count > pos_count:
                sentiment = 'negative'
                score = -min(neg_count / 10, 1.0)
            else:
                sentiment = 'neutral'
                score = 0.0

            sentiments.append({
                'sentiment': sentiment,
                'score': score,
                'confidence': min(abs(pos_count - neg_count) / 5, 1.0)
            })

        sentiment_df = pd.DataFrame(sentiments)
        result_df = pd.concat([articles_df.reset_index(drop=True), sentiment_df], axis=1)

        return result_df

    def aggregate_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        by: str = 'symbol'
    ) -> Dict[str, Dict]:
        """
        Aggregate sentiment scores by symbol or time.

        Args:
            sentiment_df: DataFrame with sentiment scores
            by: Aggregation key ('symbol' or 'date')

        Returns:
            Dictionary of aggregated sentiment
        """
        if sentiment_df.empty:
            return {}

        aggregated = {}

        for key, group in sentiment_df.groupby(by):
            sentiment_scores = group['score'].values

            aggregated[key] = {
                'mean_sentiment': float(np.mean(sentiment_scores)),
                'sentiment_std': float(np.std(sentiment_scores)),
                'positive_ratio': float(np.sum(sentiment_scores > 0) / len(sentiment_scores)),
                'negative_ratio': float(np.sum(sentiment_scores < 0) / len(sentiment_scores)),
                'neutral_ratio': float(np.sum(sentiment_scores == 0) / len(sentiment_scores)),
                'num_articles': len(group),
                'max_sentiment': float(np.max(sentiment_scores)),
                'min_sentiment': float(np.min(sentiment_scores)),
                'recent_sentiment': float(sentiment_scores[-1]) if len(sentiment_scores) > 0 else 0.0
            }

        return aggregated

    def extract_events(self, articles_df: pd.DataFrame) -> List[Dict]:
        """
        Extract key events from articles using LLM.

        Args:
            articles_df: DataFrame with articles

        Returns:
            List of extracted events
        """
        if not self.use_llm or not self.llm_client:
            logger.warning("LLM not available for event extraction")
            return []

        events = []

        # This would use the LLM to extract structured events
        # For now, return placeholder
        logger.info("Event extraction requires LLM integration")

        return events

    def get_sentiment_signal(
        self,
        symbol: str,
        sentiment_df: pd.DataFrame,
        threshold: float = 0.3
    ) -> Dict:
        """
        Generate trading signal from sentiment.

        Args:
            symbol: Symbol to analyze
            sentiment_df: DataFrame with sentiment
            threshold: Threshold for signal generation

        Returns:
            Signal dictionary
        """
        symbol_articles = sentiment_df[sentiment_df['symbol'] == symbol]

        if symbol_articles.empty:
            return {
                'signal': 'neutral',
                'strength': 0.0,
                'confidence': 0.0
            }

        mean_sentiment = symbol_articles['score'].mean()
        confidence = symbol_articles['confidence'].mean()

        if mean_sentiment > threshold:
            signal = 'bullish'
            strength = min(mean_sentiment, 1.0)
        elif mean_sentiment < -threshold:
            signal = 'bearish'
            strength = min(abs(mean_sentiment), 1.0)
        else:
            signal = 'neutral'
            strength = 0.0

        return {
            'symbol': symbol,
            'signal': signal,
            'strength': float(strength),
            'confidence': float(confidence),
            'num_articles': len(symbol_articles),
            'mean_sentiment': float(mean_sentiment)
        }


if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = NewsSentimentAnalyzer(use_llm=False)

    # Create sample articles
    sample_articles = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'title': 'Apple stock surges on strong earnings',
            'description': 'Apple beats expectations with record revenue',
            'content': 'Apple reported strong quarterly results...',
            'published_at': datetime.now().isoformat(),
            'source': 'TechNews'
        },
        {
            'symbol': 'AAPL',
            'title': 'Apple faces concerns over supply chain',
            'description': 'Supply chain issues may impact future growth',
            'content': 'Industry experts worry about...',
            'published_at': datetime.now().isoformat(),
            'source': 'BusinessDaily'
        }
    ])

    # Analyze sentiment
    sentiment_df = analyzer.analyze_sentiment(sample_articles)
    print("\nSentiment Analysis:")
    print(sentiment_df[['symbol', 'title', 'sentiment', 'score', 'confidence']])

    # Aggregate sentiment
    aggregated = analyzer.aggregate_sentiment(sentiment_df)
    print("\nAggregated Sentiment:")
    print(aggregated)

    # Get trading signal
    signal = analyzer.get_sentiment_signal('AAPL', sentiment_df)
    print("\nTrading Signal:")
    print(signal)
