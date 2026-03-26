"""
Data Validation and Quality Checks
Comprehensive validation for market data before use in trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    metrics: Dict


class DataValidator:
    """
    Comprehensive data validation for trading data.

    Checks for:
    - Missing data and gaps
    - Outliers and anomalies
    - Data consistency
    - Time series properties
    - Volume anomalies
    """

    def __init__(
        self,
        max_gap_tolerance: int = 5,
        outlier_std_threshold: float = 5.0,
        min_data_points: int = 50
    ):
        """
        Initialize data validator.

        Args:
            max_gap_tolerance: Maximum acceptable gap in trading days
            outlier_std_threshold: Standard deviations for outlier detection
            min_data_points: Minimum required data points
        """
        self.max_gap_tolerance = max_gap_tolerance
        self.outlier_std_threshold = outlier_std_threshold
        self.min_data_points = min_data_points

        logger.info("Initialized DataValidator")

    def validate(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationResult:
        """
        Perform comprehensive validation on market data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being validated

        Returns:
            ValidationResult with findings
        """
        issues = []
        warnings = []
        metrics = {}

        logger.info(f"Validating data for {symbol}: {len(df)} rows")

        # 1. Basic checks
        if df.empty:
            issues.append("DataFrame is empty")
            return ValidationResult(False, issues, warnings, metrics)

        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data: {len(df)} < {self.min_data_points}")

        # 2. Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, issues, warnings, metrics)

        # 3. Check for missing values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            for col, count in null_counts.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    if pct > 5:
                        issues.append(f"{col}: {count} missing values ({pct:.1f}%)")
                    else:
                        warnings.append(f"{col}: {count} missing values ({pct:.1f}%)")

        metrics['null_counts'] = null_counts.to_dict()

        # 4. Check for time gaps
        if hasattr(df.index, 'to_series'):
            gaps = self._check_time_gaps(df)
            if gaps['max_gap_days'] > self.max_gap_tolerance:
                issues.append(
                    f"Large time gap detected: {gaps['max_gap_days']} days "
                    f"(threshold: {self.max_gap_tolerance})"
                )
            if gaps['num_gaps'] > 0:
                warnings.append(f"Found {gaps['num_gaps']} time gaps")

            metrics['time_gaps'] = gaps

        # 5. Check OHLC consistency
        ohlc_issues = self._check_ohlc_consistency(df)
        if ohlc_issues:
            for issue in ohlc_issues:
                warnings.append(f"OHLC inconsistency: {issue}")

        metrics['ohlc_consistency_issues'] = len(ohlc_issues)

        # 6. Check for price outliers
        outliers = self._detect_price_outliers(df)
        if outliers['num_outliers'] > 0:
            pct = (outliers['num_outliers'] / len(df)) * 100
            if pct > 1:
                issues.append(
                    f"High number of price outliers: {outliers['num_outliers']} ({pct:.1f}%)"
                )
            else:
                warnings.append(
                    f"Price outliers detected: {outliers['num_outliers']} ({pct:.1f}%)"
                )

        metrics['outliers'] = outliers

        # 7. Check volume anomalies (if volume data available)
        if 'volume' in df.columns:
            volume_issues = self._check_volume_anomalies(df)
            if volume_issues['zero_volume_pct'] > 5:
                warnings.append(
                    f"High zero volume: {volume_issues['zero_volume_pct']:.1f}%"
                )

            metrics['volume_anomalies'] = volume_issues

        # 8. Check data recency
        if hasattr(df.index, 'max'):
            last_date = df.index.max()
            if isinstance(last_date, pd.Timestamp):
                days_old = (pd.Timestamp.now() - last_date).days
                if days_old > 7:
                    warnings.append(f"Data is {days_old} days old")

                metrics['data_age_days'] = days_old

        # 9. Statistical properties
        stats = self._calculate_statistics(df)
        metrics['statistics'] = stats

        # Determine if valid
        is_valid = len(issues) == 0

        # Log results
        if is_valid:
            logger.info(f"✓ Validation passed for {symbol} ({len(warnings)} warnings)")
        else:
            logger.warning(f"✗ Validation failed for {symbol}: {len(issues)} issues")

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )

    def _check_time_gaps(self, df: pd.DataFrame) -> Dict:
        """Check for gaps in time series."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'num_gaps': 0, 'max_gap_days': 0}

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Find gaps (assuming daily data, gaps > 3 days are suspicious)
        gaps = time_diffs[time_diffs > pd.Timedelta(days=3)]

        if len(gaps) > 0:
            max_gap = gaps.max()
            max_gap_days = max_gap.days
        else:
            max_gap_days = 0

        return {
            'num_gaps': len(gaps),
            'max_gap_days': max_gap_days,
            'gap_locations': gaps.index.tolist()
        }

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check OHLC relationships are consistent."""
        issues = []

        # High should be >= Low
        invalid_high_low = (df['high'] < df['low']).sum()
        if invalid_high_low > 0:
            issues.append(f"{invalid_high_low} rows where high < low")

        # High should be >= Open and Close
        invalid_high_open = (df['high'] < df['open']).sum()
        if invalid_high_open > 0:
            issues.append(f"{invalid_high_open} rows where high < open")

        invalid_high_close = (df['high'] < df['close']).sum()
        if invalid_high_close > 0:
            issues.append(f"{invalid_high_close} rows where high < close")

        # Low should be <= Open and Close
        invalid_low_open = (df['low'] > df['open']).sum()
        if invalid_low_open > 0:
            issues.append(f"{invalid_low_open} rows where low > open")

        invalid_low_close = (df['low'] > df['close']).sum()
        if invalid_low_close > 0:
            issues.append(f"{invalid_low_close} rows where low > close")

        # Check for negative prices
        negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()
        if negative_prices > 0:
            issues.append(f"{negative_prices} rows with negative prices")

        # Check for zero prices
        zero_prices = (df[['open', 'high', 'low', 'close']] == 0).any(axis=1).sum()
        if zero_prices > 0:
            issues.append(f"{zero_prices} rows with zero prices")

        return issues

    def _detect_price_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect price outliers using statistical methods."""
        # Calculate returns
        returns = df['close'].pct_change().dropna()

        # Z-score method
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return > 0:
            z_scores = np.abs((returns - mean_return) / std_return)
            outliers = z_scores > self.outlier_std_threshold
            num_outliers = outliers.sum()
            outlier_indices = returns[outliers].index.tolist()
        else:
            num_outliers = 0
            outlier_indices = []

        # IQR method
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        iqr_outliers = ((returns < lower_bound) | (returns > upper_bound)).sum()

        return {
            'num_outliers': int(num_outliers),
            'iqr_outliers': int(iqr_outliers),
            'outlier_indices': outlier_indices[:10],  # First 10
            'max_return': float(returns.max()),
            'min_return': float(returns.min())
        }

    def _check_volume_anomalies(self, df: pd.DataFrame) -> Dict:
        """Check for volume anomalies."""
        volume = df['volume']

        # Zero volume
        zero_volume = (volume == 0).sum()
        zero_volume_pct = (zero_volume / len(df)) * 100

        # Negative volume
        negative_volume = (volume < 0).sum()

        # Volume spikes
        if len(volume) > 1:
            volume_ma = volume.rolling(window=20, min_periods=1).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            spikes = (volume_ratio > 10).sum()  # 10x average
        else:
            spikes = 0

        return {
            'zero_volume_count': int(zero_volume),
            'zero_volume_pct': float(zero_volume_pct),
            'negative_volume_count': int(negative_volume),
            'volume_spikes': int(spikes)
        }

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical properties of the data."""
        returns = df['close'].pct_change().dropna()

        stats = {
            'num_rows': len(df),
            'mean_price': float(df['close'].mean()),
            'std_price': float(df['close'].std()),
            'min_price': float(df['close'].min()),
            'max_price': float(df['close'].max()),
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis())
        }

        if 'volume' in df.columns:
            stats['mean_volume'] = float(df['volume'].mean())
            stats['total_volume'] = float(df['volume'].sum())

        return stats

    def generate_report(self, result: ValidationResult, symbol: str = "UNKNOWN") -> str:
        """
        Generate a human-readable validation report.

        Args:
            result: ValidationResult object
            symbol: Symbol name

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"DATA VALIDATION REPORT: {symbol}")
        lines.append("=" * 70)
        lines.append(f"Status: {'✓ PASSED' if result.is_valid else '✗ FAILED'}")
        lines.append(f"Issues: {len(result.issues)}")
        lines.append(f"Warnings: {len(result.warnings)}")
        lines.append("")

        if result.issues:
            lines.append("ISSUES:")
            for i, issue in enumerate(result.issues, 1):
                lines.append(f"  {i}. {issue}")
            lines.append("")

        if result.warnings:
            lines.append("WARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                lines.append(f"  {i}. {warning}")
            lines.append("")

        # Statistics
        if 'statistics' in result.metrics:
            stats = result.metrics['statistics']
            lines.append("STATISTICS:")
            lines.append(f"  Rows: {stats.get('num_rows', 0)}")
            lines.append(f"  Price Range: ${stats.get('min_price', 0):.2f} - ${stats.get('max_price', 0):.2f}")
            lines.append(f"  Mean Return: {stats.get('mean_return', 0):.4f}")
            lines.append(f"  Return Volatility: {stats.get('std_return', 0):.4f}")
            lines.append("")

        # Outliers
        if 'outliers' in result.metrics:
            outliers = result.metrics['outliers']
            lines.append("OUTLIERS:")
            lines.append(f"  Detected: {outliers.get('num_outliers', 0)}")
            lines.append(f"  Max Return: {outliers.get('max_return', 0):.2%}")
            lines.append(f"  Min Return: {outliers.get('min_return', 0):.2%}")
            lines.append("")

        # Time gaps
        if 'time_gaps' in result.metrics:
            gaps = result.metrics['time_gaps']
            lines.append("TIME GAPS:")
            lines.append(f"  Number of gaps: {gaps.get('num_gaps', 0)}")
            lines.append(f"  Max gap: {gaps.get('max_gap_days', 0)} days")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


class DataQualityMonitor:
    """
    Monitor data quality over time and track degradation.
    """

    def __init__(self):
        """Initialize data quality monitor."""
        self.validation_history = []
        logger.info("Initialized DataQualityMonitor")

    def add_validation(
        self,
        result: ValidationResult,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add validation result to history.

        Args:
            result: ValidationResult
            symbol: Symbol name
            timestamp: Validation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.validation_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'is_valid': result.is_valid,
            'num_issues': len(result.issues),
            'num_warnings': len(result.warnings),
            'metrics': result.metrics
        })

    def get_quality_trend(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Get data quality trend for a symbol.

        Args:
            symbol: Symbol to analyze
            lookback_days: Days to look back

        Returns:
            Quality trend metrics
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)

        recent = [
            v for v in self.validation_history
            if v['symbol'] == symbol and v['timestamp'] >= cutoff
        ]

        if not recent:
            return {'status': 'no_data'}

        # Calculate metrics
        pass_rate = sum(1 for v in recent if v['is_valid']) / len(recent)
        avg_issues = np.mean([v['num_issues'] for v in recent])
        avg_warnings = np.mean([v['num_warnings'] for v in recent])

        # Trend (improving/degrading)
        if len(recent) >= 2:
            recent_issues = [v['num_issues'] for v in recent[-5:]]
            if len(recent_issues) >= 2:
                trend = 'improving' if recent_issues[-1] < recent_issues[0] else 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'pass_rate': pass_rate,
            'avg_issues': avg_issues,
            'avg_warnings': avg_warnings,
            'trend': trend,
            'num_validations': len(recent)
        }

    def export_history(self) -> pd.DataFrame:
        """Export validation history as DataFrame."""
        if not self.validation_history:
            return pd.DataFrame()

        # Flatten for DataFrame
        records = []
        for entry in self.validation_history:
            record = {
                'timestamp': entry['timestamp'],
                'symbol': entry['symbol'],
                'is_valid': entry['is_valid'],
                'num_issues': entry['num_issues'],
                'num_warnings': entry['num_warnings']
            }
            records.append(record)

        return pd.DataFrame(records)


if __name__ == "__main__":
    # Test data validator
    from src.data.technical_indicators import TechnicalIndicatorEngine

    logger.info("Testing Data Validator")

    # Fetch data
    engine = TechnicalIndicatorEngine(['AAPL'], ['1d'])
    df = engine.fetch_data('AAPL', '1d', lookback=200)

    # Validate
    validator = DataValidator()
    result = validator.validate(df, 'AAPL')

    # Print report
    report = validator.generate_report(result, 'AAPL')
    print(report)

    # Test quality monitor
    monitor = DataQualityMonitor()
    monitor.add_validation(result, 'AAPL')

    trend = monitor.get_quality_trend('AAPL')
    print(f"\nQuality Trend: {trend}")
