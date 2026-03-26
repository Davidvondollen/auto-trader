"""
Monitoring and Alerting System
Production observability for trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'resolved': self.resolved
        }


class MetricsCollector:
    """
    Collect and aggregate system metrics.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {}
        self.metric_history = {}
        logger.info("Initialized MetricsCollector")

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            timestamp: Metric timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Store latest value
        self.metrics[name] = {
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        }

        # Store in history
        if name not in self.metric_history:
            self.metric_history[name] = []

        self.metric_history[name].append({
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        })

        # Limit history size
        if len(self.metric_history[name]) > 10000:
            self.metric_history[name] = self.metric_history[name][-5000:]

    def get_metric(self, name: str) -> Optional[float]:
        """Get latest metric value."""
        if name in self.metrics:
            return self.metrics[name]['value']
        return None

    def get_metric_history(
        self,
        name: str,
        lookback_minutes: int = 60
    ) -> List[Dict]:
        """
        Get metric history.

        Args:
            name: Metric name
            lookback_minutes: Minutes to look back

        Returns:
            List of metric records
        """
        if name not in self.metric_history:
            return []

        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        return [
            m for m in self.metric_history[name]
            if m['timestamp'] >= cutoff
        ]

    def get_metric_stats(self, name: str, lookback_minutes: int = 60) -> Dict:
        """
        Get statistical summary of metric.

        Args:
            name: Metric name
            lookback_minutes: Minutes to look back

        Returns:
            Statistics dictionary
        """
        history = self.get_metric_history(name, lookback_minutes)

        if not history:
            return {}

        values = [m['value'] for m in history]

        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'latest': values[-1] if values else None
        }

    def export_metrics(self) -> Dict:
        """Export all current metrics."""
        return {
            name: data['value']
            for name, data in self.metrics.items()
        }


class AlertManager:
    """
    Manage alerts and notifications.
    """

    def __init__(self, alerts_file: Optional[str] = None):
        """
        Initialize alert manager.

        Args:
            alerts_file: Optional file to persist alerts
        """
        self.alerts: List[Alert] = []
        self.handlers: List[Callable] = []
        self.alerts_file = alerts_file

        if alerts_file:
            self._load_alerts()

        logger.info("Initialized AlertManager")

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Add alert handler (callback function).

        Args:
            handler: Function that takes Alert as parameter
        """
        self.handlers.append(handler)

    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> Alert:
        """
        Create and dispatch an alert.

        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            metadata: Optional metadata

        Returns:
            Created alert
        """
        alert = Alert(
            level=level,
            title=title,
            message=message,
            metadata=metadata or {}
        )

        self.alerts.append(alert)

        # Log alert
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(level, logger.info)

        log_func(f"[ALERT] {title}: {message}")

        # Call handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        # Persist if configured
        if self.alerts_file:
            self._save_alerts()

        return alert

    def get_unresolved_alerts(
        self,
        min_level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """
        Get unresolved alerts.

        Args:
            min_level: Minimum alert level to include

        Returns:
            List of unresolved alerts
        """
        alerts = [a for a in self.alerts if not a.resolved]

        if min_level:
            # Filter by level
            level_order = {
                AlertLevel.INFO: 0,
                AlertLevel.WARNING: 1,
                AlertLevel.ERROR: 2,
                AlertLevel.CRITICAL: 3
            }
            min_severity = level_order.get(min_level, 0)
            alerts = [
                a for a in alerts
                if level_order.get(a.level, 0) >= min_severity
            ]

        return alerts

    def resolve_alert(self, alert: Alert) -> None:
        """Mark alert as resolved."""
        alert.resolved = True

        if self.alerts_file:
            self._save_alerts()

    def clear_resolved_alerts(self, older_than_hours: int = 24) -> int:
        """
        Clear resolved alerts older than specified time.

        Args:
            older_than_hours: Hours threshold

        Returns:
            Number of alerts cleared
        """
        cutoff = datetime.now() - timedelta(hours=older_than_hours)

        before_count = len(self.alerts)

        self.alerts = [
            a for a in self.alerts
            if not (a.resolved and a.timestamp < cutoff)
        ]

        cleared = before_count - len(self.alerts)

        if cleared > 0 and self.alerts_file:
            self._save_alerts()

        return cleared

    def _save_alerts(self) -> None:
        """Persist alerts to file."""
        if not self.alerts_file:
            return

        try:
            alerts_data = [a.to_dict() for a in self.alerts[-1000:]]  # Keep last 1000

            Path(self.alerts_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _load_alerts(self) -> None:
        """Load alerts from file."""
        if not self.alerts_file or not Path(self.alerts_file).exists():
            return

        try:
            with open(self.alerts_file, 'r') as f:
                alerts_data = json.load(f)

            for data in alerts_data:
                alert = Alert(
                    level=AlertLevel(data['level']),
                    title=data['title'],
                    message=data['message'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    metadata=data.get('metadata', {}),
                    resolved=data.get('resolved', False)
                )
                self.alerts.append(alert)

            logger.info(f"Loaded {len(self.alerts)} alerts from {self.alerts_file}")

        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")


class PerformanceMonitor:
    """
    Monitor trading performance and detect issues.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager
    ):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: MetricsCollector instance
            alert_manager: AlertManager instance
        """
        self.metrics = metrics_collector
        self.alerts = alert_manager

        # Thresholds
        self.max_drawdown_threshold = 0.15  # 15%
        self.min_sharpe_threshold = 0.5
        self.max_daily_loss_pct = 0.05  # 5%
        self.max_position_concentration = 0.30  # 30%

        logger.info("Initialized PerformanceMonitor")

    def check_portfolio_health(
        self,
        portfolio_value: float,
        initial_value: float,
        positions: Dict[str, Dict],
        daily_pnl: float
    ) -> List[Alert]:
        """
        Check portfolio health and generate alerts.

        Args:
            portfolio_value: Current portfolio value
            initial_value: Initial portfolio value
            positions: Current positions
            daily_pnl: Daily P&L

        Returns:
            List of generated alerts
        """
        alerts_generated = []

        # Record metrics
        self.metrics.record_metric('portfolio_value', portfolio_value)
        self.metrics.record_metric('daily_pnl', daily_pnl)

        # Check drawdown
        total_return = (portfolio_value - initial_value) / initial_value
        if total_return < -self.max_drawdown_threshold:
            alert = self.alerts.create_alert(
                AlertLevel.CRITICAL,
                "Maximum Drawdown Exceeded",
                f"Portfolio down {total_return:.2%}, threshold: {self.max_drawdown_threshold:.2%}",
                {'portfolio_value': portfolio_value, 'return': total_return}
            )
            alerts_generated.append(alert)

        # Check daily loss
        daily_loss_pct = daily_pnl / portfolio_value
        if daily_loss_pct < -self.max_daily_loss_pct:
            alert = self.alerts.create_alert(
                AlertLevel.ERROR,
                "Daily Loss Limit Exceeded",
                f"Daily loss: {daily_loss_pct:.2%}, limit: {self.max_daily_loss_pct:.2%}",
                {'daily_pnl': daily_pnl}
            )
            alerts_generated.append(alert)

        # Check position concentration
        if positions:
            total_value = sum(p['value'] for p in positions.values())
            if total_value > 0:
                max_concentration = max(p['value'] / total_value for p in positions.values())

                if max_concentration > self.max_position_concentration:
                    alert = self.alerts.create_alert(
                        AlertLevel.WARNING,
                        "High Position Concentration",
                        f"Single position: {max_concentration:.2%} of portfolio",
                        {'concentration': max_concentration}
                    )
                    alerts_generated.append(alert)

        return alerts_generated

    def check_strategy_performance(
        self,
        returns: pd.Series,
        strategy_name: str
    ) -> List[Alert]:
        """
        Check strategy performance metrics.

        Args:
            returns: Strategy returns series
            strategy_name: Strategy name

        Returns:
            List of generated alerts
        """
        alerts_generated = []

        # Calculate Sharpe ratio
        if len(returns) >= 30:  # Need enough data
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

            self.metrics.record_metric(
                f'sharpe_ratio_{strategy_name}',
                sharpe,
                tags={'strategy': strategy_name}
            )

            if sharpe < self.min_sharpe_threshold:
                alert = self.alerts.create_alert(
                    AlertLevel.WARNING,
                    f"Low Sharpe Ratio: {strategy_name}",
                    f"Sharpe ratio: {sharpe:.2f}, threshold: {self.min_sharpe_threshold:.2f}",
                    {'strategy': strategy_name, 'sharpe_ratio': sharpe}
                )
                alerts_generated.append(alert)

        # Check for consecutive losses
        if len(returns) >= 5:
            consecutive_losses = 0
            for r in returns[-10:]:
                if r < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                if consecutive_losses >= 5:
                    alert = self.alerts.create_alert(
                        AlertLevel.WARNING,
                        f"Consecutive Losses: {strategy_name}",
                        f"Strategy has {consecutive_losses} consecutive losing periods",
                        {'strategy': strategy_name, 'consecutive_losses': consecutive_losses}
                    )
                    alerts_generated.append(alert)
                    break

        return alerts_generated

    def check_data_freshness(
        self,
        last_update: datetime,
        symbol: str,
        max_age_minutes: int = 60
    ) -> Optional[Alert]:
        """
        Check if data is fresh enough.

        Args:
            last_update: Last data update timestamp
            symbol: Symbol being checked
            max_age_minutes: Maximum acceptable age in minutes

        Returns:
            Alert if data is stale
        """
        age = (datetime.now() - last_update).total_seconds() / 60

        self.metrics.record_metric(
            f'data_age_minutes_{symbol}',
            age,
            tags={'symbol': symbol}
        )

        if age > max_age_minutes:
            return self.alerts.create_alert(
                AlertLevel.WARNING,
                f"Stale Data: {symbol}",
                f"Data is {age:.0f} minutes old (threshold: {max_age_minutes})",
                {'symbol': symbol, 'age_minutes': age}
            )

        return None


class SystemHealthMonitor:
    """
    Monitor overall system health.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager
    ):
        """Initialize system health monitor."""
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.last_heartbeat = datetime.now()

        logger.info("Initialized SystemHealthMonitor")

    def heartbeat(self) -> None:
        """Record system heartbeat."""
        self.last_heartbeat = datetime.now()
        self.metrics.record_metric('system_heartbeat', 1)

    def check_health(self) -> Dict:
        """
        Check overall system health.

        Returns:
            Health status dictionary
        """
        # Check heartbeat
        heartbeat_age = (datetime.now() - self.last_heartbeat).total_seconds()

        if heartbeat_age > 300:  # 5 minutes
            self.alerts.create_alert(
                AlertLevel.CRITICAL,
                "System Heartbeat Lost",
                f"No heartbeat for {heartbeat_age:.0f} seconds",
                {'heartbeat_age': heartbeat_age}
            )

        # Count unresolved alerts by level
        unresolved = self.alerts.get_unresolved_alerts()
        alert_counts = {
            'critical': sum(1 for a in unresolved if a.level == AlertLevel.CRITICAL),
            'error': sum(1 for a in unresolved if a.level == AlertLevel.ERROR),
            'warning': sum(1 for a in unresolved if a.level == AlertLevel.WARNING),
            'info': sum(1 for a in unresolved if a.level == AlertLevel.INFO)
        }

        # Overall health status
        if alert_counts['critical'] > 0:
            status = 'critical'
        elif alert_counts['error'] > 0:
            status = 'degraded'
        elif alert_counts['warning'] > 0:
            status = 'warning'
        else:
            status = 'healthy'

        return {
            'status': status,
            'heartbeat_age_seconds': heartbeat_age,
            'unresolved_alerts': alert_counts,
            'total_unresolved': len(unresolved),
            'last_check': datetime.now()
        }


if __name__ == "__main__":
    # Test monitoring system
    logger.info("Testing Monitoring System")

    # Create components
    metrics = MetricsCollector()
    alerts = AlertManager(alerts_file='logs/alerts.json')
    perf_monitor = PerformanceMonitor(metrics, alerts)
    health_monitor = SystemHealthMonitor(metrics, alerts)

    # Add alert handler
    def print_alert(alert: Alert):
        print(f"\n[{alert.level.value.upper()}] {alert.title}")
        print(f"  {alert.message}")

    alerts.add_handler(print_alert)

    # Record some metrics
    metrics.record_metric('portfolio_value', 100000)
    metrics.record_metric('daily_pnl', -2500)

    # Simulate alert
    perf_monitor.check_portfolio_health(
        portfolio_value=97500,
        initial_value=100000,
        positions={'AAPL': {'value': 50000}, 'GOOGL': {'value': 47500}},
        daily_pnl=-2500
    )

    # Check health
    health = health_monitor.check_health()
    print(f"\nSystem Health: {health['status']}")
    print(f"Unresolved Alerts: {health['total_unresolved']}")
