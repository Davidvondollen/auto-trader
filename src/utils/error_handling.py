"""
Error Handling and Resilience Utilities
Retry logic, circuit breakers, and graceful degradation.
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

        logger.info(f"Initialized CircuitBreaker: threshold={failure_threshold}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            # Check if we should try recovery
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker: transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # Success - reset if we were testing
            if self.state == CircuitState.HALF_OPEN:
                self._reset()

            return result

        except self.expected_exception as e:
            self._record_failure()
            raise e

    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout

    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        logger.info("Circuit breaker CLOSED (recovered)")

    def get_state(self) -> Dict:
        """Get current circuit breaker state."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time
        }


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback on retry (receives attempt number)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts"
                        )
                        raise e

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(attempt)

                    time.sleep(current_delay)
                    current_delay *= backoff

            # Should not reach here
            raise last_exception

        return wrapper
    return decorator


def timeout(seconds: float) -> Callable:
    """
    Timeout decorator using threading.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import threading

            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise TimeoutError(f"Function timed out after {seconds}s")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


def fallback(fallback_func: Callable) -> Callable:
    """
    Fallback decorator - calls fallback function on exception.

    Args:
        fallback_func: Fallback function to call

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Function {func.__name__} failed: {e}. Using fallback."
                )
                return fallback_func(*args, **kwargs)

        return wrapper
    return decorator


class ErrorHandler:
    """
    Centralized error handling with logging and recovery.
    """

    def __init__(self):
        """Initialize error handler."""
        self.error_counts = {}
        self.last_errors = {}

    def handle(
        self,
        exception: Exception,
        context: str = "",
        raise_exception: bool = False
    ) -> None:
        """
        Handle an exception with logging and tracking.

        Args:
            exception: Exception to handle
            context: Context string for logging
            raise_exception: Whether to re-raise after handling
        """
        error_type = type(exception).__name__

        # Track error count
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Store last occurrence
        self.last_errors[error_type] = {
            'exception': exception,
            'context': context,
            'timestamp': datetime.now()
        }

        # Log error
        logger.error(
            f"[{context}] {error_type}: {str(exception)}"
            f" (count: {self.error_counts[error_type]})"
        )

        if raise_exception:
            raise exception

    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'errors_by_type': self.error_counts.copy(),
            'last_errors': {
                k: {
                    'message': str(v['exception']),
                    'context': v['context'],
                    'timestamp': v['timestamp'].isoformat()
                }
                for k, v in self.last_errors.items()
            }
        }

    def clear_stats(self) -> None:
        """Clear error statistics."""
        self.error_counts.clear()
        self.last_errors.clear()


class APIRateLimiter:
    """
    Rate limiter for API calls.
    """

    def __init__(self, max_calls: int, period_seconds: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed
            period_seconds: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = []

        logger.info(f"Initialized RateLimiter: {max_calls} calls per {period_seconds}s")

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            result = func(*args, **kwargs)
            self.calls.append(time.time())
            return result
        return wrapper

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove old calls
        self.calls = [c for c in self.calls if now - c < self.period]

        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.period - (now - oldest_call)

            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

                # Remove old calls after waiting
                now = time.time()
                self.calls = [c for c in self.calls if now - c < self.period]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    min_rows: int = 1
) -> Tuple[bool, str]:
    """
    Validate DataFrame has required structure.

    Args:
        df: DataFrame to validate
        required_columns: Required column names
        min_rows: Minimum number of rows

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is None or empty"

    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    return True, ""


if __name__ == "__main__":
    # Test error handling utilities
    logger.info("Testing Error Handling")

    # Test circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

    def failing_function():
        raise ValueError("This always fails")

    # Test retries
    @retry(max_attempts=3, delay=0.1, exceptions=(ValueError,))
    def unreliable_function(fail_count=2):
        """Function that fails first N times."""
        if not hasattr(unreliable_function, 'attempts'):
            unreliable_function.attempts = 0

        unreliable_function.attempts += 1

        if unreliable_function.attempts <= fail_count:
            logger.info(f"Attempt {unreliable_function.attempts}: failing")
            raise ValueError("Not yet!")

        logger.info(f"Attempt {unreliable_function.attempts}: success")
        return "Success!"

    # Test retry
    try:
        result = unreliable_function(fail_count=2)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    # Test fallback
    @fallback(lambda *args, **kwargs: "Fallback value")
    def risky_function():
        raise RuntimeError("Oops!")

    print(f"Fallback result: {risky_function()}")

    # Test rate limiter
    rate_limiter = APIRateLimiter(max_calls=5, period_seconds=1.0)

    @rate_limiter
    def api_call(n):
        print(f"API call {n} at {time.time():.2f}")
        return n

    # Make multiple calls (should be rate limited)
    for i in range(8):
        api_call(i)
