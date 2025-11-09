"""Telemetry utilities for anonymous usage tracking (opt-in only)."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import aiohttp

from bnsnlp.core.config import TelemetryConfig
from bnsnlp.utils.logging import get_logger
from bnsnlp.utils.security import SecureConfig

# Get logger
logger = get_logger(__name__)


class Telemetry:
    """Privacy-focused telemetry for anonymous usage tracking.

    This class implements opt-in telemetry that collects only anonymous,
    aggregated usage statistics. It is disabled by default and requires
    explicit user consent to enable.

    Privacy guarantees:
    - Disabled by default (opt-in only)
    - No user content is collected
    - No API keys or credentials are collected
    - No personally identifiable information (PII) is collected
    - Only anonymous usage statistics are tracked
    - All data is sanitized before transmission

    Example:
        >>> from bnsnlp.core.config import TelemetryConfig
        >>> config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        >>> telemetry = Telemetry(config)
        >>> await telemetry.track_event('preprocess', {'operation': 'normalize', 'success': True})
    """

    def __init__(self, config: TelemetryConfig):
        """Initialize telemetry with configuration.

        Args:
            config: TelemetryConfig instance with enabled flag and endpoint
        """
        self.enabled = config.enabled
        self.endpoint = config.endpoint
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id = str(uuid4())  # Anonymous session identifier

        # Log telemetry status
        if self.enabled:
            logger.info(
                "Telemetry enabled (opt-in)", extra={"context": {"endpoint": self.endpoint}}
            )
        else:
            logger.debug("Telemetry disabled (default)")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.

        Returns:
            Active aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the telemetry session and cleanup resources.

        This should be called when shutting down the application to
        properly close the HTTP session.

        Example:
            >>> telemetry = Telemetry(config)
            >>> # ... use telemetry ...
            >>> await telemetry.close()
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def track_event(
        self, event_name: str, properties: Optional[Dict[str, Any]] = None, timeout: float = 5.0
    ) -> bool:
        """Track an anonymous usage event.

        This method sends anonymous usage statistics to the telemetry endpoint
        if telemetry is enabled. All data is sanitized before transmission to
        ensure no sensitive information is leaked.

        Args:
            event_name: Name of the event (e.g., 'preprocess', 'embed', 'search')
            properties: Optional dictionary of event properties (will be sanitized)
            timeout: Request timeout in seconds (default: 5.0)

        Returns:
            True if event was sent successfully, False otherwise

        Example:
            >>> await telemetry.track_event(
            ...     'embed',
            ...     {'provider': 'openai', 'duration_ms': 150, 'success': True}
            ... )
        """
        # If telemetry is disabled, do nothing
        if not self.enabled:
            return False

        # Validate endpoint
        if not self.endpoint:
            logger.warning("Telemetry enabled but no endpoint configured")
            return False

        try:
            # Create event data
            event_data = {
                "event": event_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self._session_id,
                "properties": self._sanitize_properties(properties or {}),
            }

            # Send event asynchronously (fire and forget with timeout)
            session = await self._get_session()

            try:
                async with session.post(
                    self.endpoint,
                    json=event_data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if response.status == 200:
                        logger.debug(
                            f"Telemetry event sent: {event_name}",
                            extra={"context": {"event": event_name}},
                        )
                        return True
                    else:
                        logger.warning(
                            f"Telemetry event failed: {response.status}",
                            extra={"context": {"event": event_name, "status": response.status}},
                        )
                        return False
            except aiohttp.ClientError as e:
                logger.debug(f"Telemetry client error: {str(e)}")
                return False

        except asyncio.TimeoutError:
            logger.debug(f"Telemetry event timeout: {event_name}")
            return False
        except Exception as e:
            # Never let telemetry errors affect the main application
            logger.debug(
                f"Telemetry error: {str(e)}",
                extra={"context": {"event": event_name, "error": str(e)}},
            )
            return False

    def _sanitize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event properties to remove sensitive data.

        This method ensures that only safe, anonymous properties are included
        in telemetry events. It removes any sensitive information and limits
        the properties to a predefined safe list.

        Args:
            properties: Raw event properties

        Returns:
            Sanitized properties dictionary
        """
        # Define safe property keys that can be collected
        safe_keys = {
            # Module/operation information
            "module",
            "operation",
            "provider",
            "adapter",
            # Performance metrics
            "duration_ms",
            "batch_size",
            "item_count",
            # Success/failure (boolean only)
            "success",
            "error_type",
            # Configuration (non-sensitive)
            "use_gpu",
            "async_mode",
            # Version information
            "version",
            "python_version",
        }

        sanitized = {}

        for key, value in properties.items():
            # Only include safe keys
            if key not in safe_keys:
                continue

            # Ensure value is not sensitive
            if isinstance(value, str):
                # Check if string contains sensitive data
                if SecureConfig.detect_sensitive_data(value):
                    continue
                # Limit string length to prevent data leakage
                sanitized[key] = value[:100]
            elif isinstance(value, (int, float, bool)):
                # Numeric and boolean values are safe
                sanitized[key] = value
            elif value is None:
                sanitized[key] = None
            # Skip complex types (dict, list, etc.)

        return sanitized

    def track_event_sync(
        self, event_name: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track an event synchronously (fire and forget).

        This is a convenience method that schedules the event tracking
        without waiting for completion. Useful for synchronous code paths.

        Args:
            event_name: Name of the event
            properties: Optional event properties

        Example:
            >>> telemetry.track_event_sync('classify', {'success': True})
        """
        if not self.enabled:
            return

        # Schedule the async task without waiting
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create a task
                asyncio.create_task(self.track_event(event_name, properties))
            else:
                # If no loop is running, run it
                loop.run_until_complete(self.track_event(event_name, properties))
        except RuntimeError:
            # If we can't get a loop, just skip telemetry
            logger.debug("Could not track event: no event loop available")

    @staticmethod
    def create_from_config(config: TelemetryConfig) -> "Telemetry":
        """Create a Telemetry instance from configuration.

        This is a factory method for creating Telemetry instances.

        Args:
            config: TelemetryConfig instance

        Returns:
            Telemetry instance

        Example:
            >>> from bnsnlp.core.config import TelemetryConfig
            >>> config = TelemetryConfig(enabled=False)
            >>> telemetry = Telemetry.create_from_config(config)
        """
        return Telemetry(config)

    def __enter__(self) -> "Telemetry":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Close session synchronously
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except RuntimeError:
                pass

    async def __aenter__(self) -> "Telemetry":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Global telemetry instance (disabled by default)
_global_telemetry: Optional[Telemetry] = None


def get_telemetry() -> Optional[Telemetry]:
    """Get the global telemetry instance.

    Returns:
        Global Telemetry instance or None if not initialized

    Example:
        >>> telemetry = get_telemetry()
        >>> if telemetry:
        ...     await telemetry.track_event('search', {'success': True})
    """
    return _global_telemetry


def set_telemetry(telemetry: Telemetry) -> None:
    """Set the global telemetry instance.

    Args:
        telemetry: Telemetry instance to set as global

    Example:
        >>> from bnsnlp.core.config import TelemetryConfig
        >>> config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        >>> telemetry = Telemetry(config)
        >>> set_telemetry(telemetry)
    """
    global _global_telemetry
    _global_telemetry = telemetry


def initialize_telemetry(config: TelemetryConfig) -> Telemetry:
    """Initialize and set the global telemetry instance.

    This is a convenience function that creates a Telemetry instance
    and sets it as the global instance.

    Args:
        config: TelemetryConfig instance

    Returns:
        Initialized Telemetry instance

    Example:
        >>> from bnsnlp.core.config import Config
        >>> config = Config()
        >>> telemetry = initialize_telemetry(config.telemetry)
    """
    telemetry = Telemetry.create_from_config(config)
    set_telemetry(telemetry)
    return telemetry


async def track_event(event_name: str, properties: Optional[Dict[str, Any]] = None) -> bool:
    """Track an event using the global telemetry instance.

    This is a convenience function that uses the global telemetry instance.
    If telemetry is not initialized or disabled, this is a no-op.

    Args:
        event_name: Name of the event
        properties: Optional event properties

    Returns:
        True if event was sent, False otherwise

    Example:
        >>> await track_event('preprocess', {'operation': 'normalize', 'success': True})
    """
    telemetry = get_telemetry()
    if telemetry:
        return await telemetry.track_event(event_name, properties)
    return False
