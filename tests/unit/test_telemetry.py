"""Tests for telemetry utilities including opt-in tracking and data sanitization."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bnsnlp.core.config import TelemetryConfig
from bnsnlp.utils.telemetry import (
    Telemetry,
    get_telemetry,
    initialize_telemetry,
    set_telemetry,
    track_event,
)


class TestTelemetry:
    """Test suite for Telemetry class."""
    
    def test_telemetry_disabled_by_default(self):
        """Test that telemetry is disabled by default."""
        config = TelemetryConfig()
        telemetry = Telemetry(config)
        
        assert telemetry.enabled is False
    
    def test_telemetry_opt_in(self):
        """Test that telemetry can be explicitly enabled."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        assert telemetry.enabled is True
        assert telemetry.endpoint == 'https://telemetry.example.com'
    
    @pytest.mark.asyncio
    async def test_track_event_disabled(self):
        """Test that events are not sent when telemetry is disabled."""
        config = TelemetryConfig(enabled=False)
        telemetry = Telemetry(config)
        
        result = await telemetry.track_event('test_event', {'key': 'value'})
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_track_event_no_endpoint(self):
        """Test that events are not sent when no endpoint is configured."""
        # Create telemetry with enabled=True but manually set endpoint to None after creation
        config = TelemetryConfig(enabled=True, endpoint='https://temp.example.com')
        telemetry = Telemetry(config)
        telemetry.endpoint = None  # Manually set to None to test this case
        
        result = await telemetry.track_event('test_event')
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_track_event_success(self):
        """Test successful event tracking."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Mock the HTTP session with proper async context manager
        mock_response = MagicMock()
        mock_response.status = 200
        
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_post_context.__aexit__.return_value = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_post_context
        mock_session.closed = False
        
        telemetry._session = mock_session
        
        result = await telemetry.track_event('test_event', {'operation': 'test'})
        
        assert result is True
        mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_track_event_http_error(self):
        """Test event tracking with HTTP error response."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Mock the HTTP session with error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None
        
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        mock_session.closed = False
        
        telemetry._session = mock_session
        
        result = await telemetry.track_event('test_event')
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_track_event_timeout(self):
        """Test event tracking with timeout."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Mock the HTTP session to raise timeout
        mock_session = AsyncMock()
        mock_session.post.side_effect = asyncio.TimeoutError()
        mock_session.closed = False
        
        telemetry._session = mock_session
        
        result = await telemetry.track_event('test_event', timeout=0.1)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_track_event_exception(self):
        """Test that exceptions in event tracking don't crash the application."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Mock the HTTP session to raise exception
        mock_session = AsyncMock()
        mock_session.post.side_effect = Exception("Network error")
        mock_session.closed = False
        
        telemetry._session = mock_session
        
        # Should not raise exception
        result = await telemetry.track_event('test_event')
        
        assert result is False
    
    def test_sanitize_properties_safe_keys(self):
        """Test that only safe properties are included."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'module': 'preprocess',
            'operation': 'normalize',
            'duration_ms': 150,
            'success': True,
            'user_content': 'sensitive data',  # Should be filtered
            'api_key': 'sk-123456'  # Should be filtered
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        assert 'module' in sanitized
        assert 'operation' in sanitized
        assert 'duration_ms' in sanitized
        assert 'success' in sanitized
        assert 'user_content' not in sanitized
        assert 'api_key' not in sanitized
    
    def test_sanitize_properties_detects_sensitive_strings(self):
        """Test that sensitive data in string values is filtered."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'module': 'embed',
            'provider': 'sk-1234567890abcdefghij'  # Contains API key pattern
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        assert 'module' in sanitized
        assert 'provider' not in sanitized  # Filtered due to sensitive pattern
    
    def test_sanitize_properties_limits_string_length(self):
        """Test that string values are limited in length."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        long_string = 'a' * 200
        properties = {
            'module': long_string
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        # Module is a safe key, so it should be included but truncated
        assert 'module' in sanitized
        assert len(sanitized['module']) == 100
    
    def test_sanitize_properties_numeric_values(self):
        """Test that numeric values are preserved."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'duration_ms': 150,
            'batch_size': 32,
            'item_count': 100,
            'success': True
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        assert sanitized['duration_ms'] == 150
        assert sanitized['batch_size'] == 32
        assert sanitized['item_count'] == 100
        assert sanitized['success'] is True
    
    def test_sanitize_properties_skips_complex_types(self):
        """Test that complex types (dict, list) are filtered."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'module': 'test',
            'config': {'key': 'value'},  # Should be filtered
            'items': [1, 2, 3]  # Should be filtered
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        assert 'module' in sanitized
        assert 'config' not in sanitized
        assert 'items' not in sanitized
    
    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing telemetry session."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Create a mock session
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        
        telemetry._session = mock_session
        
        await telemetry.close()
        
        mock_session.close.assert_called_once()
        assert telemetry._session is None
    
    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test async context manager."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        
        async with Telemetry(config) as telemetry:
            assert telemetry.enabled is True
        
        # Session should be closed after context exit
        assert telemetry._session is None or telemetry._session.closed
    
    def test_create_from_config(self):
        """Test factory method for creating Telemetry instance."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry.create_from_config(config)
        
        assert isinstance(telemetry, Telemetry)
        assert telemetry.enabled is True
        assert telemetry.endpoint == 'https://telemetry.example.com'


class TestTelemetryGlobalFunctions:
    """Test suite for global telemetry functions."""
    
    def test_get_telemetry_not_initialized(self):
        """Test getting telemetry when not initialized."""
        # Reset global telemetry
        from bnsnlp.utils import telemetry as telemetry_module
        telemetry_module._global_telemetry = None
        
        result = get_telemetry()
        assert result is None
    
    def test_set_and_get_telemetry(self):
        """Test setting and getting global telemetry instance."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        set_telemetry(telemetry)
        
        retrieved = get_telemetry()
        assert retrieved is telemetry
    
    def test_initialize_telemetry(self):
        """Test initializing global telemetry instance."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        
        telemetry = initialize_telemetry(config)
        
        assert isinstance(telemetry, Telemetry)
        assert get_telemetry() is telemetry
    
    @pytest.mark.asyncio
    async def test_track_event_global_enabled(self):
        """Test global track_event function when telemetry is enabled."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Mock the track_event method
        telemetry.track_event = AsyncMock(return_value=True)
        
        set_telemetry(telemetry)
        
        result = await track_event('test_event', {'key': 'value'})
        
        assert result is True
        telemetry.track_event.assert_called_once_with('test_event', {'key': 'value'})
    
    @pytest.mark.asyncio
    async def test_track_event_global_not_initialized(self):
        """Test global track_event function when telemetry is not initialized."""
        # Reset global telemetry
        from bnsnlp.utils import telemetry as telemetry_module
        telemetry_module._global_telemetry = None
        
        result = await track_event('test_event')
        
        assert result is False


class TestTelemetryPrivacy:
    """Test suite for telemetry privacy guarantees."""
    
    def test_no_user_content_collected(self):
        """Test that user content is never collected."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'module': 'preprocess',
            'text': 'User input text',  # Should be filtered
            'content': 'More user content',  # Should be filtered
            'duration_ms': 100
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        assert 'text' not in sanitized
        assert 'content' not in sanitized
        assert 'duration_ms' in sanitized
    
    def test_no_api_keys_collected(self):
        """Test that API keys are never collected."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'module': 'embed',
            'api_key': 'sk-1234567890',  # Should be filtered
            'provider': 'openai'
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        assert 'api_key' not in sanitized
        assert 'provider' in sanitized
    
    def test_only_anonymous_data_collected(self):
        """Test that only anonymous, aggregated data is collected."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        properties = {
            'module': 'search',
            'operation': 'similarity_search',
            'duration_ms': 250,
            'success': True,
            'batch_size': 10
        }
        
        sanitized = telemetry._sanitize_properties(properties)
        
        # All these are safe, anonymous metrics
        assert sanitized == properties
    
    @pytest.mark.asyncio
    async def test_telemetry_errors_dont_affect_application(self):
        """Test that telemetry errors never crash the main application."""
        config = TelemetryConfig(enabled=True, endpoint='https://telemetry.example.com')
        telemetry = Telemetry(config)
        
        # Mock session to raise various exceptions
        mock_session = AsyncMock()
        mock_session.post.side_effect = [
            Exception("Network error"),
            asyncio.TimeoutError(),
            RuntimeError("Unexpected error")
        ]
        mock_session.closed = False
        
        telemetry._session = mock_session
        
        # None of these should raise exceptions
        result1 = await telemetry.track_event('event1')
        result2 = await telemetry.track_event('event2')
        result3 = await telemetry.track_event('event3')
        
        assert result1 is False
        assert result2 is False
        assert result3 is False


class TestTelemetryIntegration:
    """Integration tests for telemetry functionality."""
    
    @pytest.mark.asyncio
    async def test_full_telemetry_workflow(self):
        """Test complete telemetry workflow from initialization to tracking."""
        # Initialize with disabled telemetry (default)
        config = TelemetryConfig(enabled=False)
        telemetry = initialize_telemetry(config)
        
        # Event should not be sent
        result = await track_event('test_event', {'module': 'test'})
        assert result is False
        
        # Enable telemetry
        config_enabled = TelemetryConfig(
            enabled=True,
            endpoint='https://telemetry.example.com'
        )
        telemetry_enabled = Telemetry(config_enabled)
        
        # Mock the session with proper async context manager
        mock_response = MagicMock()
        mock_response.status = 200
        
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_post_context.__aexit__.return_value = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_post_context
        mock_session.closed = False
        
        telemetry_enabled._session = mock_session
        set_telemetry(telemetry_enabled)
        
        # Event should be sent
        result = await track_event('test_event', {'module': 'test', 'success': True})
        assert result is True
        
        # Cleanup
        await telemetry_enabled.close()
