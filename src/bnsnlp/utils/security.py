"""Security utilities for API key management and sensitive data handling."""

import os
import re
from typing import Optional

from bnsnlp.core.exceptions import ConfigurationError


class SecureConfig:
    """Secure configuration management for API keys and sensitive data.
    
    This class provides secure methods for loading API keys from environment
    variables only (never from code or config files) and masking sensitive
    data for logging purposes.
    
    Example:
        >>> api_key = SecureConfig.get_api_key('openai')
        >>> masked = SecureConfig.mask_sensitive_data(api_key)
        >>> print(masked)  # 'sk-1...xyz'
    """
    
    # Patterns for detecting sensitive data
    SENSITIVE_PATTERNS = {
        'api_key': re.compile(r'(sk-[a-zA-Z0-9]{20,}|[a-zA-Z0-9]{32,})'),
        'bearer_token': re.compile(r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*', re.IGNORECASE),
        'password': re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
    }
    
    @staticmethod
    def get_api_key(service: str, env_var: Optional[str] = None) -> str:
        """Get API key from environment variable only.
        
        This method retrieves API keys exclusively from environment variables,
        never from code or configuration files. This ensures that sensitive
        credentials are not accidentally committed to version control.
        
        Args:
            service: Name of the service (e.g., 'openai', 'cohere', 'qdrant')
            env_var: Optional custom environment variable name. If not provided,
                    defaults to 'BNSNLP_{SERVICE}_API_KEY'
        
        Returns:
            API key string from environment variable
            
        Raises:
            ConfigurationError: If the API key is not found in environment
            
        Example:
            >>> # Set environment variable first: export BNSNLP_OPENAI_API_KEY=sk-...
            >>> api_key = SecureConfig.get_api_key('openai')
            >>> # Or use custom env var name
            >>> api_key = SecureConfig.get_api_key('openai', 'CUSTOM_OPENAI_KEY')
        """
        # Determine environment variable name
        if env_var is None:
            env_var = f"BNSNLP_{service.upper()}_API_KEY"
        
        # Get API key from environment
        api_key = os.getenv(env_var)
        
        if not api_key:
            raise ConfigurationError(
                f"API key for {service} not found. Set {env_var} environment variable.",
                context={'service': service, 'env_var': env_var}
            )
        
        # Validate that the key is not empty or just whitespace
        if not api_key.strip():
            raise ConfigurationError(
                f"API key for {service} is empty. Set a valid {env_var} environment variable.",
                context={'service': service, 'env_var': env_var}
            )
        
        return api_key.strip()
    
    @staticmethod
    def get_api_key_optional(service: str, env_var: Optional[str] = None) -> Optional[str]:
        """Get API key from environment variable, returning None if not found.
        
        This is a non-raising version of get_api_key() that returns None
        instead of raising an exception when the API key is not found.
        
        Args:
            service: Name of the service (e.g., 'openai', 'cohere', 'qdrant')
            env_var: Optional custom environment variable name
        
        Returns:
            API key string from environment variable, or None if not found
            
        Example:
            >>> api_key = SecureConfig.get_api_key_optional('openai')
            >>> if api_key:
            ...     # Use the API key
            ...     pass
        """
        if env_var is None:
            env_var = f"BNSNLP_{service.upper()}_API_KEY"
        
        api_key = os.getenv(env_var)
        
        if api_key and api_key.strip():
            return api_key.strip()
        
        return None
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = '*', visible_chars: int = 4) -> str:
        """Mask sensitive data for safe logging.
        
        This method masks sensitive strings like API keys, tokens, and passwords
        to prevent them from being exposed in logs. It shows only the first and
        last few characters for identification purposes.
        
        Args:
            data: The sensitive string to mask
            mask_char: Character to use for masking (default: '*')
            visible_chars: Number of characters to show at start and end (default: 4)
        
        Returns:
            Masked string showing only first and last few characters
            
        Example:
            >>> api_key = "sk-1234567890abcdefghij"
            >>> masked = SecureConfig.mask_sensitive_data(api_key)
            >>> print(masked)  # 'sk-1...ghij'
            
            >>> short_key = "abc"
            >>> masked = SecureConfig.mask_sensitive_data(short_key)
            >>> print(masked)  # '***'
        """
        if not data:
            return ""
        
        # For very short strings, mask completely
        if len(data) <= visible_chars * 2:
            return mask_char * 3
        
        # Show first and last few characters
        start = data[:visible_chars]
        end = data[-visible_chars:]
        middle = mask_char * 3  # Use 3 mask characters regardless of actual length
        
        return f"{start}{middle}{end}"
    
    @staticmethod
    def sanitize_dict(data: dict, sensitive_keys: Optional[set] = None) -> dict:
        """Sanitize a dictionary by masking sensitive values.
        
        This method recursively processes a dictionary and masks values for
        keys that are identified as sensitive (e.g., 'api_key', 'password').
        
        Args:
            data: Dictionary to sanitize
            sensitive_keys: Set of key names to mask. If None, uses default set.
        
        Returns:
            New dictionary with sensitive values masked
            
        Example:
            >>> config = {
            ...     'api_key': 'sk-1234567890',
            ...     'model': 'gpt-4',
            ...     'password': 'secret123'
            ... }
            >>> sanitized = SecureConfig.sanitize_dict(config)
            >>> print(sanitized)
            {'api_key': 'sk-1...890', 'model': 'gpt-4', 'password': 'sec...123'}
        """
        if sensitive_keys is None:
            sensitive_keys = {
                'api_key', 'apikey', 'api-key',
                'password', 'passwd', 'pwd',
                'token', 'auth_token', 'access_token',
                'secret', 'secret_key',
                'authorization', 'auth',
                'private_key', 'privatekey'
            }
        
        sanitized = {}
        
        for key, value in data.items():
            # Check if key is sensitive (case-insensitive)
            is_sensitive = any(
                sensitive.lower() in key.lower() 
                for sensitive in sensitive_keys
            )
            
            if is_sensitive and isinstance(value, str):
                # Mask the sensitive value
                sanitized[key] = SecureConfig.mask_sensitive_data(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = SecureConfig.sanitize_dict(value, sensitive_keys)
            elif isinstance(value, list):
                # Process lists
                sanitized[key] = [
                    SecureConfig.sanitize_dict(item, sensitive_keys) 
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # Keep non-sensitive values as-is
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def detect_sensitive_data(text: str) -> bool:
        """Detect if text contains sensitive data patterns.
        
        This method checks if the given text contains patterns that match
        common sensitive data formats like API keys, tokens, or passwords.
        
        Args:
            text: Text to check for sensitive data
        
        Returns:
            True if sensitive data patterns are detected, False otherwise
            
        Example:
            >>> text = "My API key is sk-1234567890abcdefghij"
            >>> SecureConfig.detect_sensitive_data(text)
            True
            
            >>> text = "Hello world"
            >>> SecureConfig.detect_sensitive_data(text)
            False
        """
        for pattern in SecureConfig.SENSITIVE_PATTERNS.values():
            if pattern.search(text):
                return True
        return False
    
    @staticmethod
    def redact_sensitive_data(text: str, replacement: str = '[REDACTED]') -> str:
        """Redact sensitive data from text using pattern matching.
        
        This method finds and replaces sensitive data patterns in text with
        a redaction marker. Useful for sanitizing log messages or error messages.
        
        Args:
            text: Text to redact sensitive data from
            replacement: String to replace sensitive data with (default: '[REDACTED]')
        
        Returns:
            Text with sensitive data replaced by redaction marker
            
        Example:
            >>> text = "Connect with key sk-1234567890abcdefghij to API"
            >>> redacted = SecureConfig.redact_sensitive_data(text)
            >>> print(redacted)  # "Connect with key [REDACTED] to API"
        """
        redacted_text = text
        
        for pattern in SecureConfig.SENSITIVE_PATTERNS.values():
            redacted_text = pattern.sub(replacement, redacted_text)
        
        return redacted_text
    
    @staticmethod
    def validate_api_key_format(api_key: str, service: str) -> bool:
        """Validate API key format for known services.
        
        This method performs basic format validation for API keys from
        known services to catch obvious errors early.
        
        Args:
            api_key: API key to validate
            service: Service name (e.g., 'openai', 'cohere')
        
        Returns:
            True if format appears valid, False otherwise
            
        Example:
            >>> SecureConfig.validate_api_key_format('sk-1234567890', 'openai')
            True
            >>> SecureConfig.validate_api_key_format('invalid', 'openai')
            False
        """
        service_lower = service.lower()
        
        # OpenAI keys start with 'sk-' and are typically 48+ characters
        if service_lower == 'openai':
            return api_key.startswith('sk-') and len(api_key) >= 20
        
        # Cohere keys are typically 40 characters alphanumeric
        elif service_lower == 'cohere':
            return len(api_key) >= 20 and api_key.replace('-', '').isalnum()
        
        # Pinecone keys are typically UUIDs or long alphanumeric strings
        elif service_lower == 'pinecone':
            return len(api_key) >= 20
        
        # For unknown services, just check it's not empty
        return len(api_key) > 0
