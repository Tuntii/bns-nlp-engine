"""Tests for security utilities including API key management and data sanitization."""

import os
from unittest.mock import patch

import pytest

from bnsnlp.core.exceptions import ConfigurationError
from bnsnlp.utils.security import SecureConfig


class TestSecureConfig:
    """Test suite for SecureConfig class."""
    
    def test_get_api_key_success(self):
        """Test successful API key retrieval from environment."""
        with patch.dict(os.environ, {'BNSNLP_OPENAI_API_KEY': 'sk-test123456'}):
            api_key = SecureConfig.get_api_key('openai')
            assert api_key == 'sk-test123456'
    
    def test_get_api_key_custom_env_var(self):
        """Test API key retrieval with custom environment variable name."""
        with patch.dict(os.environ, {'CUSTOM_KEY': 'test-key-789'}):
            api_key = SecureConfig.get_api_key('custom', env_var='CUSTOM_KEY')
            assert api_key == 'test-key-789'
    
    def test_get_api_key_not_found(self):
        """Test that ConfigurationError is raised when API key is not found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                SecureConfig.get_api_key('openai')
            
            assert 'not found' in str(exc_info.value).lower()
            assert 'BNSNLP_OPENAI_API_KEY' in str(exc_info.value)
    
    def test_get_api_key_empty_string(self):
        """Test that ConfigurationError is raised for empty API key."""
        with patch.dict(os.environ, {'BNSNLP_OPENAI_API_KEY': '   '}):
            with pytest.raises(ConfigurationError) as exc_info:
                SecureConfig.get_api_key('openai')
            
            assert 'empty' in str(exc_info.value).lower()
    
    def test_get_api_key_strips_whitespace(self):
        """Test that API key whitespace is stripped."""
        with patch.dict(os.environ, {'BNSNLP_OPENAI_API_KEY': '  sk-test123  '}):
            api_key = SecureConfig.get_api_key('openai')
            assert api_key == 'sk-test123'
    
    def test_get_api_key_optional_success(self):
        """Test optional API key retrieval when key exists."""
        with patch.dict(os.environ, {'BNSNLP_COHERE_API_KEY': 'cohere-key-123'}):
            api_key = SecureConfig.get_api_key_optional('cohere')
            assert api_key == 'cohere-key-123'
    
    def test_get_api_key_optional_not_found(self):
        """Test optional API key retrieval returns None when not found."""
        with patch.dict(os.environ, {}, clear=True):
            api_key = SecureConfig.get_api_key_optional('openai')
            assert api_key is None
    
    def test_get_api_key_optional_empty(self):
        """Test optional API key retrieval returns None for empty string."""
        with patch.dict(os.environ, {'BNSNLP_OPENAI_API_KEY': '   '}):
            api_key = SecureConfig.get_api_key_optional('openai')
            assert api_key is None
    
    def test_mask_sensitive_data_long_string(self):
        """Test masking of long sensitive strings."""
        api_key = "sk-1234567890abcdefghijklmnop"
        masked = SecureConfig.mask_sensitive_data(api_key)
        
        assert masked == "sk-1***mnop"
        assert len(masked) < len(api_key)
        assert "1234567890" not in masked
    
    def test_mask_sensitive_data_short_string(self):
        """Test masking of short strings."""
        short_key = "abc"
        masked = SecureConfig.mask_sensitive_data(short_key)
        
        assert masked == "***"
        assert "abc" not in masked
    
    def test_mask_sensitive_data_empty_string(self):
        """Test masking of empty string."""
        masked = SecureConfig.mask_sensitive_data("")
        assert masked == ""
    
    def test_mask_sensitive_data_custom_params(self):
        """Test masking with custom mask character and visible chars."""
        api_key = "sk-1234567890abcdefghij"
        masked = SecureConfig.mask_sensitive_data(api_key, mask_char='X', visible_chars=2)
        
        assert masked == "skXXXij"
        assert "1234567890" not in masked
    
    def test_sanitize_dict_masks_sensitive_keys(self):
        """Test dictionary sanitization masks sensitive keys."""
        data = {
            'api_key': 'sk-1234567890',
            'model': 'gpt-4',
            'password': 'secret123',
            'username': 'john'
        }
        
        sanitized = SecureConfig.sanitize_dict(data)
        
        assert sanitized['api_key'] == 'sk-1***7890'
        assert sanitized['model'] == 'gpt-4'
        assert sanitized['password'] == 'secr***t123'
        assert sanitized['username'] == 'john'
    
    def test_sanitize_dict_nested(self):
        """Test sanitization of nested dictionaries."""
        data = {
            'config': {
                'api_key': 'sk-test123456',
                'timeout': 30
            },
            'name': 'test'
        }
        
        sanitized = SecureConfig.sanitize_dict(data)
        
        assert sanitized['config']['api_key'] == 'sk-t***3456'
        assert sanitized['config']['timeout'] == 30
        assert sanitized['name'] == 'test'
    
    def test_sanitize_dict_with_lists(self):
        """Test sanitization of dictionaries containing lists."""
        data = {
            'items': [
                {'api_key': 'key1', 'value': 1},
                {'api_key': 'key2', 'value': 2}
            ]
        }
        
        sanitized = SecureConfig.sanitize_dict(data)
        
        assert sanitized['items'][0]['api_key'] == '***'
        assert sanitized['items'][0]['value'] == 1
        assert sanitized['items'][1]['api_key'] == '***'
    
    def test_sanitize_dict_custom_sensitive_keys(self):
        """Test sanitization with custom sensitive keys."""
        data = {
            'custom_secret': 'secret123',
            'public_data': 'visible'
        }
        
        sanitized = SecureConfig.sanitize_dict(data, sensitive_keys={'custom_secret'})
        
        assert sanitized['custom_secret'] == 'secr***t123'
        assert sanitized['public_data'] == 'visible'
    
    def test_detect_sensitive_data_api_key(self):
        """Test detection of API key patterns."""
        text = "My API key is sk-1234567890abcdefghijklmnop"
        assert SecureConfig.detect_sensitive_data(text) is True
    
    def test_detect_sensitive_data_bearer_token(self):
        """Test detection of Bearer token patterns."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        assert SecureConfig.detect_sensitive_data(text) is True
    
    def test_detect_sensitive_data_password(self):
        """Test detection of password patterns."""
        text = "password: mysecretpass123"
        assert SecureConfig.detect_sensitive_data(text) is True
    
    def test_detect_sensitive_data_clean_text(self):
        """Test that clean text is not detected as sensitive."""
        text = "Hello world, this is a normal message"
        assert SecureConfig.detect_sensitive_data(text) is False
    
    def test_redact_sensitive_data_api_key(self):
        """Test redaction of API keys from text."""
        text = "Connect with key sk-1234567890abcdefghijklmnop to API"
        redacted = SecureConfig.redact_sensitive_data(text)
        
        assert "[REDACTED]" in redacted
        assert "sk-1234567890" not in redacted
    
    def test_redact_sensitive_data_custom_replacement(self):
        """Test redaction with custom replacement text."""
        text = "password: secret123"
        redacted = SecureConfig.redact_sensitive_data(text, replacement='[HIDDEN]')
        
        assert "[HIDDEN]" in redacted
        assert "secret123" not in redacted
    
    def test_redact_sensitive_data_multiple_patterns(self):
        """Test redaction of multiple sensitive patterns."""
        text = "API key sk-1234567890abcdefghij and password: secret"
        redacted = SecureConfig.redact_sensitive_data(text)
        
        assert redacted.count("[REDACTED]") >= 1
        assert "sk-1234567890abcdefghij" not in redacted
    
    def test_validate_api_key_format_openai_valid(self):
        """Test OpenAI API key format validation for valid key."""
        assert SecureConfig.validate_api_key_format('sk-1234567890abcdefghij', 'openai') is True
    
    def test_validate_api_key_format_openai_invalid_prefix(self):
        """Test OpenAI API key format validation for invalid prefix."""
        assert SecureConfig.validate_api_key_format('invalid-key', 'openai') is False
    
    def test_validate_api_key_format_openai_too_short(self):
        """Test OpenAI API key format validation for too short key."""
        assert SecureConfig.validate_api_key_format('sk-123', 'openai') is False
    
    def test_validate_api_key_format_cohere_valid(self):
        """Test Cohere API key format validation for valid key."""
        assert SecureConfig.validate_api_key_format('1234567890abcdefghij1234567890', 'cohere') is True
    
    def test_validate_api_key_format_cohere_too_short(self):
        """Test Cohere API key format validation for too short key."""
        assert SecureConfig.validate_api_key_format('short', 'cohere') is False
    
    def test_validate_api_key_format_pinecone_valid(self):
        """Test Pinecone API key format validation for valid key."""
        assert SecureConfig.validate_api_key_format('1234567890abcdefghij1234567890', 'pinecone') is True
    
    def test_validate_api_key_format_unknown_service(self):
        """Test API key format validation for unknown service."""
        assert SecureConfig.validate_api_key_format('any-key', 'unknown') is True
    
    def test_validate_api_key_format_empty(self):
        """Test API key format validation for empty key."""
        assert SecureConfig.validate_api_key_format('', 'openai') is False


class TestSecureConfigIntegration:
    """Integration tests for SecureConfig with real environment variables."""
    
    def test_full_workflow(self):
        """Test complete workflow of getting and masking API key."""
        with patch.dict(os.environ, {'BNSNLP_TEST_API_KEY': 'sk-test1234567890abcdefghij'}):
            # Get API key
            api_key = SecureConfig.get_api_key('test')
            assert api_key == 'sk-test1234567890abcdefghij'
            
            # Mask it for logging
            masked = SecureConfig.mask_sensitive_data(api_key)
            assert masked == 'sk-t***ghij'
            
            # Validate format
            is_valid = SecureConfig.validate_api_key_format(api_key, 'openai')
            assert is_valid is True
    
    def test_sanitize_config_dict(self):
        """Test sanitizing a configuration dictionary."""
        with patch.dict(os.environ, {'BNSNLP_OPENAI_API_KEY': 'sk-real-key-123'}):
            config = {
                'embed': {
                    'provider': 'openai',
                    'api_key': SecureConfig.get_api_key('openai'),
                    'model': 'text-embedding-3-small'
                },
                'search': {
                    'provider': 'qdrant',
                    'url': 'http://localhost:6333'
                }
            }
            
            # Sanitize for logging
            sanitized = SecureConfig.sanitize_dict(config)
            
            assert sanitized['embed']['provider'] == 'openai'
            assert sanitized['embed']['api_key'] == 'sk-r***-123'
            assert sanitized['embed']['model'] == 'text-embedding-3-small'
            assert sanitized['search']['url'] == 'http://localhost:6333'
