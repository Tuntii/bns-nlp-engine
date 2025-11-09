"""
Unit tests for plugin registry system.
"""

from typing import Any, Dict

import pytest

from bnsnlp.core.exceptions import PluginError
from bnsnlp.core.registry import PluginRegistry


# Mock plugin classes for testing
class ValidPlugin:
    """Valid plugin implementation for testing."""
    
    name = "test_plugin"
    version = "1.0.0"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.config = config


class AnotherValidPlugin:
    """Another valid plugin implementation for testing."""
    
    name = "another_plugin"
    version = "2.0.0"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.config = config


class PluginWithoutName:
    """Invalid plugin missing 'name' attribute."""
    
    version = "1.0.0"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass


class PluginWithoutVersion:
    """Invalid plugin missing 'version' attribute."""
    
    name = "no_version_plugin"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass


class PluginWithoutInitialize:
    """Invalid plugin missing 'initialize' method."""
    
    name = "no_init_plugin"
    version = "1.0.0"


class PluginWithNonCallableInitialize:
    """Invalid plugin with non-callable 'initialize' attribute."""
    
    name = "bad_init_plugin"
    version = "1.0.0"
    initialize = "not_callable"


class TestPluginRegistry:
    """Tests for PluginRegistry class."""
    
    def test_initialization(self):
        """Test registry initialization with empty categories."""
        registry = PluginRegistry()
        
        # Check all categories are initialized
        plugins = registry.list_plugins()
        assert 'preprocess' in plugins
        assert 'embed' in plugins
        assert 'search' in plugins
        assert 'classify' in plugins
        
        # Check all categories are empty
        assert plugins['preprocess'] == []
        assert plugins['embed'] == []
        assert plugins['search'] == []
        assert plugins['classify'] == []
    
    def test_register_valid_plugin(self):
        """Test registering a valid plugin."""
        registry = PluginRegistry()
        
        # Register plugin
        registry.register('embed', 'test_embedder', ValidPlugin)
        
        # Verify plugin is registered
        plugins = registry.list_plugins('embed')
        assert 'test_embedder' in plugins['embed']
    
    def test_register_multiple_plugins(self):
        """Test registering multiple plugins in same category."""
        registry = PluginRegistry()
        
        # Register multiple plugins
        registry.register('embed', 'plugin1', ValidPlugin)
        registry.register('embed', 'plugin2', AnotherValidPlugin)
        
        # Verify both plugins are registered
        plugins = registry.list_plugins('embed')
        assert 'plugin1' in plugins['embed']
        assert 'plugin2' in plugins['embed']
        assert len(plugins['embed']) == 2
    
    def test_register_plugins_in_different_categories(self):
        """Test registering plugins in different categories."""
        registry = PluginRegistry()
        
        # Register plugins in different categories
        registry.register('preprocess', 'preprocessor', ValidPlugin)
        registry.register('embed', 'embedder', ValidPlugin)
        registry.register('search', 'searcher', ValidPlugin)
        registry.register('classify', 'classifier', ValidPlugin)
        
        # Verify all plugins are registered
        plugins = registry.list_plugins()
        assert 'preprocessor' in plugins['preprocess']
        assert 'embedder' in plugins['embed']
        assert 'searcher' in plugins['search']
        assert 'classifier' in plugins['classify']
    
    def test_register_invalid_category(self):
        """Test error when registering plugin with invalid category."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.register('invalid_category', 'plugin', ValidPlugin)
        
        assert "Invalid plugin category" in str(exc_info.value)
        assert exc_info.value.code == "PLUGIN_ERROR"
        assert 'invalid_category' in exc_info.value.context['category']
    
    def test_register_duplicate_plugin(self):
        """Test error when registering duplicate plugin name."""
        registry = PluginRegistry()
        
        # Register plugin first time
        registry.register('embed', 'duplicate', ValidPlugin)
        
        # Try to register again with same name
        with pytest.raises(PluginError) as exc_info:
            registry.register('embed', 'duplicate', AnotherValidPlugin)
        
        assert "already registered" in str(exc_info.value)
        assert exc_info.value.code == "PLUGIN_ERROR"
        assert exc_info.value.context['name'] == 'duplicate'
    
    def test_register_plugin_without_name(self):
        """Test error when registering plugin without 'name' attribute."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.register('embed', 'bad_plugin', PluginWithoutName)
        
        assert "missing required attributes" in str(exc_info.value)
        assert 'name' in exc_info.value.context['missing_attributes']
    
    def test_register_plugin_without_version(self):
        """Test error when registering plugin without 'version' attribute."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.register('embed', 'bad_plugin', PluginWithoutVersion)
        
        assert "missing required attributes" in str(exc_info.value)
        assert 'version' in exc_info.value.context['missing_attributes']
    
    def test_register_plugin_without_initialize(self):
        """Test error when registering plugin without 'initialize' method."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.register('embed', 'bad_plugin', PluginWithoutInitialize)
        
        assert "missing required attributes" in str(exc_info.value)
        assert 'initialize' in exc_info.value.context['missing_attributes']
    
    def test_register_plugin_with_non_callable_initialize(self):
        """Test error when registering plugin with non-callable 'initialize'."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.register('embed', 'bad_plugin', PluginWithNonCallableInitialize)
        
        assert "must be callable" in str(exc_info.value)
    
    def test_register_non_class_plugin(self):
        """Test error when registering non-class as plugin."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.register('embed', 'bad_plugin', "not_a_class")
        
        assert "must be a class" in str(exc_info.value)
    
    def test_get_registered_plugin(self):
        """Test retrieving a registered plugin."""
        registry = PluginRegistry()
        
        # Register plugin
        registry.register('embed', 'test_embedder', ValidPlugin)
        
        # Retrieve plugin
        plugin_class = registry.get('embed', 'test_embedder')
        
        assert plugin_class is ValidPlugin
    
    def test_get_plugin_from_invalid_category(self):
        """Test error when getting plugin from invalid category."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.get('invalid_category', 'plugin')
        
        assert "Invalid plugin category" in str(exc_info.value)
    
    def test_get_nonexistent_plugin(self):
        """Test error when getting plugin that doesn't exist."""
        registry = PluginRegistry()
        
        # Register one plugin
        registry.register('embed', 'existing', ValidPlugin)
        
        # Try to get non-existent plugin
        with pytest.raises(PluginError) as exc_info:
            registry.get('embed', 'nonexistent')
        
        assert "not found" in str(exc_info.value)
        assert exc_info.value.context['name'] == 'nonexistent'
        assert 'existing' in exc_info.value.context['available_plugins']
    
    def test_list_all_plugins(self):
        """Test listing all plugins across all categories."""
        registry = PluginRegistry()
        
        # Register plugins in different categories
        registry.register('preprocess', 'preprocessor1', ValidPlugin)
        registry.register('embed', 'embedder1', ValidPlugin)
        registry.register('embed', 'embedder2', AnotherValidPlugin)
        registry.register('search', 'searcher1', ValidPlugin)
        
        # List all plugins
        plugins = registry.list_plugins()
        
        assert len(plugins) == 4  # All categories
        assert 'preprocessor1' in plugins['preprocess']
        assert 'embedder1' in plugins['embed']
        assert 'embedder2' in plugins['embed']
        assert 'searcher1' in plugins['search']
        assert plugins['classify'] == []  # Empty category
    
    def test_list_plugins_by_category(self):
        """Test listing plugins filtered by category."""
        registry = PluginRegistry()
        
        # Register plugins in different categories
        registry.register('embed', 'embedder1', ValidPlugin)
        registry.register('embed', 'embedder2', AnotherValidPlugin)
        registry.register('search', 'searcher1', ValidPlugin)
        
        # List only embed plugins
        plugins = registry.list_plugins('embed')
        
        assert len(plugins) == 1  # Only one category
        assert 'embed' in plugins
        assert 'embedder1' in plugins['embed']
        assert 'embedder2' in plugins['embed']
        assert 'search' not in plugins
    
    def test_list_plugins_invalid_category(self):
        """Test error when listing plugins with invalid category."""
        registry = PluginRegistry()
        
        with pytest.raises(PluginError) as exc_info:
            registry.list_plugins('invalid_category')
        
        assert "Invalid plugin category" in str(exc_info.value)
    
    def test_list_plugins_empty_category(self):
        """Test listing plugins from empty category."""
        registry = PluginRegistry()
        
        # List empty category
        plugins = registry.list_plugins('embed')
        
        assert plugins == {'embed': []}


class TestPluginDiscovery:
    """Tests for plugin discovery via entry_points."""
    
    def test_discover_plugins_no_entry_points(self):
        """Test plugin discovery when no entry points are registered."""
        registry = PluginRegistry()
        
        # Should not raise error even with no entry points
        registry.discover_plugins()
        
        # Registry should still be initialized
        plugins = registry.list_plugins()
        assert len(plugins) == 4
    
    def test_discover_plugins_with_mock_entry_points(self, monkeypatch):
        """Test plugin discovery with mocked entry points."""
        registry = PluginRegistry()
        
        # Mock entry point
        class MockEntryPoint:
            def __init__(self, name, plugin_class):
                self.name = name
                self._plugin_class = plugin_class
            
            def load(self):
                return self._plugin_class
        
        # Mock entry_points function
        class MockEntryPoints:
            def select(self, group):
                if group == 'bnsnlp.embed':
                    return [
                        MockEntryPoint('openai', ValidPlugin),
                        MockEntryPoint('cohere', AnotherValidPlugin)
                    ]
                return []
        
        def mock_entry_points():
            return MockEntryPoints()
        
        # Apply mock
        import importlib.metadata
        monkeypatch.setattr(importlib.metadata, 'entry_points', mock_entry_points)
        
        # Discover plugins
        registry.discover_plugins()
        
        # Verify plugins were registered
        plugins = registry.list_plugins('embed')
        assert 'openai' in plugins['embed']
        assert 'cohere' in plugins['embed']
    
    def test_discover_plugins_handles_load_errors(self, monkeypatch):
        """Test that plugin discovery continues when a plugin fails to load."""
        registry = PluginRegistry()
        
        # Mock entry point that fails to load
        class MockEntryPoint:
            def __init__(self, name, should_fail=False):
                self.name = name
                self.should_fail = should_fail
            
            def load(self):
                if self.should_fail:
                    raise ImportError("Failed to load plugin")
                return ValidPlugin
        
        # Mock entry_points function
        class MockEntryPoints:
            def select(self, group):
                if group == 'bnsnlp.embed':
                    return [
                        MockEntryPoint('working', False),
                        MockEntryPoint('broken', True),
                        MockEntryPoint('another_working', False)
                    ]
                return []
        
        def mock_entry_points():
            return MockEntryPoints()
        
        # Apply mock
        import importlib.metadata
        monkeypatch.setattr(importlib.metadata, 'entry_points', mock_entry_points)
        
        # Discover plugins - should not raise error
        with pytest.warns(RuntimeWarning, match="Failed to load plugin 'broken'"):
            registry.discover_plugins()
        
        # Verify working plugins were registered
        plugins = registry.list_plugins('embed')
        assert 'working' in plugins['embed']
        assert 'another_working' in plugins['embed']
        assert 'broken' not in plugins['embed']


class TestPluginRegistryIntegration:
    """Integration tests for plugin registry."""
    
    def test_register_retrieve_and_instantiate(self):
        """Test complete workflow: register, retrieve, and instantiate plugin."""
        registry = PluginRegistry()
        
        # Register plugin
        registry.register('embed', 'test_embedder', ValidPlugin)
        
        # Retrieve plugin class
        plugin_class = registry.get('embed', 'test_embedder')
        
        # Instantiate plugin
        config = {'api_key': 'test-key', 'model': 'test-model'}
        plugin_instance = plugin_class()
        plugin_instance.initialize(config)
        
        # Verify instance
        assert plugin_instance.name == 'test_plugin'
        assert plugin_instance.version == '1.0.0'
        assert plugin_instance.config == config
    
    def test_multiple_registries_are_independent(self):
        """Test that multiple registry instances are independent."""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()
        
        # Register plugin in first registry
        registry1.register('embed', 'plugin1', ValidPlugin)
        
        # Second registry should not have the plugin
        plugins2 = registry2.list_plugins('embed')
        assert 'plugin1' not in plugins2['embed']
        
        # Register different plugin in second registry
        registry2.register('embed', 'plugin2', AnotherValidPlugin)
        
        # First registry should not have the second plugin
        plugins1 = registry1.list_plugins('embed')
        assert 'plugin2' not in plugins1['embed']
