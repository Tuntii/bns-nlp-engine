"""
Plugin registry system for bns-nlp-engine.

This module provides the plugin registry that manages plugin discovery,
registration, and lifecycle for all plugin categories (preprocess, embed,
search, classify).
"""

import importlib.metadata
from typing import Any, Dict, List, Optional, Type

from bnsnlp.core.exceptions import PluginError
from bnsnlp.core.types import PluginInterface


class PluginRegistry:
    """
    Central registry for managing plugins.
    
    The registry maintains a category-based storage system for plugins
    and provides methods for registration, retrieval, and discovery.
    
    Supported categories:
        - preprocess: Text preprocessing plugins
        - embed: Text embedding plugins
        - search: Semantic search plugins
        - classify: Text classification plugins
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register('embed', 'openai', OpenAIEmbedder)
        >>> embedder_class = registry.get('embed', 'openai')
        >>> plugins = registry.list_plugins('embed')
    """
    
    def __init__(self) -> None:
        """Initialize the plugin registry with empty category storage."""
        self._plugins: Dict[str, Dict[str, Type[PluginInterface]]] = {
            'preprocess': {},
            'embed': {},
            'search': {},
            'classify': {}
        }
    
    def _validate_plugin(
        self,
        plugin_class: Type[PluginInterface],
        name: str,
        category: str
    ) -> None:
        """
        Validate that a plugin class implements the required interface.
        
        Checks that the plugin:
        1. Implements PluginInterface protocol
        2. Has 'name' attribute
        3. Has 'version' attribute
        4. Has 'initialize' method
        
        Args:
            plugin_class: Plugin class to validate
            name: Name of the plugin (for error messages)
            category: Category of the plugin (for error messages)
            
        Raises:
            PluginError: If plugin validation fails
        """
        # Check if plugin implements PluginInterface
        if not isinstance(plugin_class, type):
            raise PluginError(
                f"Plugin '{name}' must be a class, not {type(plugin_class).__name__}",
                context={
                    'category': category,
                    'name': name,
                    'type': str(type(plugin_class))
                }
            )
        
        # Check for required attributes
        missing_attrs = []
        
        # Check for 'name' attribute
        if not hasattr(plugin_class, 'name'):
            missing_attrs.append('name')
        
        # Check for 'version' attribute
        if not hasattr(plugin_class, 'version'):
            missing_attrs.append('version')
        
        # Check for 'initialize' method
        if not hasattr(plugin_class, 'initialize'):
            missing_attrs.append('initialize')
        
        if missing_attrs:
            raise PluginError(
                f"Plugin '{name}' missing required attributes: {', '.join(missing_attrs)}",
                context={
                    'category': category,
                    'name': name,
                    'missing_attributes': missing_attrs,
                    'required_attributes': ['name', 'version', 'initialize']
                }
            )
        
        # Validate that 'initialize' is callable
        if not callable(getattr(plugin_class, 'initialize', None)):
            raise PluginError(
                f"Plugin '{name}' 'initialize' attribute must be callable",
                context={
                    'category': category,
                    'name': name
                }
            )
    
    def register(
        self,
        category: str,
        name: str,
        plugin_class: Type[PluginInterface]
    ) -> None:
        """
        Register a plugin in the specified category.
        
        The plugin is validated to ensure it implements the PluginInterface
        and has required attributes (name, version) before registration.
        
        Args:
            category: Plugin category ('preprocess', 'embed', 'search', 'classify')
            name: Unique name for the plugin within its category
            plugin_class: Plugin class that implements PluginInterface
            
        Raises:
            PluginError: If category is invalid, plugin is already registered,
                        or plugin validation fails
            
        Example:
            >>> registry.register('embed', 'openai', OpenAIEmbedder)
        """
        # Validate category
        if category not in self._plugins:
            raise PluginError(
                f"Invalid plugin category: {category}",
                context={
                    'category': category,
                    'valid_categories': list(self._plugins.keys())
                }
            )
        
        # Validate plugin interface compliance
        self._validate_plugin(plugin_class, name, category)
        
        # Check if plugin already registered
        if name in self._plugins[category]:
            raise PluginError(
                f"Plugin '{name}' already registered in category '{category}'",
                context={
                    'category': category,
                    'name': name,
                    'existing_plugin': str(self._plugins[category][name])
                }
            )
        
        # Register the plugin
        self._plugins[category][name] = plugin_class
    
    def get(self, category: str, name: str) -> Type[PluginInterface]:
        """
        Retrieve a registered plugin by category and name.
        
        Args:
            category: Plugin category ('preprocess', 'embed', 'search', 'classify')
            name: Name of the plugin to retrieve
            
        Returns:
            Plugin class that implements PluginInterface
            
        Raises:
            PluginError: If category is invalid or plugin not found
            
        Example:
            >>> embedder_class = registry.get('embed', 'openai')
            >>> embedder = embedder_class(config={'api_key': 'sk-...'})
        """
        # Validate category
        if category not in self._plugins:
            raise PluginError(
                f"Invalid plugin category: {category}",
                context={
                    'category': category,
                    'valid_categories': list(self._plugins.keys())
                }
            )
        
        # Check if plugin exists
        if name not in self._plugins[category]:
            available_plugins = list(self._plugins[category].keys())
            raise PluginError(
                f"Plugin '{name}' not found in category '{category}'",
                context={
                    'category': category,
                    'name': name,
                    'available_plugins': available_plugins
                }
            )
        
        return self._plugins[category][name]
    
    def list_plugins(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered plugins, optionally filtered by category.
        
        Args:
            category: Optional category to filter by. If None, returns all plugins.
            
        Returns:
            Dictionary mapping categories to lists of plugin names
            
        Raises:
            PluginError: If specified category is invalid
            
        Example:
            >>> # List all plugins
            >>> all_plugins = registry.list_plugins()
            >>> # {'preprocess': ['turkish'], 'embed': ['openai', 'cohere'], ...}
            
            >>> # List plugins in specific category
            >>> embed_plugins = registry.list_plugins('embed')
            >>> # {'embed': ['openai', 'cohere', 'huggingface']}
        """
        if category is not None:
            # Validate category
            if category not in self._plugins:
                raise PluginError(
                    f"Invalid plugin category: {category}",
                    context={
                        'category': category,
                        'valid_categories': list(self._plugins.keys())
                    }
                )
            # Return only specified category
            return {category: list(self._plugins[category].keys())}
        
        # Return all categories
        return {
            cat: list(plugins.keys())
            for cat, plugins in self._plugins.items()
        }
    
    def discover_plugins(self) -> None:
        """
        Discover and load plugins via Python entry_points mechanism.
        
        This method scans for plugins registered via setuptools entry_points
        in the following groups:
            - bnsnlp.preprocess
            - bnsnlp.embed
            - bnsnlp.search
            - bnsnlp.classify
        
        Plugins are automatically registered in their respective categories.
        If a plugin fails to load, a warning is logged but discovery continues.
        
        Raises:
            PluginError: If there are critical errors during plugin discovery
            
        Example:
            >>> registry = PluginRegistry()
            >>> registry.discover_plugins()
            >>> # All plugins from entry_points are now registered
            >>> plugins = registry.list_plugins()
        """
        # Map entry point groups to categories
        entry_point_groups = {
            'bnsnlp.preprocess': 'preprocess',
            'bnsnlp.embed': 'embed',
            'bnsnlp.search': 'search',
            'bnsnlp.classify': 'classify'
        }
        
        for group, category in entry_point_groups.items():
            try:
                # Get all entry points for this group
                entry_points = importlib.metadata.entry_points()
                
                # Handle different return types based on Python version
                if hasattr(entry_points, 'select'):
                    # Python 3.10+ returns EntryPoints object with select method
                    group_entries = entry_points.select(group=group)
                else:
                    # Python 3.9 returns dict
                    group_entries = entry_points.get(group, [])
                
                # Load each entry point
                for entry_point in group_entries:
                    try:
                        # Load the plugin class
                        plugin_class = entry_point.load()
                        
                        # Register the plugin
                        # Use entry point name as plugin name
                        self.register(category, entry_point.name, plugin_class)
                        
                    except Exception as e:
                        # Log warning but continue with other plugins
                        # In production, this would use proper logging
                        import warnings
                        warnings.warn(
                            f"Failed to load plugin '{entry_point.name}' "
                            f"from group '{group}': {e}",
                            RuntimeWarning
                        )
                        
            except Exception as e:
                # Critical error in entry point discovery
                raise PluginError(
                    f"Failed to discover plugins for group '{group}'",
                    context={
                        'group': group,
                        'category': category,
                        'error': str(e)
                    }
                )


__all__ = ["PluginRegistry"]
