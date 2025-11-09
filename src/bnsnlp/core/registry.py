"""
Plugin registry system for bns-nlp-engine.

This module provides the plugin registry that manages plugin discovery,
registration, and lifecycle for all plugin categories (preprocess, embed,
search, classify).
"""

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
    
    def register(
        self,
        category: str,
        name: str,
        plugin_class: Type[PluginInterface]
    ) -> None:
        """
        Register a plugin in the specified category.
        
        Args:
            category: Plugin category ('preprocess', 'embed', 'search', 'classify')
            name: Unique name for the plugin within its category
            plugin_class: Plugin class that implements PluginInterface
            
        Raises:
            PluginError: If category is invalid or plugin is already registered
            
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


__all__ = ["PluginRegistry"]
