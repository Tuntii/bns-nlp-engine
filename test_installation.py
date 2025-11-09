#!/usr/bin/env python
"""Test script to verify package installation and basic functionality."""

import sys


def test_imports():
    """Test that all core imports work."""
    print("Testing imports...")
    try:
        # Core imports
        from bnsnlp import (
            Config,
            Pipeline,
            PluginRegistry,
            TurkishPreprocessor,
            BaseEmbedder,
            BaseSearch,
            TurkishClassifier,
            __version__,
        )

        print(f"✓ Core imports successful (version {__version__})")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from bnsnlp import Config

        config = Config()
        print(f"✓ Config created with log level: {config.logging.level}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_preprocessor():
    """Test preprocessor instantiation."""
    print("\nTesting preprocessor...")
    try:
        from bnsnlp import TurkishPreprocessor

        preprocessor = TurkishPreprocessor(
            {"lowercase": True, "remove_punctuation": True}
        )
        print("✓ TurkishPreprocessor instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Preprocessor test failed: {e}")
        return False


def test_pipeline():
    """Test pipeline creation."""
    print("\nTesting pipeline...")
    try:
        from bnsnlp import Config, Pipeline, PluginRegistry

        config = Config()
        registry = PluginRegistry()
        pipeline = Pipeline(config, registry)
        print("✓ Pipeline created successfully")
        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def test_plugin_registry():
    """Test plugin registry."""
    print("\nTesting plugin registry...")
    try:
        from bnsnlp import PluginRegistry

        registry = PluginRegistry()
        # Discover plugins
        registry.discover_plugins()
        plugins = registry.list_plugins()
        print(f"✓ Plugin registry working, found {sum(len(v) for v in plugins.values())} plugins")
        for category, plugin_list in plugins.items():
            if plugin_list:
                print(f"  - {category}: {', '.join(plugin_list)}")
        return True
    except Exception as e:
        print(f"✗ Plugin registry test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("BNS-NLP-Engine Installation Verification")
    print("=" * 70)

    tests = [
        test_imports,
        test_config,
        test_preprocessor,
        test_pipeline,
        test_plugin_registry,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)

    if all(results):
        print("\n✓ All tests passed! Package is correctly installed.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
