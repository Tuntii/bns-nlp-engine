"""
Pytest configuration and fixtures for bns-nlp-engine tests.
"""

import asyncio
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_turkish_text() -> str:
    """Sample Turkish text for testing."""
    return "Türkçe doğal dil işleme kütüphanesi"


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create temporary directory for config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir
