"""Shared fixtures for Voca test suites."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for stages that need a config_dir."""
    config_dir = tmp_path / "voca_config"
    config_dir.mkdir()
    return config_dir
