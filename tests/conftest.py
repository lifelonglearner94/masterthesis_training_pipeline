"""Pytest configuration and fixtures."""

import pytest
import pyrootutils

# Setup root for tests
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary for testing."""
    return {
        "seed": 42,
        "train": True,
        "test": False,
    }
