import pytest
import tempfile
from pathlib import Path
from src.utils.config import get_default_config, save_config

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def test_config(temp_config_dir):
    """Create a test configuration file."""
    config = get_default_config()
    config_path = temp_config_dir / "test_config.yaml"
    save_config(config, str(config_path))
    return config_path

@pytest.fixture
def test_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def audio_config():
    """Create a test audio configuration."""
    return {
        "samplerate": 16000,
        "blocksize": 8000,
        "device": None
    }

@pytest.fixture
def logging_config(test_log_dir):
    """Create a test logging configuration."""
    return {
        "commands": {
            "logging": {
                "level": "DEBUG",
                "file": str(test_log_dir / "test.log"),
                "max_size": 1024,
                "backup_count": 2
            }
        }
    } 