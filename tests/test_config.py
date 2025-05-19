import pytest
import os
import tempfile
from pathlib import Path
from src.utils.config import load_config, save_config, validate_config, get_default_config

def test_get_default_config():
    """Test getting default configuration."""
    config = get_default_config()
    
    # Check required sections
    assert "speech" in config
    assert "llm" in config
    assert "commands" in config
    
    # Check speech configuration
    assert "model" in config["speech"]
    assert "language" in config["speech"]
    assert "device" in config["speech"]
    
    # Check LLM configuration
    assert "provider" in config["llm"]
    assert "model" in config["llm"]
    assert "temperature" in config["llm"]
    
    # Check commands configuration
    assert "spotify" in config["commands"]
    assert "smart_home" in config["commands"]
    assert "system" in config["commands"]

def test_validate_config():
    """Test configuration validation."""
    # Valid configuration
    valid_config = get_default_config()
    validate_config(valid_config)
    
    # Missing required section
    invalid_config = valid_config.copy()
    del invalid_config["speech"]
    with pytest.raises(ValueError, match="Missing required configuration section: speech"):
        validate_config(invalid_config)
    
    # Missing required field
    invalid_config = valid_config.copy()
    del invalid_config["speech"]["model"]
    with pytest.raises(ValueError, match="Missing required speech configuration: model"):
        validate_config(invalid_config)
    
    # Invalid commands configuration
    invalid_config = valid_config.copy()
    invalid_config["commands"] = "not a dictionary"
    with pytest.raises(ValueError, match="Commands configuration must be a dictionary"):
        validate_config(invalid_config)

def test_save_and_load_config():
    """Test saving and loading configuration."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.yaml"
        
        # Get default configuration
        config = get_default_config()
        
        # Save configuration
        save_config(config, str(config_path))
        assert config_path.exists()
        
        # Load configuration
        loaded_config = load_config(str(config_path))
        
        # Compare configurations
        assert loaded_config == config

def test_load_nonexistent_config():
    """Test loading nonexistent configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")

def test_save_config_to_nonexistent_directory():
    """Test saving configuration to nonexistent directory."""
    config = get_default_config()
    config_path = "nonexistent/dir/config.yaml"
    
    # Should create directory and save file
    save_config(config, config_path)
    assert Path(config_path).exists()
    
    # Clean up
    Path(config_path).unlink()
    Path("nonexistent/dir").rmdir()
    Path("nonexistent").rmdir() 