import pytest
import logging
import os
import tempfile
from pathlib import Path
from src.utils.logging import setup_logging, get_logger

def test_setup_logging():
    """Test logging setup."""
    # Create temporary directory for log file
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        
        # Test configuration
        config = {
            "commands": {
                "logging": {
                    "level": "DEBUG",
                    "file": str(log_file),
                    "max_size": 1024,
                    "backup_count": 2
                }
            }
        }
        
        # Setup logging
        setup_logging(config)
        
        # Check root logger
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        # Check handlers
        assert len(root_logger.handlers) == 2
        file_handler = next(h for h in root_logger.handlers if isinstance(h, logging.FileHandler))
        console_handler = next(h for h in root_logger.handlers if isinstance(h, logging.StreamHandler))
        
        assert isinstance(file_handler, logging.FileHandler)
        assert isinstance(console_handler, logging.StreamHandler)
        
        # Test logging
        test_message = "Test log message"
        logging.info(test_message)
        
        # Check log file
        assert log_file.exists()
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert test_message in log_content

def test_get_logger():
    """Test getting logger instance."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"

def test_logging_levels():
    """Test different logging levels."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        
        config = {
            "commands": {
                "logging": {
                    "level": "DEBUG",
                    "file": str(log_file),
                    "max_size": 1024,
                    "backup_count": 2
                }
            }
        }
        
        setup_logging(config)
        logger = get_logger("test_module")
        
        # Test different logging levels
        test_messages = {
            "debug": "Debug message",
            "info": "Info message",
            "warning": "Warning message",
            "error": "Error message",
            "critical": "Critical message"
        }
        
        logger.debug(test_messages["debug"])
        logger.info(test_messages["info"])
        logger.warning(test_messages["warning"])
        logger.error(test_messages["error"])
        logger.critical(test_messages["critical"])
        
        # Check log file
        with open(log_file, 'r') as f:
            log_content = f.read()
            for message in test_messages.values():
                assert message in log_content

def test_logging_rotation():
    """Test log file rotation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        
        config = {
            "commands": {
                "logging": {
                    "level": "INFO",
                    "file": str(log_file),
                    "max_size": 100,  # Small size to trigger rotation
                    "backup_count": 2
                }
            }
        }
        
        setup_logging(config)
        logger = get_logger("test_module")
        
        # Generate enough log messages to trigger rotation
        for i in range(10):
            logger.info(f"Test message {i}" * 20)  # Make messages large
        
        # Check log files
        assert log_file.exists()
        backup_files = list(log_file.parent.glob("test.log.*"))
        assert len(backup_files) <= 2  # Should not exceed backup_count 