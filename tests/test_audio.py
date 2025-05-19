import pytest
import numpy as np
from src.utils.audio import AudioInput, list_audio_devices

def test_list_audio_devices():
    """Test listing audio devices."""
    devices = list_audio_devices()
    assert isinstance(devices, dict)
    assert all(isinstance(k, int) for k in devices.keys())
    assert all(isinstance(v, str) for v in devices.values())

def test_audio_input_initialization():
    """Test audio input initialization."""
    config = {
        "samplerate": 16000,
        "blocksize": 8000,
        "device": None
    }
    
    audio_input = AudioInput(config)
    assert audio_input.samplerate == 16000
    assert audio_input.blocksize == 8000
    assert audio_input.channels == 1
    assert not audio_input.is_recording

def test_audio_input_context_manager():
    """Test audio input context manager."""
    config = {
        "samplerate": 16000,
        "blocksize": 8000,
        "device": None
    }
    
    with AudioInput(config) as audio_input:
        assert audio_input.is_recording
        # Get some audio data
        audio_data = audio_input.get_audio_data(timeout=1.0)
        if audio_data is not None:
            assert isinstance(audio_data, np.ndarray)
            assert audio_data.dtype == np.float32
    
    assert not audio_input.is_recording

def test_audio_input_queue_operations():
    """Test audio input queue operations."""
    config = {
        "samplerate": 16000,
        "blocksize": 8000,
        "device": None
    }
    
    audio_input = AudioInput(config)
    audio_input.start_recording()
    
    # Test getting audio data with timeout
    audio_data = audio_input.get_audio_data(timeout=1.0)
    if audio_data is not None:
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.dtype == np.float32
    
    # Test clearing queue
    audio_input.clear_queue()
    assert audio_input.audio_queue.empty()
    
    audio_input.stop_recording()

def test_audio_input_invalid_config():
    """Test audio input with invalid configuration."""
    config = {
        "samplerate": -1,  # Invalid sample rate
        "blocksize": 0,    # Invalid block size
        "device": 999999   # Invalid device
    }
    
    with pytest.raises(Exception):
        AudioInput(config) 