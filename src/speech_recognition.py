import queue
import sounddevice as sd
import numpy as np
import torch
import warnings
import psutil
# Set torch.load to use weights_only=True for security before importing whisper
torch.serialization._weights_only_default = True
# Suppress the specific torch.load warning from whisper
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
import whisper
import sys
import json
import argparse
from typing import Optional, Dict, Tuple
from pathlib import Path

def get_optimal_settings(model_name: str) -> Tuple[int, bool, int]:
    """
    Automatically determine optimal settings based on hardware and model size.
    
    Returns:
        Tuple[int, bool, int]: (batch_size, use_fp16, blocksize)
    """
    # Get system information
    total_memory = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    cuda_available = torch.cuda.is_available()
    gpu_memory = 0
    
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f}GB")
    else:
        print("\nNo GPU detected, using CPU")
    
    print(f"System memory: {total_memory:.1f}GB")
    
    # Model size requirements (approximate)
    model_sizes = {
        "tiny": {"gpu_mem": 1, "cpu_mem": 2},
        "base": {"gpu_mem": 1.5, "cpu_mem": 3},
        "small": {"gpu_mem": 2, "cpu_mem": 4},
        "medium": {"gpu_mem": 5, "cpu_mem": 8},
        "large": {"gpu_mem": 10, "cpu_mem": 16}
    }
    
    # Get model requirements
    model_req = model_sizes.get(model_name, model_sizes["base"])
    
    # Determine if we can use FP16
    use_fp16 = cuda_available and gpu_memory >= model_req["gpu_mem"]
    
    # Determine optimal batch size
    if cuda_available:
        if model_name == "large":
            batch_size = 1 if gpu_memory < 12 else 2
        elif model_name == "medium":
            batch_size = 2 if gpu_memory >= 8 else 1
        else:
            batch_size = 4 if gpu_memory >= 6 else 2
    else:
        # For CPU, use smaller batch sizes
        batch_size = 1
    
    # Determine optimal block size
    if cuda_available:
        blocksize = 8000 if gpu_memory >= 6 else 4000
    else:
        blocksize = 4000  # Smaller blocks for CPU
    
    print(f"\nOptimized settings for {model_name} model:")
    print(f"- Batch size: {batch_size}")
    print(f"- FP16 enabled: {use_fp16}")
    print(f"- Block size: {blocksize}")
    
    return batch_size, use_fp16, blocksize

class WhisperSpeechRecognizer:
    def __init__(self,
                 model_name: str = "base",
                 device: Optional[int] = None,
                 samplerate: int = 16000,
                 blocksize: Optional[int] = None,
                 language: Optional[str] = None,
                 task: str = "transcribe",
                 batch_size: Optional[int] = None,
                 use_fp16: Optional[bool] = None):
        """
        Initialize the Whisper-based speech recognizer
        
        Args:
            model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
            device (int, optional): Audio input device ID. None for default device
            samplerate (int): Audio sample rate in Hz
            blocksize (int, optional): Audio block size in samples. None for automatic
            language (str, optional): Language code (e.g., 'en', 'es'). None for auto-detection
            task (str): Either 'transcribe' or 'translate' (translate will convert any language to English)
            batch_size (int, optional): Number of audio chunks to process in parallel. None for automatic
            use_fp16 (bool, optional): Whether to use half-precision (FP16). None for automatic
        """
        # Get optimal settings
        optimal_batch_size, optimal_fp16, optimal_blocksize = get_optimal_settings(model_name)
        
        # Use provided values or optimal defaults
        self.batch_size = batch_size if batch_size is not None else optimal_batch_size
        self.use_fp16 = use_fp16 if use_fp16 is not None else optimal_fp16
        self.blocksize = blocksize if blocksize is not None else optimal_blocksize
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        
        # Print system information
        print("\nSystem Information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {self.cuda_available}")
        if self.cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {self.device}")
        
        # Store model name and load Whisper model
        self.model_name = model_name
        print(f"\nLoading Whisper model '{model_name}' on {self.device}...")
        
        # Load model with optimizations
        self.model = whisper.load_model(model_name, device=self.device)
        if self.use_fp16 and self.cuda_available:
            self.model = self.model.half()  # Convert to FP16 for faster inference
        
        self.language = language
        self.task = task
        
        # Audio settings
        self.audio_device = device
        self.samplerate = samplerate
        self.q = queue.Queue()
        self.running = False
        
        # Buffer for collecting audio
        self.audio_buffer = []
        self.silence_threshold = 0.01
        self.silence_frames = 0
        self.MIN_SILENCE_FRAMES = 3

        # Print configuration
        print("\nConfiguration:")
        print(f"- Model: {model_name}")
        print(f"- Language: {'Auto-detect' if language is None else language}")
        print(f"- Task: {task} {'(will translate to English)' if task == 'translate' else ''}")
        print(f"- Sample rate: {samplerate} Hz")
        print(f"- Block size: {self.blocksize} samples")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Using FP16: {self.use_fp16 and self.cuda_available}")
        
    def callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(status)
        # Convert to float32 before adding to queue
        self.q.put(indata.astype(np.float32).copy())
    
    def process_audio(self):
        """Process audio input and perform real-time transcription"""
        try:
            # Print audio device info
            if self.audio_device is not None:
                device_info = sd.query_devices(self.audio_device, 'input')
                print(f"\nUsing audio device: {device_info['name']}")
            
            print("\nListening... Press Ctrl+C to stop")
            print("Speak in any language - Whisper will automatically detect it!")
            
            self.running = True
            
            with sd.InputStream(samplerate=self.samplerate,
                              blocksize=self.blocksize,
                              device=self.audio_device,
                              dtype=np.float32,
                              channels=1,
                              callback=self.callback):
                
                while self.running:
                    audio_blocks = []
                    # Collect batch_size number of blocks
                    for _ in range(self.batch_size):
                        try:
                            audio_block = self.q.get(timeout=0.1)
                            audio_blocks.append(audio_block)
                        except queue.Empty:
                            break
                    
                    if not audio_blocks:
                        continue
                    
                    # Process all blocks in the batch
                    for audio_block in audio_blocks:
                        # Check if it's silence
                        is_silence = np.abs(audio_block).mean() < self.silence_threshold
                        
                        if is_silence:
                            self.silence_frames += 1
                        else:
                            self.silence_frames = 0
                        
                        # Add audio to buffer
                        self.audio_buffer.extend(audio_block.flatten().tolist())
                    
                    # Process if we have enough silence
                    if self.silence_frames >= self.MIN_SILENCE_FRAMES and len(self.audio_buffer) > self.samplerate:
                        # Convert buffer to numpy array with explicit float32 dtype
                        audio_data = np.array(self.audio_buffer, dtype=np.float32)
                        
                        # Ensure audio is in the correct range [-1, 1]
                        if np.abs(audio_data).max() > 1.0:
                            audio_data = np.clip(audio_data, -1.0, 1.0)
                        
                        try:
                            # Transcribe with optimizations
                            with torch.cuda.amp.autocast(enabled=self.cuda_available):
                                result = self.model.transcribe(
                                    audio_data,
                                    language=self.language,
                                    task=self.task,
                                    fp16=self.cuda_available
                                )
                            
                            if result["text"].strip():
                                detected_language = result.get("language", "unknown")
                                print(f"\nDetected Language: {detected_language}")
                                print(f"{'Transcribed' if self.task == 'transcribe' else 'Translated'}: {result['text'].strip()}")
                        except Exception as e:
                            print(f"\nError during transcription: {str(e)}")
                        
                        # Clear buffer
                        self.audio_buffer = []
                        self.silence_frames = 0
                    
        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False
        except Exception as e:
            print(f"\nError: {str(e)}")
            self.running = False
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")

def main():
    parser = argparse.ArgumentParser(description="Multilingual Speech Recognition using OpenAI Whisper")
    
    parser.add_argument("--model", default="base",
                      help="Whisper model name (tiny, base, small, medium, large)")
    parser.add_argument("--device", type=int,
                      help="Audio input device ID (default: system default)")
    parser.add_argument("--samplerate", type=int, default=16000,
                      help="Audio sample rate in Hz (default: 16000)")
    parser.add_argument("--blocksize", type=int,
                      help="Audio block size in samples (default: automatic)")
    parser.add_argument("--language", type=str,
                      help="Language code (e.g., 'en', 'es'). Default: auto-detect")
    parser.add_argument("--translate", action="store_true",
                      help="Translate speech to English (default: transcribe in original language)")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio input devices and exit")
    parser.add_argument("--batch-size", type=int,
                      help="Number of audio chunks to process in parallel (default: automatic)")
    parser.add_argument("--no-fp16", action="store_true",
                      help="Disable FP16 optimization (default: automatic)")
    
    args = parser.parse_args()
    
    if args.list_devices:
        recognizer = WhisperSpeechRecognizer()
        recognizer.list_audio_devices()
        return
    
    try:
        recognizer = WhisperSpeechRecognizer(
            model_name=args.model,
            device=args.device,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            language=args.language,
            task="translate" if args.translate else "transcribe",
            batch_size=args.batch_size,
            use_fp16=not args.no_fp16 if args.no_fp16 else None
        )
        recognizer.process_audio()
    except KeyboardInterrupt:
        print('\nStopping...')
        sys.exit(0)

if __name__ == "__main__":
    main() 