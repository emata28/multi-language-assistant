import queue
import sounddevice as sd
import numpy as np
import whisper
import torch
import sys
import json
import argparse
from typing import Optional, Dict
from pathlib import Path

class WhisperSpeechRecognizer:
    def __init__(self,
                 model_name: str = "base",
                 device: Optional[int] = None,
                 samplerate: int = 16000,
                 blocksize: int = 8000,
                 language: Optional[str] = None,
                 task: str = "transcribe"):
        """
        Initialize the Whisper-based speech recognizer
        
        Args:
            model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
            device (int, optional): Audio input device ID. None for default device
            samplerate (int): Audio sample rate in Hz
            blocksize (int): Audio block size in samples
            language (str, optional): Language code (e.g., 'en', 'es'). None for auto-detection
            task (str): Either 'transcribe' or 'translate' (translate will convert any language to English)
        """
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
        
        # Set torch.load to use weights_only=True for security
        torch.serialization._weights_only_default = True
        self.model = whisper.load_model(model_name).to(self.device)
        self.language = language
        self.task = task
        
        # Audio settings
        self.audio_device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
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
        print(f"- Block size: {blocksize} samples")
        
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
                    audio_block = self.q.get()
                    
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
                            # Transcribe
                            result = self.model.transcribe(
                                audio_data,
                                language=self.language,  # None will enable auto-detection
                                task=self.task,  # 'transcribe' or 'translate'
                                fp16=self.cuda_available  # Use fp16 if CUDA is available
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
    parser.add_argument("--blocksize", type=int, default=8000,
                      help="Audio block size in samples (default: 8000)")
    parser.add_argument("--language", type=str,
                      help="Language code (e.g., 'en', 'es'). Default: auto-detect")
    parser.add_argument("--translate", action="store_true",
                      help="Translate speech to English (default: transcribe in original language)")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio input devices and exit")
    
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
            task="translate" if args.translate else "transcribe"
        )
        recognizer.process_audio()
    except KeyboardInterrupt:
        print('\nStopping...')
        sys.exit(0)

if __name__ == "__main__":
    main() 