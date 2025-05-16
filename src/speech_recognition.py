import queue
import sounddevice as sd
import vosk
import sys
import json
import argparse
import os
from pathlib import Path
from download_model import setup_model, DEFAULT_MODEL, get_model_dir, list_installed_models, MODELS_ROOT_DIR

class ContinuousSpeechRecognizer:
    def __init__(self, 
                 model_name=DEFAULT_MODEL,
                 device=None, 
                 samplerate=16000,
                 blocksize=8000,
                 show_partial=True):
        """
        Initialize the speech recognizer
        
        Args:
            model_name (str): Name of the Vosk model to use
            device (int, optional): Audio input device ID. None for default device
            samplerate (int): Audio sample rate in Hz
            blocksize (int): Audio block size in samples
            show_partial (bool): Whether to show partial recognition results
        """
        model_path = self.ensure_model_exists(model_name)
        self.model = vosk.Model(str(model_path))
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.show_partial = show_partial
        self.q = queue.Queue()
        self.running = False

    @staticmethod
    def ensure_model_exists(model_name):
        """
        Check if the model exists, if not, download it
        Returns the path to the model directory
        """
        model_dir = get_model_dir(model_name)
        if not model_dir.exists() or not any(model_dir.iterdir()):
            print(f"Model '{model_name}' not found. Downloading...")
            try:
                setup_model(model_name)
            except Exception as e:
                print(f"Error downloading model: {e}")
                sys.exit(1)
        return model_dir

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def list_audio_devices(self):
        """List all available audio input devices"""
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']}")
                print(f"    Max input channels: {device['max_input_channels']}")
                print(f"    Default samplerate: {device['default_samplerate']}")
        print()

    def process_audio(self):
        try:
            rec = vosk.KaldiRecognizer(self.model, self.samplerate)
            
            # Print audio device info
            if self.device is not None:
                device_info = sd.query_devices(self.device, 'input')
                print(f"Using audio device: {device_info['name']}")
            
            with sd.RawInputStream(samplerate=self.samplerate,
                                 blocksize=self.blocksize,
                                 device=self.device,
                                 dtype='int16',
                                 channels=1,
                                 callback=self.callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('#' * 80)

                self.running = True
                while self.running:
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if result["text"]:
                            self.on_speech_detected(result["text"])
                    else:
                        if self.show_partial:
                            partial = json.loads(rec.PartialResult())
                            if partial["partial"]:
                                self.on_partial_speech(partial["partial"])

        except KeyboardInterrupt:
            print('\nDone')
        except Exception as e:
            print(f"Error: {str(e)}")

    def on_speech_detected(self, text):
        """Override this method to handle detected speech"""
        print(f"Detected: {text}")

    def on_partial_speech(self, text):
        """Override this method to handle partial speech"""
        print(f"Partial: {text}")

    def stop(self):
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="Continuous Speech Recognition using Vosk")
    
    parser.add_argument("--model", default=DEFAULT_MODEL,
                      help=f"Name of the Vosk model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", type=int,
                      help="Audio input device ID (default: system default)")
    parser.add_argument("--samplerate", type=int, default=16000,
                      help="Audio sample rate in Hz (default: 16000)")
    parser.add_argument("--blocksize", type=int, default=8000,
                      help="Audio block size in samples (default: 8000)")
    parser.add_argument("--no-partial", action="store_true",
                      help="Disable partial recognition results")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio input devices and exit")
    parser.add_argument("--list-models", action="store_true",
                      help="List installed Vosk models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_installed_models()
        return
    
    if args.list_devices:
        recognizer = ContinuousSpeechRecognizer(model_name=args.model)
        recognizer.list_audio_devices()
        return
    
    try:
        recognizer = ContinuousSpeechRecognizer(
            model_name=args.model,
            device=args.device,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            show_partial=not args.no_partial
        )
        recognizer.process_audio()
    except KeyboardInterrupt:
        print('\nStopping...')
        sys.exit(0)

if __name__ == "__main__":
    main() 