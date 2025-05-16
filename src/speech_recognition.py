import queue
import sounddevice as sd
import vosk
import sys
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List
from download_model import setup_model, get_model_dir, list_installed_models, MODELS_ROOT_DIR

# Default models for English and Spanish
DEFAULT_EN_MODEL = "vosk-model-en-us-0.22"
DEFAULT_ES_MODEL = "vosk-model-es-0.42"
DEFAULT_MODELS = [DEFAULT_EN_MODEL, DEFAULT_ES_MODEL]

class MultilingualSpeechRecognizer:
    def __init__(self, 
                 model_names: List[str] = DEFAULT_MODELS,
                 device=None, 
                 samplerate=16000,
                 blocksize=8000,
                 show_partial=True,
                 min_confidence=0.5,
                 min_word_length=2):
        """
        Initialize the multilingual speech recognizer
        
        Args:
            model_names (List[str]): List of Vosk model names to use
            device (int, optional): Audio input device ID. None for default device
            samplerate (int): Audio sample rate in Hz
            blocksize (int): Audio block size in samples
            show_partial (bool): Whether to show partial recognition results
            min_confidence (float): Minimum confidence threshold (0.0 to 1.0)
            min_word_length (int): Minimum length of words to consider as valid speech
        """
        self.models: Dict[str, vosk.Model] = {}
        self.recognizers: Dict[str, vosk.KaldiRecognizer] = {}
        
        # Initialize models for each language
        for model_name in model_names:
            model_path = self.ensure_model_exists(model_name)
            self.models[model_name] = vosk.Model(str(model_path))
        
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.show_partial = show_partial
        self.min_confidence = min_confidence
        self.min_word_length = min_word_length
        self.q = queue.Queue()
        self.running = False
        self.last_text = ""
        self.silence_frames = 0
        self.MIN_SILENCE_FRAMES = 3

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

    def is_valid_speech(self, text, confidence=None):
        """
        Check if the detected speech is valid based on various criteria
        """
        # Skip if text is empty or just whitespace
        if not text or text.isspace():
            return False
            
        # Skip if confidence is too low
        if confidence is not None and confidence < self.min_confidence:
            return False
            
        # Split into words and filter
        words = text.split()
        
        # Skip if all words are too short
        if all(len(word) < self.min_word_length for word in words):
            return False
            
        # Skip common false positives when they appear alone
        # Include common Spanish interjections
        common_false_positives = {'the', 'a', 'an', 'uh', 'um', 'eh', 'ah', 'eh', 'ay', 'oh'}
        if len(words) == 1 and words[0].lower() in common_false_positives:
            return False
            
        return True

    def calculate_confidence(self, text: str) -> float:
        """
        Calculate a confidence score based on text characteristics
        Returns a value between 0.0 and 1.0
        """
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        # Factors that increase confidence:
        # 1. Number of words (more words = higher confidence)
        # 2. Average word length (longer words = higher confidence)
        # 3. No common false positives
        
        # Calculate word-based metrics
        num_words = len(words)
        avg_word_length = sum(len(word) for word in words) / num_words
        
        # Common false positives reduce confidence
        common_false_positives = {'the', 'a', 'an', 'uh', 'um', 'eh', 'ah', 'eh', 'ay', 'oh'}
        false_positive_ratio = sum(1 for word in words if word.lower() in common_false_positives) / num_words
        
        # Calculate base confidence
        word_count_factor = min(num_words / 5, 1.0)  # Max out at 5 words
        length_factor = min(avg_word_length / 5, 1.0)  # Max out at 5 chars average
        false_positive_penalty = false_positive_ratio * 0.5
        
        # Combine factors
        confidence = (word_count_factor * 0.6 + length_factor * 0.4) * (1 - false_positive_penalty)
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def process_audio(self):
        try:
            # Initialize recognizers for each model
            for model_name, model in self.models.items():
                self.recognizers[model_name] = vosk.KaldiRecognizer(model, self.samplerate)
            
            # Print audio device info
            if self.device is not None:
                device_info = sd.query_devices(self.device, 'input')
                print(f"Using audio device: {device_info['name']}")
            
            print("\nInitialized models:")
            for model_name in self.models.keys():
                print(f"- {model_name}")
            
            with sd.RawInputStream(samplerate=self.samplerate,
                                 blocksize=self.blocksize,
                                 device=self.device,
                                 dtype='int16',
                                 channels=1,
                                 callback=self.callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('Ready to recognize both English and Spanish!')
                print('#' * 80)

                self.running = True
                while self.running:
                    data = self.q.get()
                    
                    # Process audio with all recognizers and store results
                    recognition_results = {}
                    for model_name, recognizer in self.recognizers.items():
                        if recognizer.AcceptWaveform(bytes(data)):
                            result = json.loads(recognizer.Result())
                            text = result.get("text", "").strip()
                            
                            if text:  # Only store non-empty results
                                recognition_results[model_name] = {
                                    'text': text,
                                    'word_count': len(text.split())
                                }
                    
                    # If we have results, determine the most likely language
                    if recognition_results:
                        # Find the model with the most words recognized
                        best_model = max(recognition_results.items(), 
                                       key=lambda x: x[1]['word_count'], 
                                       default=(None, None))[0]
                        
                        if best_model:
                            text = recognition_results[best_model]['text']
                            if self.is_valid_speech(text):
                                self.on_speech_detected(text, best_model, recognition_results)
                                self.last_text = text
                                self.silence_frames = 0
                            else:
                                self.silence_frames += 1

        except KeyboardInterrupt:
            print('\nDone')
        except Exception as e:
            print(f"Error: {str(e)}")

    def on_speech_detected(self, text: str, model_name: str, recognition_results: dict):
        """Override this method to handle detected speech"""
        # Add language label for clearer output
        lang = "🇺🇸" if model_name == DEFAULT_EN_MODEL else "🇪🇸" if model_name == DEFAULT_ES_MODEL else "❓"
        
        # Get word counts for comparison
        en_words = recognition_results.get(DEFAULT_EN_MODEL, {}).get('word_count', 0)
        es_words = recognition_results.get(DEFAULT_ES_MODEL, {}).get('word_count', 0)
        
        # Calculate language certainty based on relative difference between word counts
        total_words = en_words + es_words
        if total_words > 0:
            if model_name == DEFAULT_EN_MODEL:
                certainty = (en_words - es_words) / total_words * 100
            else:
                certainty = (es_words - en_words) / total_words * 100
            # Normalize certainty to be between 0 and 100
            certainty = max(0, min(100, 50 + certainty))
        else:
            certainty = 0
        
        print(f"{lang} Detected: {text}")
        print(f"Language detected: {'English' if model_name == DEFAULT_EN_MODEL else 'Spanish' if model_name == DEFAULT_ES_MODEL else 'Unknown'}")
        print(f"Language certainty: {round(certainty, 2)}%")
        print(f"Word count comparison - English: {en_words}, Spanish: {es_words}")
        print("-" * 50)

    def on_partial_speech(self, text: str, model_name: str):
        """Override this method to handle partial speech"""
        # Add language label for clearer output
        lang = "🇺🇸" if model_name == DEFAULT_EN_MODEL else "🇪🇸" if model_name == DEFAULT_ES_MODEL else "❓"
        print(f"{lang} Partial: {text}")

    def stop(self):
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="Multilingual Speech Recognition using Vosk")
    
    parser.add_argument("--models", nargs='+', default=DEFAULT_MODELS,
                      help=f"List of model names to use (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--device", type=int,
                      help="Audio input device ID (default: system default)")
    parser.add_argument("--samplerate", type=int, default=16000,
                      help="Audio sample rate in Hz (default: 16000)")
    parser.add_argument("--blocksize", type=int, default=8000,
                      help="Audio block size in samples (default: 8000)")
    parser.add_argument("--no-partial", action="store_true",
                      help="Disable partial recognition results")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                      help="Minimum confidence threshold (0.0 to 1.0)")
    parser.add_argument("--min-word-length", type=int, default=2,
                      help="Minimum length of words to consider as valid speech")
    parser.add_argument("--list-devices", action="store_true",
                      help="List available audio input devices and exit")
    parser.add_argument("--list-models", action="store_true",
                      help="List installed Vosk models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_installed_models()
        return
    
    if args.list_devices:
        recognizer = MultilingualSpeechRecognizer()
        recognizer.list_audio_devices()
        return
    
    try:
        recognizer = MultilingualSpeechRecognizer(
            model_names=args.models,
            device=args.device,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            show_partial=not args.no_partial,
            min_confidence=args.min_confidence,
            min_word_length=args.min_word_length
        )
        recognizer.process_audio()
    except KeyboardInterrupt:
        print('\nStopping...')
        sys.exit(0)

if __name__ == "__main__":
    main() 