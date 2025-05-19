import torch
import whisper
import sounddevice as sd
import numpy as np
import queue
import time
import psutil
import re

# Auto-select best Whisper model based on hardware

def select_whisper_model():
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 10:
            return "large-v2"
        elif vram_gb >= 5:
            return "medium"
        elif vram_gb >= 3:
            return "small"
        else:
            return "base"
    else:
        ram_gb = psutil.virtual_memory().total / 1e9
        if ram_gb >= 16:
            return "medium"
        elif ram_gb >= 8:
            return "small"
        else:
            return "base"

# Noise/output filter: only accept non-empty, non-trivial, high-confidence results

def is_valid_transcription(result):
    """
    Smart filtering of transcription results.
    Filters out:
    - Single words or very short phrases
    - Low confidence transcriptions
    - Incomplete sentences
    - Common noise patterns
    """
    text = result.get("text", "").strip()
    
    # Skip empty or very short text
    if not text or len(text) < 3:
        return False
        
    # Count words (handle multiple spaces and punctuation)
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 2:  # Require at least 2 words
        return False
        
    # Check confidence score if available
    if "avg_logprob" in result:
        if result["avg_logprob"] < -1.0:  # More lenient threshold
            return False
            
    # Check for sentence structure
    # Look for common sentence endings
    if not any(text.rstrip().endswith(end) for end in ['.', '?', '!', '...']):
        # If no sentence ending, check if it's a complete phrase
        if not any(text.lower().startswith(start) for start in ['can you', 'could you', 'please', 'i want', 'i need']):
            return False
            
    # Check for repeated words (common in noise)
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
        if word_counts[word] > 2:  # More than 2 repetitions of the same word
            return False
            
    return True

class WhisperListener:
    def __init__(self, samplerate=16000, blocksize=4000, device=None, on_command=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.q = queue.Queue()
        self.running = False
        self.model_name = select_whisper_model()
        print(f"[Whisper] Using model: {self.model_name}")
        self.model = whisper.load_model(self.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.last_transcription = ""
        self.last_transcription_time = 0
        self.transcription_buffer = []
        self.silence_threshold = 0.01
        self.silence_duration = 0
        self.min_silence_duration = 1.0  # Wait for 1 second of silence before processing
        self.on_command = on_command  # Callback for when a command is recognized

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio] Status: {status}")
        # Calculate audio level to detect silence
        audio_level = np.abs(indata).mean()
        if audio_level > self.silence_threshold:
            self.silence_duration = 0
        else:
            self.silence_duration += frames / self.samplerate
        self.q.put(indata.copy())

    def combine_transcriptions(self, new_text):
        """Combine consecutive transcriptions that are part of the same phrase."""
        current_time = time.time()
        
        # If it's been more than 3 seconds since last transcription, clear buffer
        if current_time - self.last_transcription_time > 3.0:
            self.transcription_buffer = []
            
        self.last_transcription_time = current_time
        
        # Clean and normalize the new text
        new_text = new_text.strip().lower()
        
        # Skip if it's just noise words
        if new_text in ['you', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
            return " ".join(self.transcription_buffer) if self.transcription_buffer else ""
            
        # Add new text to buffer if it's not empty
        if new_text:
            # If the new text starts with a word that's at the end of the buffer, remove the duplicate
            if self.transcription_buffer:
                last_words = self.transcription_buffer[-1].split()
                new_words = new_text.split()
                if last_words and new_words and last_words[-1] == new_words[0]:
                    new_text = " ".join(new_words[1:])
            
            self.transcription_buffer.append(new_text)
            
        # Combine buffer into a single phrase
        combined = " ".join(self.transcription_buffer)
        
        # Clear buffer if we have a complete phrase
        if (any(combined.endswith(end) for end in ['.', '?', '!', '...']) or
            any(combined.startswith(start) for start in ['can you', 'could you', 'please', 'i want', 'i need', 'necesito'])):
            self.transcription_buffer = []
            
        return combined

    def is_valid_transcription(self, result):
        """
        Smart filtering of transcription results.
        Filters out:
        - Single words or very short phrases
        - Low confidence transcriptions
        - Incomplete sentences
        - Common noise patterns
        """
        text = result.get("text", "").strip().lower()
        
        # Skip empty or very short text
        if not text or len(text) < 2:
            return False
            
        # Skip if it's just noise words
        if text in ['you', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
            return False
            
        # Count words (handle multiple spaces and punctuation)
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 1:  # Allow single words for Spanish
            return False
            
        # Check confidence score if available
        if "avg_logprob" in result:
            if result["avg_logprob"] < -1.0:  # More lenient threshold
                return False
                
        # Check for sentence structure
        # Look for common sentence endings
        if not any(text.rstrip().endswith(end) for end in ['.', '?', '!', '...']):
            # If no sentence ending, check if it's a complete phrase
            if not any(text.startswith(start) for start in ['can you', 'could you', 'please', 'i want', 'i need', 'necesito']):
                return False
                
        # Check for repeated words (common in noise)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > 2:  # More than 2 repetitions of the same word
                return False
                
        return True

    def listen(self):
        self.running = True
        with sd.InputStream(samplerate=self.samplerate, channels=1, blocksize=self.blocksize, dtype='float32', callback=self.audio_callback, device=self.device):
            print("[Whisper] Listening for speech...")
            buffer = []
            last_audio = time.time()
            while self.running:
                try:
                    data = self.q.get(timeout=0.5)
                    buffer.append(data)
                    
                    # Process audio if we have enough silence or buffer is very large
                    if self.silence_duration >= self.min_silence_duration or len(buffer) * self.blocksize / self.samplerate > 10:
                        if len(buffer) > 0:  # Only process if we have data
                            audio = np.concatenate(buffer, axis=0).flatten()
                            buffer = []
                            last_audio = time.time()
                            self.silence_duration = 0
                            
                            # Transcribe
                            result = self.model.transcribe(audio, fp16=torch.cuda.is_available())
                            text = result.get("text", "").strip()
                            
                            if text:
                                # Combine with previous transcriptions
                                combined_text = self.combine_transcriptions(text)
                                if self.is_valid_transcription({"text": combined_text, "avg_logprob": result.get("avg_logprob", 0)}):
                                    print(f"[Whisper] Recognized: {combined_text}")
                                    # Call the command callback if provided
                                    if self.on_command:
                                        self.on_command(combined_text)
                except queue.Empty:
                    continue

    def stop(self):
        self.running = False

if __name__ == "__main__":
    listener = WhisperListener()
    try:
        listener.listen()
    except KeyboardInterrupt:
        print("[Whisper] Stopped.") 