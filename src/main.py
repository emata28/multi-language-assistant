import os
import sys
import requests
import time
from src.whisper_listener import WhisperListener
from src.llm_command import map_to_command

def get_ollama_model():
    """Get the current Ollama model being used."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                return models[0].get("name", "tinyllama")
    except Exception as e:
        print(f"[Main] Error getting Ollama model: {e}")
    return "tinyllama"

def process_command(text, language):
    """Process a recognized command."""
    print(f"[Main] Sending to LLM: {text}")
    result = map_to_command(text, language)
    print(f"[Draco] {result['text']}")
    return result['command']

def main():
    """Main function to run the voice command system."""
    # Get and display model information
    whisper_model = "whisper-1"
    llm_model = get_ollama_model()
    print(f"[Main] Using Whisper model: {whisper_model}")
    print(f"[Main] Using LLM model: {llm_model}")
    
    # Start the voice command system
    listener = WhisperListener()
    try:
        print("[Main] Starting voice command system...")
        print("[Main] Listening for commands...")
        
        # Start listening in a separate thread
        import threading
        listen_thread = threading.Thread(target=listener.listen)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Main loop to process transcriptions
        while True:
            try:
                text, language = listener.get_transcription()
                if text:
                    print(f"[Main] Got transcription: {text} ({language})")
                    # Process the command with the detected language
                    command = process_command(text, language)
                    if command == "exit":
                        break
                time.sleep(0.1)  # Small sleep to prevent CPU spinning
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Main] Error in main loop: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        print("[Main] Stopping listener...")
        listener.stop()
        print("[Main] Stopped.")

if __name__ == "__main__":
    main() 