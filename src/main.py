from src.whisper_listener import WhisperListener
from src.llm_command import map_to_command
import requests

def get_ollama_model():
    """Get the current model being used by Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                return models[0].get("name", "unknown")
    except:
        pass
    return "tinyllama"  # Default model

def process_command(text):
    """Process a recognized command through the LLM."""
    print(f"[Main] Sending to LLM: {text}")
    command = map_to_command(text)
    if command != "none":
        print(f"[Main] Command recognized: {command}")
    else:
        print("[Main] No command recognized")

def main():
    # Get and display model information
    llm_model = get_ollama_model()
    print(f"[LLM] Using model: {llm_model}")
    
    # Create listener with command processing callback
    listener = WhisperListener(on_command=process_command)
    try:
        print("[Main] Starting voice command system...")
        print("[Main] Listening for commands...")
        listener.listen()
    except KeyboardInterrupt:
        print("[Main] Stopping...")
    finally:
        listener.stop()

if __name__ == "__main__":
    main() 