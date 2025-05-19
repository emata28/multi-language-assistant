import requests

def map_to_command(text):
    prompt = f"""You are a command interpreter. Map the user's input to one of these commands: help, list, status, play, pause, next, previous, turn_on, turn_off, set_temperature, open, close.

Examples:
"ayuda" -> help
"necesito ayuda" -> help
"help me" -> help
"¿Me puedes ayudar?" -> help
"Can you help me?" -> help
"I need help" -> help
"quiero escuchar música" -> play
"muéstrame los comandos" -> list
"what can you do" -> list
"cómo estás" -> status
"system status" -> status

Rules:
1. Return ONLY the command name, nothing else
2. Map any request for help to "help"
3. Map any request to see commands to "list"
4. Map any request about status to "status"
5. If no command matches, return "none"

User input: {text}

Command:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "num_ctx": 512,
                "num_thread": 4,
                "num_gpu": 1,
                "num_predict": 50
            }
        )
        response.raise_for_status()
        result = response.json()
        command = result.get("response", "").strip().lower()
        command = command.strip('"\'.,!?').strip()
        if command in ["help", "list", "status", "play", "pause", "next", "previous", "turn_on", "turn_off", "set_temperature", "open", "close"]:
            return command
        else:
            return "none"
    except Exception as e:
        print(f"Error mapping command: {e}")
        return "none"

if __name__ == "__main__":
    # Test the command mapping
    test_inputs = ["ayuda", "help me", "quiero escuchar música", "muéstrame los comandos", "cómo estás", "invalid command"]
    for text in test_inputs:
        command = map_to_command(text)
        print(f"Input: {text} -> Command: {command}") 