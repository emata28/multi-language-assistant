import json
import requests
import os
import yaml

def load_config():
    """Load the configuration from the YAML file."""
    try:
        with open('config/user.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[LLM] Error loading config: {e}")
        return None

def load_prompt(prompt_file):
    """Load a prompt from the config file."""
    try:
        with open(f'config/{prompt_file}', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[LLM] Error loading prompt: {e}")
        return ""

def call_llm(prompt: str, model_config: dict) -> str:
    """Make a call to the LLM with the given prompt and configuration."""
    try:
        request_data = {
            "model": model_config["model"],
            "prompt": prompt,
            "stream": False,
            "num_predict": model_config["num_predict"],
            "temperature": model_config["temperature"],
            "top_p": model_config["top_p"],
            "repeat_penalty": model_config["repeat_penalty"],
            "stop": model_config["stop"]
        }
        
        response = requests.post('http://localhost:11434/api/generate', json=request_data)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    except Exception as e:
        print(f"[LLM] Error calling Ollama API: {e}")
        return ""

def map_to_command(text: str, language: str) -> dict:
    """
    Map user input to a command using two LLM calls:
    1. First LLM classifies the command
    2. Second LLM generates a natural response
    
    Args:
        text: The user's input text
        language: The language of the input ('en' or 'es')
    
    Returns:
        dict: A dictionary containing the command and response text
    """
    config = load_config()
    if not config:
        return {
            "command": "none",
            "text": "Error: Could not load configuration"
        }
    
    # Step 1: Command Classification
    command_prompt = load_prompt('command_prompt.txt')
    if not command_prompt:
        return {
            "command": "none",
            "text": "Error: Could not load command prompt"
        }
    
    command_prompt = command_prompt.replace("{{user_input}}", text)
    command = call_llm(command_prompt, config["llm"]["command_model"]).strip().lower()
    
    # Validate the command
    valid_commands = ['help', 'list', 'status', 'play', 'stop', 'exit', 'none']
    if command not in valid_commands:
        print(f"[LLM] Invalid command: {command}")
        return {
            "command": "none",
            "text": "Lo siento, no entendí ese comando." if language == "es" else "Sorry, I didn't understand that command."
        }
    
    # Step 2: Response Generation
    response_prompt = load_prompt('response_prompt.txt')
    if not response_prompt:
        return {
            "command": command,
            "text": "Error: Could not load response prompt"
        }
    
    response_prompt = response_prompt.replace("{{command}}", command)
    response_prompt = response_prompt.replace("{{user_input}}", text)
    response_prompt = response_prompt.replace("{{language}}", language)
    
    response_text = call_llm(response_prompt, config["llm"]["response_model"]).strip()
    
    return {
        "command": command,
        "text": response_text
    }

if __name__ == "__main__":
    # Test the command mapper
    test_inputs = [
        ("I need help", "en"),
        ("¿me puedes ayudar?", "es"),
        ("What can you do?", "en"),
        ("¿Qué puedes hacer?", "es"),
        ("How are you?", "en"),
        ("¿Cómo estás?", "es"),
        ("Play some music", "en"),
        ("reproduce música", "es"),
        ("Stop that", "en"),
        ("detén eso", "es"),
        ("I want to exit", "en"),
        ("quiero salir", "es")
    ]
    
    for text, lang in test_inputs:
        print(f"\nTesting: {text} ({lang})")
        result = map_to_command(text, lang)
        print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}") 