import json
import requests
import os

def load_prompt():
    """Load the prompt from the config file."""
    try:
        with open('config/llm_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[LLM] Error loading prompt: {e}")
        return ""

def map_to_command(text: str, language: str) -> dict:
    """
    Map user input to a command using the LLM.
    
    Args:
        text: The user's input text
        language: The language of the input ('en' or 'es')
    
    Returns:
        dict: A dictionary containing the command, response text, and language
    """
    prompt = load_prompt()
    if not prompt:
        return {
            "command": "none",
            "text": "Error: Could not load prompt configuration",
            "language": language
        }
    
    # Prepare the input in the expected format
    input_data = {
        "text": text,
        "language": language
    }
    
    # Prepare the request to Ollama
    request_data = {
        "model": "tinyllama",
        "prompt": f"{prompt}\n{json.dumps(input_data, ensure_ascii=False)}",
        "stream": False,
        "num_predict": 100,  # Increased for JSON responses
        "temperature": 0.1,  # Low temperature for more consistent outputs
        "top_p": 0.1,  # Low top_p for more focused outputs
        "repeat_penalty": 1.1  # Slight penalty for repetition
    }
    
    try:
        # Send request to Ollama
        response = requests.post('http://localhost:11434/api/generate', json=request_data)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        response_text = result.get('response', '').strip()
        
        # Try to parse the JSON response
        try:
            # Find the first '{' and last '}' to extract just the JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                # Clean up any potential formatting issues
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                parsed = json.loads(json_str)
                
                # Validate the response format
                if not all(key in parsed for key in ['command', 'text', 'language']):
                    print(f"[LLM] Invalid response format: {parsed}")
                    return {
                        "command": "none",
                        "text": "Lo siento, hubo un error procesando tu solicitud." if language == "es" else "Sorry, there was an error processing your request.",
                        "language": language
                    }
                
                # Validate the command
                valid_commands = ['help', 'list', 'status', 'play', 'stop', 'exit', 'none']
                if parsed['command'] not in valid_commands:
                    print(f"[LLM] Invalid command: {parsed['command']}")
                    return {
                        "command": "none",
                        "text": "Lo siento, no entendí ese comando." if language == "es" else "Sorry, I didn't understand that command.",
                        "language": language
                    }
                
                # Validate language matches
                if parsed['language'] != language:
                    print(f"[LLM] Warning: Response language {parsed['language']} doesn't match input language {language}")
                    parsed['language'] = language
                
                return parsed
            else:
                print(f"[LLM] No JSON object found in response: {response_text}")
                return {
                    "command": "none",
                    "text": "Lo siento, hubo un error en la respuesta." if language == "es" else "Sorry, there was an error in the response.",
                    "language": language
                }
        except json.JSONDecodeError as e:
            print(f"[LLM] Invalid JSON response: {response_text}")
            return {
                "command": "none",
                "text": "Lo siento, hubo un error procesando la respuesta." if language == "es" else "Sorry, there was an error processing the response.",
                "language": language
            }
            
    except requests.exceptions.RequestException as e:
        print(f"[LLM] Error calling Ollama API: {e}")
        return {
            "command": "none",
            "text": "Lo siento, no pude conectar con el asistente." if language == "es" else "Sorry, I couldn't connect to the assistant.",
            "language": language
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