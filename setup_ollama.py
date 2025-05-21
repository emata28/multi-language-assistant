#!/usr/bin/env python3
"""
Setup script for Ollama Docker container and models.
"""

import os
import sys
import time
import yaml
import subprocess
import requests
import json
from typing import Optional, List, Dict

def load_config(config_path: str = "config/user.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)

def check_docker() -> bool:
    """Check if Docker is installed and running."""
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        print("Docker is not installed or not running. Please install/start Docker.")
        sys.exit(1)

def wait_for_ollama(timeout: int = 60) -> bool:
    """Wait for Ollama API to be ready."""
    print("Waiting for Ollama to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                print("Ollama is ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print("\nOllama did not start in time.")
    return False

def start_ollama() -> bool:
    """Start Ollama Docker container."""
    print("Starting Ollama Docker container...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        return wait_for_ollama()
    except subprocess.CalledProcessError as e:
        print(f"Error starting Ollama container: {e}")
        return False

def get_installed_models() -> List[str]:
    """Get list of installed models."""
    try:
        r = requests.get("http://localhost:11434/api/tags")
        if r.status_code == 200:
            return [model["name"] for model in r.json()["models"]]
        return []
    except Exception:
        return []

def pull_model(model: str) -> bool:
    """Pull a specific model with progress tracking."""
    print(f"\nPulling model: {model}")
    try:
        r = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model},
            stream=True
        )
        
        if r.status_code != 200:
            print(f"Failed to pull model {model}: {r.text}")
            return False
            
        # Track progress
        for line in r.iter_lines():
            if line:
                try:
                    status = json.loads(line)
                    if "status" in status:
                        print(f"\r{status['status']}", end="", flush=True)
                    if "completed" in status and status["completed"]:
                        print("\nModel pulled successfully!")
                        return True
                except json.JSONDecodeError:
                    continue
                    
        return True
    except Exception as e:
        print(f"\nError pulling model {model}: {e}")
        return False

def get_required_models(config: Dict) -> List[str]:
    """Get list of required models from config."""
    models = set()
    if "llm" in config:
        if "command_model" in config["llm"]:
            models.add(config["llm"]["command_model"]["model"])
        if "response_model" in config["llm"]:
            models.add(config["llm"]["response_model"]["model"])
    return list(models)

def main():
    """Main setup function."""
    print("Setting up Ollama...")
    
    # Load configuration
    config = load_config()
    required_models = get_required_models(config)
    
    if not required_models:
        print("Error: No models specified in configuration.")
        sys.exit(1)
    
    # Check if Docker is running
    if not check_docker():
        print("Error: Docker is not running. Please start Docker Desktop first.")
        sys.exit(1)
    
    # Start container
    if not start_ollama():
        print("Error: Failed to start Ollama container.")
        sys.exit(1)
    
    # Get installed models
    installed_models = get_installed_models()
    
    # Pull missing models
    for model in required_models:
        if model not in installed_models:
            if not pull_model(model):
                print(f"Error: Failed to pull model {model}")
                sys.exit(1)
        else:
            print(f"Model {model} is already installed.")
    
    print("\nSetup completed successfully!")
    print("You can now run the assistant with: python -m src.cli")

if __name__ == "__main__":
    main() 