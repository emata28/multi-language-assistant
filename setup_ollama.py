#!/usr/bin/env python3
"""
Setup script for Ollama Docker container and model.
"""

import os
import sys
import time
import yaml
import subprocess
import requests
from typing import Optional

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

def check_docker():
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        print("Docker is not installed or not running. Please install/start Docker.")
        sys.exit(1)

def start_ollama():
    print("Starting Ollama Docker container...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    # Wait for Ollama API to be ready
    for _ in range(30):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                print("Ollama is ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    print("Ollama did not start in time.")
    sys.exit(1)

def pull_model(model="tinyllama"):
    print(f"Pulling LLM model: {model}")
    r = requests.post("http://localhost:11434/api/pull", json={"name": model})
    if r.status_code == 200:
        print(f"Model {model} pulled successfully.")
    else:
        print(f"Failed to pull model {model}: {r.text}")
        sys.exit(1)

def main():
    """Main setup function."""
    print("Setting up Ollama...")
    
    # Load configuration
    config = load_config()
    model = config.get('llm', {}).get('model', 'phi')
    
    # Check if Docker is running
    if not check_docker():
        print("Error: Docker is not running. Please start Docker Desktop first.")
        sys.exit(1)
    
    # Start container
    print("Starting Ollama container...")
    start_ollama()
    
    # Pull model
    print(f"Pulling {model} model...")
    pull_model(model)
    
    print("\nSetup completed successfully!")
    print("You can now run the assistant with: python -m src.cli")

if __name__ == "__main__":
    main() 