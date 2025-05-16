import os
import sys
import requests
from tqdm import tqdm
import zipfile
import argparse
from pathlib import Path

DEFAULT_MODEL = "vosk-model-en-us-0.22"
DEFAULT_MODEL_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_MODEL}.zip"
MODELS_ROOT_DIR = "models"  # Root directory for all models

def ensure_models_directory() -> Path:
    """
    Ensure the models directory exists and return its path.
    Creates the directory if it doesn't exist.
    
    Returns:
        Path: Path object pointing to the models directory
    
    Raises:
        OSError: If directory creation fails
    """
    models_root = Path(MODELS_ROOT_DIR)
    try:
        models_root.mkdir(exist_ok=True)
        if not models_root.exists():
            raise OSError(f"Failed to create models directory at {models_root}")
        return models_root
    except Exception as e:
        print(f"Error ensuring models directory exists: {e}")
        sys.exit(1)

def download_file(url: str, destination: str) -> None:
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(destination, 'wb') as file, \
         tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

def extract_zip(zip_path: str, extract_path: str) -> None:
    """
    Extract a zip file with progress bar
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.filelist)
        print(f"\nExtracting {total_files} files...")
        
        for i, file in enumerate(zip_ref.filelist, 1):
            zip_ref.extract(file, extract_path)
            progress = (i / total_files) * 100
            print(f"\rProgress: {progress:.1f}%", end="")
    print("\nExtraction complete!")

def get_model_dir(model_name: str) -> Path:
    """
    Get the directory path for a specific model
    """
    models_root = ensure_models_directory()
    return models_root / model_name

def setup_model(model_name: str = DEFAULT_MODEL) -> None:
    """
    Download and setup the Vosk model
    """
    # Ensure models directory exists
    models_root = ensure_models_directory()
    
    # Create temporary directory if needed
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Setup paths
    model_dir = get_model_dir(model_name)
    zip_path = temp_dir / f"{model_name}.zip"
    
    try:
        # Check if model already exists
        if model_dir.exists() and any(model_dir.iterdir()):
            print(f"Model '{model_name}' already exists in {model_dir}")
            return
        
        # Create model directory
        model_dir.mkdir(exist_ok=True)
        
        # Construct the download URL
        url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        
        print(f"Downloading {model_name}...")
        download_file(url, str(zip_path))
        
        print(f"\nExtracting model to {model_dir}...")
        # Extract the contents of the model directory from the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the name of the root directory in the zip
            root_dir = zip_ref.filelist[0].filename.split('/')[0]
            
            # Extract everything
            extract_zip(str(zip_path), str(temp_dir))
            
            # Move contents from the extracted directory to the model directory
            source_dir = temp_dir / root_dir
            
            # Move all contents from source_dir to model_dir
            for item in source_dir.iterdir():
                target = model_dir / item.name
                if target.exists():
                    if target.is_file():
                        target.unlink()
                    else:
                        import shutil
                        shutil.rmtree(target)
                item.rename(target)
            
        print("Cleaning up...")
        # Clean up temporary files
        zip_path.unlink()
        source_dir.rmdir()
        temp_dir.rmdir()
        
        print(f"\nModel setup complete! The model is ready in '{model_dir}'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the model: {e}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Clean up any partial downloads/extractions
        if zip_path.exists():
            zip_path.unlink()
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        sys.exit(1)

def list_installed_models():
    """
    List all installed models in the models directory
    """
    try:
        models_root = ensure_models_directory()
        models = [d for d in models_root.iterdir() if d.is_dir()]
        
        if not models:
            print("No models installed.")
            return
        
        print("\nInstalled models:")
        for model_dir in models:
            print(f"- {model_dir.name}")
        print()
    except Exception as e:
        print(f"Error listing models: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download and setup Vosk speech recognition model")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                      help=f"Model name to download (default: {DEFAULT_MODEL})")
    parser.add_argument("--list", action="store_true",
                      help="List installed models")
    
    args = parser.parse_args()
    
    if args.list:
        list_installed_models()
        return
    
    setup_model(args.model)

if __name__ == "__main__":
    main() 