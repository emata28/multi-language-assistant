# Python Speech Recognition Project

This is a Python project that implements continuous speech recognition using Vosk.

```
project/
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── speech_recognition.py
│   └── download_model.py
│
├── models/                 # Speech recognition models
│   ├── vosk-model-small-en-us-0.15/    # Default English model
│   └── [other-models]/                 # Additional models
│
├── tests/                  # Test files
│   └── __init__.py
│
├── docs/                   # Documentation
│
├── requirements.txt        # Project dependencies
│
└── README.md              # Project documentation
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:

```bash
.\venv\Scripts\activate
```

- Unix/MacOS:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Manage Vosk models:

```bash
# List installed models
python src/download_model.py --list

# Download specific models
python src/download_model.py --model vosk-model-es-0.42        # Big Spanish model
python src/download_model.py --model vosk-model-small-fr-0.22  # French model
python src/download_model.py --model vosk-model-small-cn-0.22  # Chinese model
```

Note: Models are automatically downloaded when needed. The default English model will be downloaded on first use if not present.

Available models can be found at: https://alphacephei.com/vosk/models

## Usage

### Basic Usage

Run the speech recognition script with default settings (uses English model):

```bash
python src/speech_recognition.py
```

The script will:

1. Check if the requested model exists in the `models` directory
2. Download the model if not found
3. Start the speech recognition system

### Model Management

1. List installed models:

```bash
python src/speech_recognition.py --list-models
```

2. Use a specific model:

```bash
python src/speech_recognition.py --model vosk-model-small-fr-0.22  # Use French model
```

### Audio Device Configuration

1. List available audio devices:

```bash
python src/speech_recognition.py --list-devices
```

2. Use a specific audio device:

```bash
python src/speech_recognition.py --device 1
```

### Advanced Configuration

Configure audio settings and behavior:

```bash
# Adjust audio settings
python src/speech_recognition.py --samplerate 44100 --blocksize 4000

# Disable partial results
python src/speech_recognition.py --no-partial
```

Available options:

- `--model`: Name of the Vosk model to use (default: vosk-model-small-en-us-0.15)
- `--device`: Audio input device ID (default: system default)
- `--samplerate`: Audio sample rate in Hz (default: 16000)
- `--blocksize`: Audio block size in samples (default: 8000)
- `--no-partial`: Disable partial recognition results
- `--list-devices`: List available audio input devices and exit
- `--list-models`: List installed Vosk models and exit

### Customizing Speech Recognition

You can extend the `ContinuousSpeechRecognizer` class to handle detected speech in your own way:

```python
from src.speech_recognition import ContinuousSpeechRecognizer

class MyRecognizer(ContinuousSpeechRecognizer):
    def on_speech_detected(self, text):
        # Handle complete speech detection
        print(f"You said: {text}")

    def on_partial_speech(self, text):
        # Handle partial speech detection
        print(f"Partial text: {text}")
```
