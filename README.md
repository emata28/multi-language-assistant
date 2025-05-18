# Draco Assistant - Multilingual Speech Recognition

This project implements real-time multilingual speech recognition using OpenAI's Whisper model, with support for GPU acceleration.

## Features

- Real-time speech recognition in multiple languages
- Automatic language detection
- GPU acceleration support (NVIDIA GPUs)
- Option to translate any language to English
- High accuracy with various model sizes (tiny to large)
- Continuous speech recognition with silence detection

## Project Structure

```
project/
│
├── src/                    # Source code
│   ├── __init__.py
│   └── speech_recognition.py
│
├── models/                 # Whisper model cache
│
├── tests/                  # Test files
│   └── __init__.py
│
├── requirements.txt        # Project dependencies
│
└── README.md              # Project documentation
```

## Requirements

- Python 3.8 or higher
- PyTorch with CUDA support (for GPU acceleration)
- NVIDIA GPU (optional, but recommended)
- CUDA toolkit and cuDNN (for GPU support)

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

4. (Optional) Verify GPU Support:

```bash
python test_cuda.py
```

## Usage

### Basic Usage

Run the speech recognition with default settings (base model, auto-detect language):

```bash
python src/speech_recognition.py
```

### Model Selection

Choose from different model sizes based on your needs:

- `tiny`: Fastest, lowest accuracy
- `base`: Good balance for most uses
- `small`: Better accuracy, still reasonable speed
- `medium`: High accuracy, recommended for GPU
- `large`: Highest accuracy, requires GPU

```bash
# Use medium model for better accuracy
python src/speech_recognition.py --model medium

# Use tiny model for fastest processing
python src/speech_recognition.py --model tiny
```

### Language Options

1. Auto-detect language (default):

```bash
python src/speech_recognition.py
```

2. Specify language:

```bash
python src/speech_recognition.py --language en  # Force English
python src/speech_recognition.py --language es  # Force Spanish
```

3. Translate to English:

```bash
python src/speech_recognition.py --translate  # Translates any language to English
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

Configure audio settings:

```bash
# Adjust audio settings
python src/speech_recognition.py --samplerate 44100 --blocksize 4000
```

Available options:

- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--device`: Audio input device ID (default: system default)
- `--samplerate`: Audio sample rate in Hz (default: 16000)
- `--blocksize`: Audio block size in samples (default: 8000)
- `--language`: Language code (e.g., 'en', 'es') or None for auto-detection
- `--translate`: Translate speech to English
- `--list-devices`: List available audio input devices and exit

## GPU Support

For optimal performance, the system will automatically use your NVIDIA GPU if available. To ensure GPU support:

1. Install NVIDIA GPU drivers
2. Install CUDA toolkit (compatible with PyTorch)
3. Install cuDNN
4. Verify GPU support:

```bash
python test_cuda.py
```

## Performance Tips

1. Choose the right model size:
   - CPU: Use `tiny` or `base` models
   - GPU: Can use `medium` or `large` models
2. For multilingual use, `medium` model provides the best balance
3. Use `--translate` for real-time translation to English
4. Adjust `blocksize` if you need faster/slower response time
