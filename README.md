# Draco Assistant - Multilingual Speech Recognition

This project implements real-time multilingual speech recognition using OpenAI's Whisper model, with support for GPU acceleration and automatic hardware optimization.

## Features

- Real-time speech recognition in multiple languages
- Automatic language detection
- GPU acceleration support (NVIDIA GPUs)
- Option to translate any language to English
- High accuracy with various model sizes (tiny to large)
- Continuous speech recognition with silence detection
- Automatic hardware optimization
- Batch processing for improved performance
- FP16 (half-precision) support for faster inference
- Memory-aware settings adjustment

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

Run the speech recognition with automatic hardware optimization:

```bash
python src/speech_recognition.py
```

The system will automatically:

- Detect your GPU and available memory
- Choose optimal batch size
- Enable/disable FP16 based on hardware capabilities
- Adjust block size for best performance
- Select appropriate processing settings

### Model Selection

Choose from different model sizes based on your needs:

- `tiny`: Fastest, lowest accuracy (1GB GPU memory)
- `base`: Good balance for most uses (1.5GB GPU memory)
- `small`: Better accuracy, still reasonable speed (2GB GPU memory)
- `medium`: High accuracy, recommended for GPU (5GB GPU memory)
- `large`: Highest accuracy, requires GPU (10GB GPU memory)

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

The system automatically optimizes settings, but you can override them:

```bash
# Override automatic batch size
python src/speech_recognition.py --batch-size 2

# Disable FP16 optimization
python src/speech_recognition.py --no-fp16

# Override automatic block size
python src/speech_recognition.py --blocksize 4000
```

Available options:

- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--device`: Audio input device ID (default: system default)
- `--samplerate`: Audio sample rate in Hz (default: 16000)
- `--blocksize`: Audio block size in samples (default: automatic)
- `--language`: Language code (e.g., 'en', 'es') or None for auto-detection
- `--translate`: Translate speech to English
- `--list-devices`: List available audio input devices and exit
- `--batch-size`: Override automatic batch size
- `--no-fp16`: Disable FP16 optimization

## Automatic Hardware Optimization

The system automatically optimizes settings based on your hardware:

### GPU Memory-based Optimizations

- Large model (10GB+ GPU):
  - Batch size: 1-2
  - FP16: Enabled
  - Block size: 8000
- Medium model (5GB+ GPU):
  - Batch size: 1-2
  - FP16: Enabled
  - Block size: 8000
- Small/Base model (2GB+ GPU):
  - Batch size: 2-4
  - FP16: Enabled
  - Block size: 8000
- Tiny model (1GB+ GPU):
  - Batch size: 4
  - FP16: Enabled
  - Block size: 8000

### CPU Optimizations

- All models:
  - Batch size: 1
  - FP16: Disabled
  - Block size: 4000

## Performance Tips

1. Let the system automatically optimize for your hardware
2. For best results:
   - GPU: Use `medium` or `large` models
   - CPU: Use `tiny` or `base` models
3. For multilingual use, `medium` model provides the best balance
4. Use `--translate` for real-time translation to English
5. Monitor system performance and adjust settings if needed

## Troubleshooting

1. If you experience memory issues:

   - Try a smaller model
   - Use `--no-fp16` to disable half-precision
   - Reduce batch size with `--batch-size 1`

2. If you experience slow performance:

   - Check if GPU is being used
   - Try a smaller model
   - Adjust block size with `--blocksize`

3. If you experience audio issues:
   - List available devices with `--list-devices`
   - Try a different audio device with `--device`
   - Adjust sample rate with `--samplerate`
