import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Try to perform a simple operation on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            print("\nGPU computation test successful!")
            print(f"Test tensor: {y}")
        except Exception as e:
            print(f"\nGPU computation test failed: {str(e)}")
    else:
        print("\nCUDA is not available. Please check:")
        print("1. NVIDIA GPU is present")
        print("2. NVIDIA drivers are installed")
        print("3. CUDA toolkit is installed")
        print("4. PyTorch CUDA version matches your CUDA installation")

if __name__ == "__main__":
    check_cuda() 