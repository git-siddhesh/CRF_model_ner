import torch
import platform

def check_pytorch_setup():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"\nAvailable devices:")
    print(f"CPU: {torch.device('cpu')}")
    
    # Check MPS (Metal Performance Shaders) availability
    print("\nMPS (Apple Silicon GPU) Support:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Check CUDA availability (will be False on Mac)
    print("\nCUDA Support:")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Enable async operations
    # torch.backends.mps.enable_async()

if __name__ == "__main__":
    check_pytorch_setup() 