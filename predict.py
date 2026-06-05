"""Minimal test to verify container starts on Replicate."""
import time
import torch

class Predictor:
    def setup(self):
        print("[setup] ===== MINIMAL TEST PREDICTOR =====")
        print(f"[setup] Python: {__import__('sys').version}")
        print(f"[setup] CWD: {__import__('os').getcwd()}")
        print(f"[setup] Files: {__import__('os').listdir('/src')[:10]}")
        print(f"[setup] PyTorch: {torch.__version__}")
        print(f"[setup] CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
        print("[setup] ===== SETUP COMPLETE =====")
    
    def predict(self, prompt: str = "hello") -> str:
        print(f"[predict] Running with prompt: {prompt}")
        return f"Success! GPU: {torch.cuda.is_available()}, torch: {torch.__version__}"
