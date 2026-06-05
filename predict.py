"""Minimal test to verify container starts."""
import torch

class Predictor:
    def setup(self):
        print("[setup] ===== MINIMAL TEST ON longcat-avatar-v2 =====")
        print(f"[setup] PyTorch: {torch.__version__}")
        print(f"[setup] CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
        print("[setup] DONE")
    
    def predict(self, prompt: str = "hello") -> str:
        return f"Success! GPU: {torch.cuda.is_available()}"
