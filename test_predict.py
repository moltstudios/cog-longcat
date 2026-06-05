"""
Minimal test predictor to validate container starts on Replicate.
"""
import time

class Predictor:
    def setup(self):
        print("[setup] Hello from LongCat test predictor!")
        print("[setup] Setup complete")
    
    def predict(self, prompt: str = "test") -> str:
        print(f"[predict] Prompt: {prompt}")
        return f"Echo: {prompt}"
