#!/usr/bin/env python3
"""Pre-download all LongCat model weights during Docker build."""
import os
from pathlib import Path

MODELS_DIR = Path("/opt/models/longcat")

def main():
    from huggingface_hub import snapshot_download

    avatar_dir = MODELS_DIR / "LongCat-Video-Avatar-1.5"
    base_dir = MODELS_DIR / "LongCat-Video"

    # Download Avatar 1.5 (INT8 DiT, LoRA, Whisper, configs)
    if not (avatar_dir / "base_model_int8").exists():
        print("[build] Downloading LongCat-Video-Avatar-1.5 (INT8 + LoRA + Whisper)...")
        snapshot_download(
            "meituan-longcat/LongCat-Video-Avatar-1.5",
            local_dir=str(avatar_dir),
            # Skip vocal separator (deferred)
            ignore_patterns=["vocal_separator/*"],
        )
    else:
        print("[build] Avatar 1.5 already cached")

    # Download base model components (text_encoder, vae, tokenizer)
    if not (base_dir / "text_encoder").exists():
        print("[build] Downloading LongCat-Video base (text_encoder, vae, tokenizer)...")
        snapshot_download(
            "meituan-longcat/LongCat-Video",
            local_dir=str(base_dir),
            allow_patterns=[
                "text_encoder/*",
                "vae/*",
                "tokenizer/*",
                "scheduler/*",
                "config.json",
                "model_index.json",
            ],
        )
    else:
        print("[build] Base model already cached")

    print("[build] All models downloaded!")

if __name__ == "__main__":
    main()
