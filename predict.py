"""
LongCat-Video-Avatar-1.5 — Cog Predictor for Replicate

Single-GPU deployment (A100 80GB) using INT8 quantized DiT + 8-step DMD distillation.
Supports: Audio-Image-to-Video (AI2V), Audio-Text-to-Video (AT2V), Video Continuation.

Adapted from: https://github.com/meituan-longcat/LongCat-Video
"""

import os
import sys
import json
import time
import math
import datetime
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import PIL.Image
import librosa

# Enable fast parallel downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Print startup diagnostics
print(f"[startup] Python: {sys.version}")
print(f"[startup] CWD: {os.getcwd()}")
print(f"[startup] Files: {os.listdir('/src')}")

# ---------------------------------------------------------------------------
# Single-process distributed init shim
# ---------------------------------------------------------------------------

def _init_single_process_distributed():
    """Initialize a single-process distributed environment for Cog."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=3600),
            rank=0,
            world_size=1,
        )
        print("[init] Single-process distributed initialized")


# ---------------------------------------------------------------------------
# Model download with caching
# ---------------------------------------------------------------------------

def _download_avatar_model():
    """Download LongCat-Video-Avatar-1.5 (INT8 + LoRA + configs only)."""
    from huggingface_hub import snapshot_download
    
    return snapshot_download(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        allow_patterns=[
            "base_model_int8/*",
            "lora/*",
            "scheduler/*",
            "config.json",
            "model_index.json",
        ],
    )


def _download_base_model():
    """Download LongCat-Video base (text_encoder, vae, tokenizer)."""
    from huggingface_hub import snapshot_download
    
    return snapshot_download(
        "meituan-longcat/LongCat-Video",
        allow_patterns=[
            "text_encoder/*",
            "vae/*",
            "tokenizer/*",
            "scheduler/*",
        ],
    )


def _download_whisper():
    """Download Whisper-large-v3 (only when audio input is provided)."""
    from huggingface_hub import snapshot_download
    
    return snapshot_download(
        "openai/whisper-large-v3",
        allow_patterns=["model.safetensors", "config.json", "*.json", "merges.txt", "vocab.json"],
    )


# ---------------------------------------------------------------------------
# Cog Predictor
# ---------------------------------------------------------------------------

class Predictor:
    def setup(self):
        """Load models into GPU memory. Called once at container start."""
        from transformers import AutoTokenizer, UMT5EncoderModel
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        from longcat_video.modules.quantization import load_quantized_dit
        from longcat_video.context_parallel import context_parallel_util
        
        t_start = time.time()
        
        # Initialize single-process distributed
        _init_single_process_distributed()
        
        # Initialize context parallel for single GPU
        context_parallel_util.init_context_parallel(
            context_parallel_size=1,
            global_rank=0,
            world_size=1,
        )
        
        # Download models (uses HF cache — fast on subsequent runs)
        print("[setup] Downloading/loading avatar model...")
        t0 = time.time()
        avatar_dir = _download_avatar_model()
        print(f"[setup] Avatar model ready ({time.time()-t0:.1f}s)")
        
        print("[setup] Downloading/loading base model...")
        t0 = time.time()
        base_dir = _download_base_model()
        print(f"[setup] Base model ready ({time.time()-t0:.1f}s)")
        
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        
        print("[setup] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_dir), subfolder="tokenizer", torch_dtype=dtype
        )
        
        print("[setup] Loading UMT5 text encoder...")
        t0 = time.time()
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        print(f"[setup] Text encoder loaded ({time.time()-t0:.1f}s)")
        
        print("[setup] Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            str(base_dir), subfolder="vae", torch_dtype=dtype
        ).to(device)
        
        print("[setup] Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(avatar_dir), subfolder="scheduler", torch_dtype=dtype
        )
        
        print("[setup] Loading INT8 DiT model...")
        t0 = time.time()
        cp_split_hw = context_parallel_util.get_optimal_split(1)
        dit = load_quantized_dit(
            str(avatar_dir),
            subfolder="base_model_int8",
            cp_split_hw=cp_split_hw,
        ).to(device)
        print(f"[setup] DiT loaded ({time.time()-t0:.1f}s)")
        
        print("[setup] Loading DMD distillation LoRA...")
        distill_path = str(Path(avatar_dir) / "lora" / "dmd_lora.safetensors")
        
        print("[setup] Building pipeline...")
        self.pipeline = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=dit,
            scheduler=scheduler,
        )
        
        # Load LoRA weights
        self.pipeline.load_lora_weights(str(Path(avatar_dir) / "lora"), weight_name="dmd_lora.safetensors")
        self.pipeline.fuse_lora()
        print("[setup] LoRA fused")
        
        self.pipeline = self.pipeline.to(device)
        self.device = device
        self.dtype = dtype
        self.avatar_dir = avatar_dir
        
        # Whisper loaded lazily when audio input is provided
        self._whisper_loaded = False
        self._audio_encoder = None
        self._audio_feature_extractor = None
        
        print(f"[setup] Complete in {time.time()-t_start:.1f}s")
    
    def _ensure_whisper(self):
        """Load Whisper model lazily (only when audio input is provided)."""
        if self._whisper_loaded:
            return
        
        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        
        print("[setup] Loading Whisper-large-v3 (lazy, audio input detected)...")
        t0 = time.time()
        whisper_dir = _download_whisper()
        self._audio_encoder = get_audio_encoder(whisper_dir, device=self.device)
        self._audio_feature_extractor = get_audio_feature_extractor(whisper_dir)
        self._whisper_loaded = True
        print(f"[setup] Whisper loaded ({time.time()-t0:.1f}s)")
    
    def predict(
        self,
        prompt: str = "A person is speaking naturally with expressive facial movements",
        image: Optional[str] = None,
        audio: Optional[str] = None,
        resolution: str = "480p",
        num_frames: int = 49,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_inference_steps: int = 8,
        seed: int = -1,
        fps: int = 16,
    ) -> str:
        """Generate a talking avatar video.
        
        Args:
            prompt: Text prompt describing the video content
            image: Reference face image URL (for AI2V mode)
            audio: Audio file URL for lip sync (enables audio-driven generation)
            resolution: Output resolution - "480p" or "720p"
            num_frames: Number of frames to generate (max 81 for 480p)
            text_guidance_scale: CFG scale for text conditioning (1.0-10.0)
            audio_guidance_scale: CFG scale for audio conditioning (1.0-10.0)
            num_inference_steps: Number of denoising steps (default: 8 for DMD)
            seed: Random seed (-1 for random)
            fps: Output video FPS
        
        Returns:
            URL to the generated video
        """
        # Lazy load whisper if audio is provided
        if audio:
            self._ensure_whisper()
        
        # Set seed
        if seed >= 0:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Resolution mapping
        res_map = {
            "480p": (480, 854),
            "720p": (720, 1280),
            "1080p": (1080, 1920),
        }
        height, width = res_map.get(resolution, (480, 854))
        
        # Load reference image
        ref_image = None
        if image:
            ref_image = self._load_image(image)
        
        # Process audio
        audio_emb = None
        if audio:
            audio_emb = self._process_audio(audio)
        
        print(f"[predict] Generating {num_frames} frames at {width}x{height}...")
        t0 = time.time()
        
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                image=ref_image,
                audio_emb=audio_emb,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                generator=generator,
            )
        
        video = result.videos[0]  # [C, T, H, W]
        print(f"[predict] Generated in {time.time()-t0:.1f}s")
        
        # Save video
        output_path = self._save_video(video, fps)
        return output_path
    
    def _load_image(self, image_url: str) -> PIL.Image.Image:
        """Load and preprocess a reference image."""
        import requests
        from io import BytesIO
        
        if image_url.startswith(("http://", "https://")):
            resp = requests.get(image_url, timeout=30)
            img = PIL.Image.open(BytesIO(resp.content))
        else:
            img = PIL.Image.open(image_url)
        
        return img.convert("RGB")
    
    def _process_audio(self, audio_url: str) -> torch.Tensor:
        """Process audio file into embeddings."""
        import requests
        from io import BytesIO
        
        # Download audio
        if audio_url.startswith(("http://", "https://")):
            resp = requests.get(audio_url, timeout=60)
            audio_bytes = BytesIO(resp.content)
        else:
            audio_bytes = open(audio_url, "rb")
        
        # Load and resample audio
        waveform, sr = librosa.load(audio_bytes, sr=16000, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
        
        # Extract audio features using Whisper
        inputs = self._audio_feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        audio_emb = self._audio_encoder(
            inputs.input_features.to(self.device, dtype=self.dtype)
        )
        
        return audio_emb
    
    def _save_video(self, video: torch.Tensor, fps: int) -> str:
        """Save video tensor to file."""
        import imageio
        
        output_path = tempfile.mktemp(suffix=".mp4")
        
        # [C, T, H, W] → [T, H, W, C]
        frames = (video.permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        return output_path
