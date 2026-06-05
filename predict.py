"""
LongCat-Video-Avatar-1.5 — Cog Predictor for Replicate

Fully baked image: ALL deps and models in Docker image.
setup() only loads from /opt/models/ — zero runtime downloads.

Single-GPU (A100 80GB) using INT8 quantized DiT + 8-step DMD distillation.
Supports: Audio-Image-to-Video (AI2V), Audio-Text-to-Video (AT2V).

Source: https://github.com/meituan-longcat/LongCat-Video
License: MIT
"""

import os
import sys
import time
import tempfile
import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import numpy as np
import PIL.Image
import imageio
import requests
import librosa
from io import BytesIO

# Enable HF transfer for any incidental downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

print(f"[startup] Python: {sys.version}")
print(f"[startup] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

MODELS_DIR = Path("/opt/models")
AVATAR_DIR = MODELS_DIR / "LongCat-Video-Avatar-1.5"
BASE_DIR = MODELS_DIR / "LongCat-Video"
WHISPER_DIR = MODELS_DIR / "whisper-large-v3"


# ---------------------------------------------------------------------------
# Single-process distributed init (LongCat requires dist even for 1 GPU)
# ---------------------------------------------------------------------------

def _init_distributed():
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
# Cog Predictor
# ---------------------------------------------------------------------------

class Predictor:
    def setup(self):
        """Load all models from local disk. Called once at container start."""
        t_start = time.time()

        # Verify baked models exist
        assert AVATAR_DIR.exists(), f"Avatar model not found at {AVATAR_DIR}"
        assert BASE_DIR.exists(), f"Base model not found at {BASE_DIR}"
        assert WHISPER_DIR.exists(), f"Whisper not found at {WHISPER_DIR}"
        print("[setup] All model directories found ✓")

        # Init distributed (required by LongCat)
        _init_distributed()

        # Import LongCat modules (all deps baked into image)
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        from longcat_video.modules.quantization import load_quantized_dit
        from longcat_video.context_parallel import context_parallel_util
        from transformers import AutoTokenizer, UMT5EncoderModel, WhisperModel, WhisperFeatureExtractor

        # Init context parallel for single GPU
        context_parallel_util.init_context_parallel(
            context_parallel_size=1,
            global_rank=0,
            world_size=1,
        )

        device = torch.device("cuda:0")
        dtype = torch.bfloat16

        # Load tokenizer
        print("[setup] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(BASE_DIR), subfolder="tokenizer", torch_dtype=dtype
        )

        # Load UMT5 text encoder (~10GB)
        print("[setup] Loading UMT5 text encoder...")
        t0 = time.time()
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(BASE_DIR), subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        print(f"[setup] Text encoder loaded ({time.time()-t0:.1f}s)")

        # Load VAE
        print("[setup] Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            str(BASE_DIR), subfolder="vae", torch_dtype=dtype
        ).to(device)

        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(AVATAR_DIR), subfolder="scheduler", torch_dtype=dtype
        )

        # Load INT8 quantized DiT (~8GB)
        print("[setup] Loading INT8 DiT...")
        t0 = time.time()
        cp_split_hw = context_parallel_util.get_optimal_split(1)
        dit = load_quantized_dit(
            str(AVATAR_DIR),
            subfolder="base_model_int8",
            cp_split_hw=cp_split_hw,
        ).to(device)
        print(f"[setup] DiT loaded ({time.time()-t0:.1f}s)")

        # Build pipeline
        print("[setup] Building pipeline...")
        self.pipeline = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=dit,
            scheduler=scheduler,
        )

        # Load DMD distillation LoRA and fuse
        print("[setup] Loading DMD LoRA...")
        lora_path = str(Path(AVATAR_DIR) / "lora")
        self.pipeline.load_lora_weights(lora_path, weight_name="dmd_lora.safetensors")
        self.pipeline.fuse_lora()
        print("[setup] LoRA fused ✓")

        self.pipeline = self.pipeline.to(device)
        self.device = device
        self.dtype = dtype

        # Load Whisper-large-v3 (for audio processing)
        print("[setup] Loading Whisper-large-v3...")
        t0 = time.time()
        self.audio_encoder = WhisperModel.from_pretrained(
            str(WHISPER_DIR), torch_dtype=dtype
        ).to(device)
        self.audio_feature_extractor = WhisperFeatureExtractor.from_pretrained(str(WHISPER_DIR))
        print(f"[setup] Whisper loaded ({time.time()-t0:.1f}s)")

        print(f"[setup] Complete in {time.time()-t_start:.1f}s ✓")

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
            image: Reference face image URL or path (for AI2V mode)
            audio: Audio file URL for lip sync (enables audio-driven generation)
            resolution: Output resolution — "480p" or "720p"
            num_frames: Number of frames (max 81 for 480p, 49 for 720p)
            text_guidance_scale: Text CFG scale (1.0-10.0, default 4.0)
            audio_guidance_scale: Audio CFG scale (1.0-10.0, default 4.0)
            num_inference_steps: Denoising steps (default 8 for DMD distillation)
            seed: Random seed (-1 for random)
            fps: Output video FPS (default 16)

        Returns:
            Path to generated MP4 video
        """
        # Set seed
        if seed >= 0:
            torch.manual_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Resolution mapping
        res_map = {"480p": (480, 854), "720p": (720, 1280)}
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

        with torch.no_grad():
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

    def _load_image(self, image_url: str):
        """Load and preprocess a reference image."""
        if image_url.startswith(("http://", "https://")):
            resp = requests.get(image_url, timeout=30)
            img = PIL.Image.open(BytesIO(resp.content))
        else:
            img = PIL.Image.open(image_url)
        return img.convert("RGB")

    def _process_audio(self, audio_url: str):
        """Process audio file into embeddings using Whisper."""
        if audio_url.startswith(("http://", "https://")):
            resp = requests.get(audio_url, timeout=60)
            audio_bytes = BytesIO(resp.content)
        else:
            audio_bytes = open(audio_url, "rb")

        try:
            waveform, sr = librosa.load(audio_bytes, sr=16000, mono=True)
            waveform = torch.tensor(waveform).unsqueeze(0)

            inputs = self.audio_feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )

            with torch.no_grad():
                encoder_output = self.audio_encoder.encoder(
                    inputs.input_features.to(self.device, dtype=self.dtype)
                )
                audio_emb = encoder_output.last_hidden_state

            return audio_emb
        finally:
            if hasattr(audio_bytes, 'close'):
                audio_bytes.close()

    def _save_video(self, video, fps: int) -> str:
        """Save video tensor to MP4 file."""
        output_path = tempfile.mktemp(suffix=".mp4")

        # [C, T, H, W] → [T, H, W, C]
        frames = (video.permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        return output_path
