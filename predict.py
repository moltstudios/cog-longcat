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

# ---------------------------------------------------------------------------
# Single-process distributed init shim
# ---------------------------------------------------------------------------

def _init_single_process_distributed():
    """Initialize a single-process distributed environment for Cog.
    
    The LongCat pipeline code expects torch.distributed to be initialized
    (it reads RANK, calls dist.init_process_group, etc.).
    For single-GPU Cog inference, we mock a 1-process distributed group.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",  # gloo works on single process; NCCL needs >=2
            timeout=datetime.timedelta(seconds=3600),
            rank=0,
            world_size=1,
        )
        print("[init] Single-process distributed initialized (gloo, rank=0, world_size=1)")


# ---------------------------------------------------------------------------
# Model downloads via pget (fast for Replicate)
# ---------------------------------------------------------------------------

MODELS_DIR = Path(os.environ.get("LONGCAT_WEIGHTS_DIR", "/opt/models/longcat"))

def _download_models():
    """Verify model files exist (pre-downloaded during Docker build)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    avatar_dir = MODELS_DIR / "LongCat-Video-Avatar-1.5"
    base_dir = MODELS_DIR / "LongCat-Video"
    
    # Check if models exist from build-time download
    if (avatar_dir / "base_model_int8").exists():
        print("[setup] Models found from build-time download")
        return avatar_dir, base_dir
    
    # Fallback: download at runtime if not baked in
    print("[setup] Models not found, downloading at runtime (slow)...")
    t0 = time.time()
    _download_with_hf_hub(avatar_dir, base_dir)
    elapsed = time.time() - t0
    print(f"[setup] Download complete in {elapsed:.0f}s")
    return avatar_dir, base_dir


def _download_with_hf_hub(avatar_dir, base_dir):
    """Fallback: download using huggingface_hub."""
    from huggingface_hub import snapshot_download
    
    print("[download] Downloading LongCat-Video-Avatar-1.5...")
    snapshot_download(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        local_dir=str(avatar_dir),
        # Skip the full bf16 base model — we only need INT8
        ignore_patterns=["base_model/diffusion_pytorch_model*", "assets/*", "*.mp4"],
    )
    
    print("[download] Downloading LongCat-Video base (text_encoder, vae, tokenizer)...")
    snapshot_download(
        "meituan-longcat/LongCat-Video",
        local_dir=str(base_dir),
        # Only download what we need
        allow_patterns=["text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*"],
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
        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        from longcat_video.context_parallel import context_parallel_util
        # audio_separator removed — vocal separation handled via ffmpeg fallback
        
        # Initialize single-process distributed
        _init_single_process_distributed()
        
        # Initialize context parallel for single GPU
        context_parallel_util.init_context_parallel(
            context_parallel_size=1,
            global_rank=0,
            world_size=1,
        )
        
        # Download models if needed
        avatar_dir, base_dir = _download_models()
        
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        
        print("[setup] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_dir), subfolder="tokenizer", torch_dtype=dtype
        )
        
        print("[setup] Loading UMT5 text encoder (~21GB)...")
        t0 = time.time()
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype
        )
        print(f"[setup] Text encoder loaded in {time.time()-t0:.1f}s")
        
        print("[setup] Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            str(base_dir), subfolder="vae", torch_dtype=dtype
        )
        
        print("[setup] Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(avatar_dir), subfolder="scheduler", torch_dtype=dtype
        )
        
        print("[setup] Loading INT8 DiT model (~15GB)...")
        t0 = time.time()
        cp_split_hw = context_parallel_util.get_optimal_split(1)  # [1, 1] for single GPU
        dit = load_quantized_dit(
            str(avatar_dir),
            subfolder="base_model_int8",
            cp_split_hw=cp_split_hw,
        )
        print(f"[setup] DiT loaded in {time.time()-t0:.1f}s")
        
        print("[setup] Loading DMD distillation LoRA...")
        distill_path = str(avatar_dir / "lora" / "dmd_lora.safetensors")
        if os.path.exists(distill_path):
            dit.load_lora(distill_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
            dit.enable_loras(["dmd"])
            print("[setup] DMD LoRA loaded")
        else:
            print(f"[setup] WARNING: DMD LoRA not found at {distill_path}")
        
        print("[setup] Loading Whisper-Large-v3 audio encoder...")
        whisper_path = str(avatar_dir / "whisper-large-v3")
        audio_encoder = get_audio_encoder(whisper_path, model_type="avatar-v1.5").to(device)
        audio_feature_extractor = get_audio_feature_extractor(whisper_path, model_type="avatar-v1.5")
        
        # Vocal separator deferred — not needed for initial testing
        # Will add back via lighter-weight method after image builds
        print("[setup] Skipping vocal separator (using raw audio)")
        
        print("[setup] Initializing pipeline...")
        self.pipe = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
            audio_encoder=audio_encoder,
            audio_feature_extractor=audio_feature_extractor,
            model_type="avatar-v1.5",
        )
        self.pipe.to(device)
        
        self.vocal_separator = None  # Deferred
        self.device = device
        
        # Free CPU memory
        torch.cuda.empty_cache()
        
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[setup] GPU after loading: {gpu_alloc:.1f}GB alloc, {gpu_reserved:.1f}GB reserved")
        print("[setup] ✅ Ready")
    
    def predict(
        self,
        image: str = "",
        audio: str = "",
        prompt: str = "A person is speaking naturally with expressive facial movements.",
        negative_prompt: str = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        resolution: str = "480p",
        num_segments: int = 1,
        num_frames: int = 93,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        seed: int = 42,
        ref_img_index: int = 10,
        mask_frame_range: int = 3,
    ) -> str:
        """
        Generate a talking avatar video from an image or text prompt + audio.
        
        Args:
            image: URL or path to reference image (for AI2V mode). Leave empty for AT2V mode.
            audio: URL or path to audio file (speech).
            prompt: Text prompt describing the scene and character.
            negative_prompt: Negative text prompt.
            resolution: "480p" (480x832) or "720p" (768x1280).
            num_segments: Number of video segments (for long video via continuation).
            num_frames: Frames per segment (default 93, must satisfy (n-1)%4==0).
            text_guidance_scale: Text CFG scale (1-10, default 4.0).
            audio_guidance_scale: Audio CFG scale (1-10, default 4.0). Higher = stronger lip sync.
            seed: Random seed for reproducibility.
            ref_img_index: Reference frame index for video continuation (0-24 for consistency, 30 to reduce repetition).
            mask_frame_range: Mask frame range for video continuation (higher = less repetition but possible artifacts).
        
        Returns:
            Path to output MP4 video file.
        """
        from diffusers.utils import load_image
        from longcat_video.audio_process.torch_utils import save_video_ffmpeg
        from longcat_video.context_parallel import context_parallel_util
        
        assert audio, "Audio input is required"
        
        print(f"[predict] Starting: mode={'AI2V' if image else 'AT2V'}, resolution={resolution}, frames={num_frames}, segments={num_segments}")
        t_start = time.time()
        
        device = self.device
        use_distill = True  # Required for v1.5
        model_type = "avatar-v1.5"
        save_fps = 25  # v1.5 default
        audio_stride = 1  # v1.5 default
        num_cond_frames = 13
        
        # Setup generator
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        # ---- Audio processing ----
        print("[predict] Processing audio...")
        audio_path = audio
        if audio.startswith("http"):
            print(f"[predict] Downloading audio from {audio[:80]}...")
            import urllib.request
            audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            urllib.request.urlretrieve(audio, audio_tmp.name)
            audio_path = audio_tmp.name
        
        # Vocal extraction — skip for now, use raw audio directly
        # TODO: Add back vocal separation via lighter-weight method
        print("[predict] Using raw audio (vocal separation deferred)")
        vocal_path = audio_path
        
        # Load and pad audio
        speech_array, sr = librosa.load(vocal_path, sr=16000)
        source_duration = len(speech_array) / sr
        
        generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
        added_samples = math.ceil((generate_duration - source_duration) * sr)
        if added_samples > 0:
            speech_array = np.append(speech_array, [0.] * added_samples)
        
        # Audio embedding
        print(f"[predict] Computing Whisper audio embeddings ({len(speech_array)/sr:.1f}s audio)...")
        full_audio_emb = self.pipe.get_audio_embedding(
            speech_array, fps=save_fps * audio_stride, device=device, sample_rate=sr, model_type=model_type
        )
        if torch.isnan(full_audio_emb).any():
            raise ValueError("Audio embedding contains NaN values")
        
        # Prepare first segment audio
        indices = torch.arange(2 * 2 + 1) - 2
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(device)
        
        # ---- Stage 1: Generate first segment ----
        if image:
            # AI2V mode: Image + Audio → Video
            print(f"[predict] AI2V: Generating from image + audio...")
            img = load_image(image)
            output_tuple = self.pipe.generate_ai2v(
                image=img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                num_frames=num_frames,
                num_inference_steps=8,  # Distilled
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                output_type="both",
                generator=generator,
                audio_emb=audio_emb,
                use_distill=use_distill,
            )
        else:
            # AT2V mode: Text + Audio → Video
            print(f"[predict] AT2V: Generating from text + audio...")
            output_tuple = self.pipe.generate_at2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=480 if resolution == "480p" else 768,
                width=832 if resolution == "480p" else 1280,
                num_frames=num_frames,
                num_inference_steps=8,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                generator=generator,
                output_type="both",
                audio_emb=audio_emb,
                use_distill=use_distill,
            )
        
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img_frame) for img_frame in video]
        del output
        torch.cuda.empty_cache()
        
        all_generated_frames = list(video)
        
        # ---- Video continuation (multi-segment) ----
        if num_segments > 1:
            width, height = video[0].size
            current_video = video
            ref_latent = latent[:, :, :1].clone()
            
            for segment_idx in range(1, num_segments):
                print(f"[predict] Continuation segment {segment_idx+1}/{num_segments}...")
                
                audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
                audio_end_idx = audio_start_idx + audio_stride * num_frames
                center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
                audio_emb = full_audio_emb[center_indices][None, ...].to(device)
                
                output_tuple = self.pipe.generate_avc(
                    video=current_video,
                    video_latent=latent,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_cond_frames=num_cond_frames,
                    num_inference_steps=8,
                    text_guidance_scale=text_guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                    generator=generator,
                    output_type="both",
                    use_kv_cache=True,
                    offload_kv_cache=False,
                    enhance_hf=False,  # False for distilled
                    audio_emb=audio_emb,
                    ref_latent=ref_latent,
                    ref_img_index=ref_img_index,
                    mask_frame_range=mask_frame_range,
                    use_distill=use_distill,
                )
                output, latent = output_tuple
                output = output[0]
                new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                new_video = [PIL.Image.fromarray(img_frame) for img_frame in new_video]
                del output
                torch.cuda.empty_cache()
                
                all_generated_frames.extend(new_video[num_cond_frames:])
                current_video = new_video
        
        # ---- Save output ----
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "output.mp4")
        
        output_tensor = torch.from_numpy(np.array(all_generated_frames))
        save_video_ffmpeg(output_tensor, output_path.replace(".mp4", ""), vocal_path, fps=save_fps, quality=5)
        
        elapsed = time.time() - t_start
        total_frames = len(all_generated_frames)
        duration = total_frames / save_fps
        print(f"[predict] ✅ Done: {total_frames} frames, {duration:.1f}s video, {elapsed:.0f}s wall time")
        
        # Cleanup temp files
        try:
            if audio.startswith("http"):
                os.unlink(audio_tmp.name)
        except:
            pass
        
        return output_path
