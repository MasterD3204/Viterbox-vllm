"""
Viterbox - Vietnamese Text-to-Speech with vLLM
Based on Chatterbox architecture, fine-tuned for Vietnamese.
Combines vLLM inference with original Viterbox processing pipeline.
"""
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

from src.chatterbox_vllm.models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings

import asyncio
import uuid
from vllm.sampling_params import RequestOutputKind

# Nếu môi trường của bạn hỗ trợ vllm.v1 (như doc bạn gửi)
try:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    # Fallback cho các bản vLLM cũ hơn nếu cần
    from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs

# Try to import Vietnamese normalizer
try:
    from soe_vinorm import SoeNormalizer
    _normalizer = SoeNormalizer()
    HAS_VINORM = True
except ImportError:
    HAS_VINORM = False
    _normalizer = None

REPO_ID = "dolly-vn/viterbox"
WAVS_DIR = Path("wavs")

# Supported languages
SUPPORTED_LANGUAGES = {
    "vi": "Vietnamese",
    "en": "English",
}

# ==================== VAD Model (Singleton) ====================
_VAD_MODEL = None
_VAD_UTILS = None


def get_vad_model():
    """Load Silero VAD model (singleton)"""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            _VAD_MODEL = model
            _VAD_UTILS = utils
        except Exception as e:
            print(f"⚠️ Could not load Silero VAD: {e}")
            return None, None
    return _VAD_MODEL, _VAD_UTILS


# ==================== Utility Functions ====================
def get_random_voice() -> Optional[Path]:
    """Get a random voice file from wavs folder"""
    if WAVS_DIR.exists():
        voices = list(WAVS_DIR.glob("*.wav"))
        if voices:
            import random
            return random.choice(voices)
    return None


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if len(text) > 0 and text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ('"', '"'),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text

def clean_text_s2(text: str) -> str:
    if not text:
        return ""
    
    # 1. Loại bỏ nội dung trong ngoặc đơn/kép/vuông
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    
    text = re.sub(
        r'(\d{4})[/-](0?\d{1,2})[/-](0?\d{1,2})',
        r'ngày \3 tháng \2 năm \1',
        text
    )

    # dd/mm/yyyy
    text = re.sub(
        r'(0?\d{1,2})[/-](0?\d{1,2})[/-](\d{4})',
        r'ngày \1 tháng \2 năm \3',
        text
    )
    text = re.sub(
    r'\bngày\s+(1|2|3|4|5|6|7|8|9|10)\b',
    r'ngày mùng \1',
    text
)
    # 3. Thay thế ký tự đặc biệt
    text = text.replace('%', ' phần trăm ')
    
    # 3.5 Thay ! bằng .
    text = text.replace('!', '.')
    
    # 4. Gộp số bị tách bởi dấu .
    # Ví dụ: 1.000.000 -> 1000000
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1\2', text)
    
    # 4.5 ✅ ĐỔI DẤU , GIỮA HAI SỐ THÀNH "phẩy"
    # Ví dụ: 159,2 -> 159 phẩy 2
    text = re.sub(r'(?<=\d),(?=\d)', ' phẩy ', text)
    
    # 5. Xóa dấu . trong viết tắt ngắn (F., J.)
    text = re.sub(r'(\s[A-Za-z]{1,2})\.\s+(?=[A-Za-zÀ-ỹ])', r'\1 ', text)
    
    # 6. Loại bỏ ký tự không hợp lệ
    text = re.sub(
        r'[^\w\s,.-\?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        r'ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]',
        ' ',
        text
    )
    
    # 7. Dọn dẹp khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_text(text: str, language: str = "vi") -> str:
    if language == "en":
        return text 
    """Normalize Vietnamese text (numbers, abbreviations, etc.)"""
    if language == "vi" and HAS_VINORM and _normalizer is not None:
        try:
            text = clean_text_s2(text)
            text = _normalizer.normalize(text)
            
            return text
        except Exception:
            return text
    return text


def _split_text_to_sentences(text: str) -> List[str]:
    """Split text into sentences by punctuation marks."""
    pattern = r'([.?!]+)'
    parts = re.split(pattern, text)
    
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if re.match(pattern, part):
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current = part
    
    if current.strip():
        sentences.append(current.strip())
    
    return [s for s in sentences if s.strip()]


# ==================== Audio Processing Functions ====================
def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Legacy trim silence (energy based)."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def vad_trim(audio: np.ndarray, sr: int, margin_s: float = 0.01) -> np.ndarray:
    """
    Trim audio using Silero VAD to strictly keep only speech.
    """
    if len(audio) == 0:
        return audio
        
    model, utils = get_vad_model()
    if model is None:
        return trim_silence(audio, sr, top_db=20)
        
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    wav = torch.tensor(audio, dtype=torch.float32)
    
    try:
        vad_sr = 16000
        if sr != vad_sr:
            wav_16k = librosa.resample(audio, orig_sr=sr, target_sr=vad_sr)
            wav_tensor = torch.tensor(wav_16k, dtype=torch.float32)
        else:
            wav_tensor = wav
            
        timestamps = get_speech_timestamps(
            wav_tensor, 
            model, 
            sampling_rate=vad_sr, 
            threshold=0.35,
            min_speech_duration_ms=250, 
            min_silence_duration_ms=100
        )
        
        if not timestamps:
            return trim_silence(audio, sr, top_db=25)
            
        last_end_sample_16k = timestamps[-1]['end']
        last_end_sample = int(last_end_sample_16k * (sr / vad_sr))
        margin_samples = int(margin_s * sr)
        cut_point = min(last_end_sample + margin_samples, len(audio))
        
        return audio[:cut_point]
        
    except Exception as e:
        print(f"⚠️ VAD Error: {e}")
        return trim_silence(audio, sr, top_db=20)


def apply_fade_out(audio: np.ndarray, sr: int, fade_duration: float = 0.01) -> np.ndarray:
    """Apply smooth fade-out to prevent click artifacts."""
    if len(audio) == 0:
        return audio
    
    fade_samples = min(int(fade_duration * sr), len(audio))
    if fade_samples <= 0:
        return audio
    
    fade_curve = np.linspace(1.0, 0.0, fade_samples)
    audio_copy = audio.copy()
    audio_copy[-fade_samples:] = audio_copy[-fade_samples:] * fade_curve
    
    return audio_copy


def apply_fade_in(audio: np.ndarray, sr: int, fade_duration: float = 0.005) -> np.ndarray:
    """Apply smooth fade-in to prevent click artifacts."""
    if len(audio) == 0:
        return audio
    
    fade_samples = min(int(fade_duration * sr), len(audio))
    if fade_samples <= 0:
        return audio
    
    fade_curve = np.linspace(0.0, 1.0, fade_samples)
    audio_copy = audio.copy()
    audio_copy[:fade_samples] = audio_copy[:fade_samples] * fade_curve
    
    return audio_copy


def crossfade_concat(audios: List[np.ndarray], sr: int, fade_ms: int = 50, pause_ms: int = 500) -> np.ndarray:
    """Concatenate audio segments with crossfading and optional pause."""
    if not audios:
        return np.array([])
    if len(audios) == 1:
        return audios[0]
    
    fade_samples = int(sr * fade_ms / 1000)
    pause_samples = int(sr * pause_ms / 1000)
    
    result = audios[0].copy()
    
    for i in range(1, len(audios)):
        next_audio = audios[i]
        
        if pause_samples > 0:
            silence = np.zeros(pause_samples, dtype=result.dtype)
            result = np.concatenate([result, silence])
        
        if len(result) < fade_samples or len(next_audio) < fade_samples:
            result = np.concatenate([result, next_audio])
            continue
        
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        
        result_end = result[-fade_samples:] * fade_out
        next_start = next_audio[:fade_samples] * fade_in
        crossfaded = result_end + next_start
        
        result = np.concatenate([
            result[:-fade_samples],
            crossfaded,
            next_audio[fade_samples:]
        ])
    
    return result


# ==================== Conditionals Dataclass ====================
@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    """
    t3: T3Cond
    gen: dict
    ref_wav: Optional[torch.Tensor] = None

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, path):
        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            elif isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            elif hasattr(x, '__dict__'):
                return {k: to_cpu(v) for k, v in vars(x).items()}
            return x
        
        torch.save({
            't3': to_cpu(self.t3),
            'gen': to_cpu(self.gen),
        }, path)

    @classmethod
    def load(cls, fpath, device='cpu'):
        kwargs = torch.load(fpath, weights_only=True, map_location=device)
        t3_data = kwargs.get('t3', {})
        gen_data = kwargs.get('gen', kwargs.get('s3', {}))
        ref_wav = kwargs.get('ref_wav', None)
        
        if isinstance(t3_data, dict):
            t3_cond = T3Cond(**t3_data)
        else:
            t3_cond = t3_data
            
        return cls(t3_cond, gen_data, ref_wav)


# ==================== Main TTS Class ====================
class ChatterboxTTS:
    """
    Vietnamese Text-to-Speech model with vLLM backend.
    Combines vLLM inference efficiency with Viterbox processing pipeline.
    
    Example:
        >>> tts = ChatterboxTTS.from_pretrained("cuda")
        >>> audio = tts.generate("Xin chào!")
        >>> tts.save_audio(audio, "output.wav")
    """
    
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self, 
        target_device: str, 
        max_model_len: int,
        t3: LLM, 
        t3_config: T3Config, 
        t3_cond_enc: T3CondEnc, 
        t3_speech_emb: torch.nn.Embedding, 
        t3_speech_pos_emb: LearnedPositionEmbeddings,
        s3gen: S3Gen, 
        ve: VoiceEncoder, 
        default_conds: Conditionals,
        variant: str = "vi"
    ):
        self.target_device = target_device
        self.device = target_device  # Alias for compatibility
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb
        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.conds: Optional[Conditionals] = default_conds  # Current conditioning
        self.variant = variant
        self.sr = S3GEN_SR  # Output sample rate (24kHz)

    @classmethod
    def from_local(
        cls, 
        ckpt_dir: Union[str, Path], 
        target_device: str = "cuda", 
        max_model_len: int = 1000, 
        compile: bool = False,
        max_batch_size: int = 10,
        variant: str = "vi",
        s3gen_use_fp16: bool = False,
        **kwargs
    ) -> 'ChatterboxTTS':
        """Load model from local directory"""
        ckpt_dir = Path(ckpt_dir)
        t3_config = T3Config()

        # Select checkpoint file based on variant
        if variant == "english":
            t3_ckpt_file = "t3_cfg.safetensors"
        else:
            t3_ckpt_file = "t3_ml24ls_v2_merged.safetensors"
        
        t3_weights = load_file(ckpt_dir / t3_ckpt_file)

        # ========== T3 Conditional Encoder ==========
        t3_enc = T3CondEnc(t3_config)
        t3_enc_state = {
            k.replace('cond_enc.', ''): v 
            for k, v in t3_weights.items() 
            if k.startswith('cond_enc.')
        }
        t3_enc.load_state_dict(t3_enc_state)
        t3_enc = t3_enc.to(device=target_device).eval()

        # ========== Speech Embedding ==========
        t3_speech_emb = torch.nn.Embedding(
            t3_config.speech_tokens_dict_size, 
            t3_config.n_channels
        )
        t3_speech_emb_state = {
            k.replace('speech_emb.', ''): v 
            for k, v in t3_weights.items() 
            if k.startswith('speech_emb.') and not k.startswith('speech_emb.emb')
        }
        t3_speech_emb.load_state_dict(t3_speech_emb_state)
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        # ========== Speech Position Embedding ==========
        t3_speech_pos_emb = LearnedPositionEmbeddings(
            t3_config.max_speech_tokens + 2 + 2, 
            t3_config.n_channels
        )
        t3_speech_pos_emb_state = {
            k.replace('speech_pos_emb.', ''): v 
            for k, v in t3_weights.items() 
            if k.startswith('speech_pos_emb.')
        }
        t3_speech_pos_emb.load_state_dict(t3_speech_pos_emb_state)
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        # ========== GPU Memory Calculation ==========
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        # ========== vLLM LLM ==========
        base_vllm_kwargs = {
            "model": "./t3-model" if variant == "vi" else "./t3-model-multilingual",
            "task": "generate",
            "tokenizer": "ViTokenizer" if variant == "vi" else "MtlTokenizer",
            "tokenizer_mode": "custom",
            # "gpu_memory_utilization": vllm_memory_percent,
            "gpu_memory_utilization": 0.6,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }
        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        # ========== Voice Encoder ==========
        ve = VoiceEncoder()
        ve_path = ckpt_dir / "ve.safetensors"
        ve_path_pt = ckpt_dir / "ve.pt"
        
        if ve_path.exists():
            ve.load_state_dict(load_file(ve_path))
        elif ve_path_pt.exists():
            ve.load_state_dict(torch.load(ve_path_pt, map_location=target_device, weights_only=True))
        else:
            raise FileNotFoundError(f"Cannot find ve.safetensors or ve.pt in {ckpt_dir}")
        
        ve = ve.to(device=target_device).eval()

        # ========== S3Gen ==========
        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen_path = ckpt_dir / "s3gen.safetensors"
        s3gen_path_pt = ckpt_dir / "s3gen.pt"
        
        if s3gen_path.exists():
            s3gen.load_state_dict(load_file(s3gen_path), strict=False)
        elif s3gen_path_pt.exists():
            s3gen.load_state_dict(
                torch.load(s3gen_path_pt, map_location=target_device, weights_only=True), 
                strict=False
            )
        else:
            raise FileNotFoundError(f"Cannot find s3gen.safetensors or s3gen.pt in {ckpt_dir}")
        
        s3gen = s3gen.to(device=target_device).eval()

        # ========== Default Conditionals ==========
        default_conds = Conditionals.load(ckpt_dir / "conds.pt", device=target_device)
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, 
            max_model_len=max_model_len,
            t3=t3, 
            t3_config=t3_config, 
            t3_cond_enc=t3_enc, 
            t3_speech_emb=t3_speech_emb, 
            t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, 
            ve=ve, 
            default_conds=default_conds,
            variant=variant,
        )

    @classmethod
    def from_pretrained(
        cls,
        target_device: str = "cuda",
        repo_id: str = REPO_ID,
        *args, **kwargs
    ) -> 'ChatterboxTTS':
        """Load model from HuggingFace Hub"""
        local_path = "/data/MultiSpeakerXphoneBert/dqdung/viterbox_model"
        
        # Ensure the symlink points to correct checkpoint
        t3_cfg_path = Path(local_path) / "t3_ml24ls_v2.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.parent.mkdir(parents=True, exist_ok=True)
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path), target_device=target_device, variant="vi", *args, **kwargs)

    def get_supported_languages(self) -> dict:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        elif self.variant == "vi":
            return {"vi": "Vietnamese", "en": "English"}
        else:
            return {"en": "English"}

    def prepare_conditionals(
        self, 
        audio_prompt: Union[str, Path, torch.Tensor], 
        exaggeration: float = 0.5
    ) -> Conditionals:
        """
        Prepare conditioning from reference audio.
        
        Args:
            audio_prompt: Path to WAV file or audio tensor
            exaggeration: Expression intensity (0.0 - 2.0)
        """
        # Load audio at S3Gen sample rate (24kHz)
        if isinstance(audio_prompt, (str, Path)):
            s3gen_ref_wav, _ = librosa.load(str(audio_prompt), sr=S3GEN_SR, mono=True)
        else:
            s3gen_ref_wav = audio_prompt.cpu().numpy()
            if s3gen_ref_wav.ndim > 1:
                s3gen_ref_wav = s3gen_ref_wav.squeeze()
        
        # Resample to 16kHz for voice encoder and tokenizer
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        
        # Limit conditioning length
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        
        with torch.inference_mode():
            # Get S3Gen conditioning
            s3_cond = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.target_device)
            
            # Speech cond prompt tokens for T3
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:self.ENC_COND_LEN]], 
                max_len=self.t3_config.speech_cond_prompt_len
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.target_device)
            
            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.target_device)
            
            # Create T3Cond
            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1),
            ).to(device=self.target_device)
        
        self.conds = Conditionals(
            t3=t3_cond, 
            gen=s3_cond, 
            ref_wav=torch.from_numpy(s3gen_ref_wav).unsqueeze(0)
        )
        return self.conds

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict, torch.Tensor]:
        """Get cached audio conditionals for vLLM inference."""
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            # Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:self.ENC_COND_LEN]], 
                max_len=self.t3_config.speech_cond_prompt_len
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = (
            self.t3_speech_emb(t3_cond_prompt_tokens)[0] + 
            self.t3_speech_pos_emb(t3_cond_prompt_tokens)
        )

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        """Update exaggeration in conditioning embedding."""
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    def _generate_single_vllm(
        self,
        text: str,
        s3gen_ref: dict,
        cond_emb: torch.Tensor,
        language: str,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
        diffusion_steps: int,
    ) -> np.ndarray:
        """Generate audio for a single text using vLLM."""
        # Normalize and prepare text
        text = punc_norm(text)
        prompt = "[START]" + text + "[STOP]"
        
        # Add language prefix for multilingual
        if self.variant in ["vi", "multilingual"]:
            prompt = f"<{language.lower()}>{prompt}"

        with torch.inference_mode():
            # Generate speech tokens with vLLM
            batch_results = self.t3.generate(
                [{
                    "prompt": prompt,
                    "multi_modal_data": {
                        "conditionals": [cond_emb],
                    },
                }],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            )

            # Process output
            for batch_result in batch_results:
                for output in batch_result.outputs:
                    speech_tokens = torch.tensor(
                        [token - SPEECH_TOKEN_OFFSET for token in output.token_ids], 
                        device=self.target_device
                    )
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    
                    # Remove last token to prevent click artifacts
                    if len(speech_tokens) > 1:
                        speech_tokens = speech_tokens[:-1]

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )
                    return wav[0].cpu().numpy()
        
        return np.zeros(self.sr)  # Fallback

    def _detect_and_remove_repetition(self, speech_tokens: torch.Tensor, min_repeat_len: int = 10) -> torch.Tensor:
        """
        Detect and remove repetition patterns at the end of speech tokens.
        
        Args:
            speech_tokens: Generated speech tokens
            min_repeat_len: Minimum length to consider as repetition pattern
        
        Returns:
            Cleaned speech tokens
        """
        if len(speech_tokens) < min_repeat_len * 2:
            return speech_tokens
        
        tokens_np = speech_tokens.cpu().numpy()
        
        # Check for repetition patterns at the end
        for pattern_len in range(min_repeat_len, len(tokens_np) // 3):
            # Get the last pattern_len tokens
            end_pattern = tokens_np[-pattern_len:]
            
            # Check if this pattern repeats before
            prev_pattern = tokens_np[-2*pattern_len:-pattern_len]
            
            if np.array_equal(end_pattern, prev_pattern):
                # Found repetition, truncate
                print(f"⚠️ Detected repetition pattern (len={pattern_len}), truncating...")
                return speech_tokens[:-pattern_len]
        
        return speech_tokens


    def _generate_batch_vllm(
        self,
        texts: List[str],
        s3gen_ref: dict,
        cond_emb: torch.Tensor,
        language: str,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
        diffusion_steps: int,
    ) -> List[np.ndarray]:
        """Generate audio for multiple texts using vLLM batch inference."""
        
        # Prepare all prompts
        prompts = []
        for text in texts:
            text = punc_norm(text)
            prompt = "[START]" + text + "[STOP]"
            
            if self.variant in ["vi", "multilingual"]:
                prompt = f"<{language.lower()}>{prompt}"
            prompts.append(prompt)

        with torch.inference_mode():
            start_time = time.time()
            batch_results = self.t3.generate(
                [{
                    "prompt": prompt,
                    "multi_modal_data": {
                        "conditionals": [cond_emb],
                    },
                } for prompt in prompts],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    min_p=0.05,  # Thêm min_p để tránh low-probability tokens
                    repetition_penalty=repetition_penalty,
                    # Thêm các params để giảm repetition
                    presence_penalty=0.5,  # Penalize tokens đã xuất hiện
                    frequency_penalty=0.5,  # Penalize tokens xuất hiện nhiều lần
                )
            )
            t3_time = time.time() - start_time
            print(f"[T3] Batch generation: {t3_time:.2f}s for {len(texts)} sentences")

            torch.cuda.empty_cache()

            start_time = time.time()
            audio_results = []
            
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    speech_tokens = torch.tensor(
                        [token - SPEECH_TOKEN_OFFSET for token in output.token_ids], 
                        device=self.target_device
                    )
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    
                    # ========== FIX: Detect and remove repetition ==========
                    speech_tokens = self._detect_and_remove_repetition(speech_tokens)
                    
                    # Remove last token to prevent click artifacts
                    if len(speech_tokens) > 1:
                        speech_tokens = speech_tokens[:-1]

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )
                    audio_results.append(wav[0].cpu().numpy())
                    
                    if (i + 1) % 10 == 0:
                        torch.cuda.empty_cache()
            
            s3_time = time.time() - start_time
            print(f"[S3Gen] Waveform generation: {s3_time:.2f}s")
            
            return audio_results

    def generate(
        self,
        text: str,
        language: str = "vi",
        audio_prompt: Optional[Union[str, Path, torch.Tensor]] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 2.0,
        split_sentences: bool = True,
        crossfade_ms: int = 50,
        sentence_pause_ms: int = 500,
        max_tokens: int = 1000,
        diffusion_steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate speech from text (Viterbox-style interface).
        
        Args:
            text: Input text to synthesize
            language: Language code ('vi' or 'en')
            audio_prompt: Optional reference audio for voice cloning
            exaggeration: Expression intensity (0.0 - 2.0)
            cfg_weight: Classifier-free guidance weight (unused in vLLM, kept for compatibility)
            temperature: Sampling temperature (0.1 - 1.0)
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty for T3
            split_sentences: Whether to split text by punctuation
            crossfade_ms: Crossfade duration in milliseconds
            sentence_pause_ms: Pause duration between sentences
            max_tokens: Maximum tokens to generate
            diffusion_steps: Number of diffusion steps for S3Gen
            
        Returns:
            Audio tensor (1, samples) at 24kHz
        """
        # Prepare conditioning
        audio_prompt_path = None
        if audio_prompt is not None:
            if isinstance(audio_prompt, (str, Path)):
                audio_prompt_path = str(audio_prompt)
                self.prepare_conditionals(audio_prompt, exaggeration)
            else:
                self.prepare_conditionals(audio_prompt, exaggeration)
        elif self.conds is None:
            random_voice = get_random_voice()
            if random_voice is not None:
                audio_prompt_path = str(random_voice)
                self.prepare_conditionals(random_voice, exaggeration)
            else:
                # Use default conds
                self.conds = self.default_conds

        # Get vLLM conditionals
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)
        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Normalize text
        # text = normalize_dot_spacing(text)
        text = normalize_text(text, language)
        
        if split_sentences:
            sentences = _split_text_to_sentences(text)
            
            if len(sentences) == 0:
                sentences = [text]
            
            print(f"📝 Processing {len(sentences)} sentences in batch...")
            for i, s in enumerate(sentences):
                print(f"  [{i+1}/{len(sentences)}] {s[:50]}...")
            
            # ========== BATCH GENERATION ==========
            audio_list = self._generate_batch_vllm(
                texts=sentences,
                s3gen_ref=s3gen_ref,
                cond_emb=cond_emb,
                language=language,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                diffusion_steps=diffusion_steps,
            )
            
            # Post-process each audio segment
            audio_segments = []
            for i, audio_np in enumerate(audio_list):
                # VAD trim
                audio_np = vad_trim(audio_np, self.sr, margin_s=0.05)
                
                # Apply fades
                audio_np = apply_fade_out(audio_np, self.sr, fade_duration=0.01)
                audio_np = apply_fade_in(audio_np, self.sr, fade_duration=0.005)
                
                if len(audio_np) > 0:
                    audio_segments.append(audio_np)
            
            # Merge with crossfading
            if audio_segments:
                merged = crossfade_concat(
                    audio_segments, self.sr, 
                    fade_ms=crossfade_ms, 
                    pause_ms=sentence_pause_ms
                )
                merged = apply_fade_out(merged, self.sr, fade_duration=0.015)
                return torch.from_numpy(merged).unsqueeze(0)
            else:
                return torch.zeros(1, self.sr)
        else:
            # Single generation (still uses batch with 1 item for consistency)
            audio_list = self._generate_batch_vllm(
                texts=[text],
                s3gen_ref=s3gen_ref,
                cond_emb=cond_emb,
                language=language,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                diffusion_steps=diffusion_steps,
            )
            return torch.from_numpy(audio_list[0]).unsqueeze(0)

    def save_audio(
        self, 
        audio: torch.Tensor, 
        path: Union[str, Path], 
        trim_silence_flag: bool = True
    ):
        """Save audio to file."""
        import soundfile as sf
        
        audio_np = audio[0].cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
        
        if trim_silence_flag:
            audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
        
        sf.write(str(path), audio_np, self.sr)

    def shutdown(self):
        """Clean up resources."""
        del self.t3
        torch.cuda.empty_cache()


