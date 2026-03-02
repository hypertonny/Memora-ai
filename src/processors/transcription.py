"""
Audio transcription using faster-whisper.
Supports Hindi, English, and Hinglish (code-mixed).
Model is loaded at startup for faster processing.
"""

import asyncio
from pathlib import Path
from typing import Optional
import logging

from faster_whisper import WhisperModel

from ..config import settings

logger = logging.getLogger(__name__)

# Global model instance - loaded once at startup
_whisper_model: Optional[WhisperModel] = None


def load_whisper_model() -> WhisperModel:
    """Load the Whisper model (called at startup)."""
    global _whisper_model
    
    if _whisper_model is None:
        model_name = settings.whisper_model
        logger.info(f"Loading faster-whisper model: {model_name}")
        
        # Use CUDA if available, otherwise CPU with int8 quantization
        try:
            _whisper_model = WhisperModel(
                model_name,
                device="cuda",
                compute_type="float16"
            )
            logger.info(f"Whisper model loaded on CUDA (float16)")
        except Exception:
            logger.info("CUDA not available, loading on CPU with int8")
            _whisper_model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8"
            )
            logger.info(f"Whisper model loaded on CPU (int8)")
    
    return _whisper_model


class TranscriptionProcessor:
    """
    Audio transcription processor using faster-whisper.
    Supports multilingual transcription including Hindi, English, and code-switching.
    """
    
    def __init__(self):
        """Initialize the transcription processor."""
        # Model is loaded globally at startup
        pass
    
    @property
    def model(self) -> WhisperModel:
        """Get the loaded Whisper model."""
        global _whisper_model
        if _whisper_model is None:
            load_whisper_model()
        return _whisper_model
    
    async def transcribe(
        self, 
        audio_path: Path,
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Optional language hint ('en', 'hi', or None for auto-detect)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with transcription results:
            {
                'text': str,           # Full transcription
                'language': str,       # Detected language
                'segments': list,      # Time-stamped segments
                'duration': float,     # Audio duration in seconds
            }
        """
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return {'text': '', 'language': '', 'segments': [], 'duration': 0}
        
        try:
            if progress_callback:
                await progress_callback("transcription", "starting", 0)
            
            # Run transcription in thread pool
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._transcribe_sync, str(audio_path), language
            )
            
            if progress_callback:
                await progress_callback("transcription", "completed", 100)
            
            return result
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return {'text': '', 'language': '', 'segments': [], 'error': str(e)}
    
    def _transcribe_sync(
        self, 
        audio_path: str, 
        language: Optional[str]
    ) -> dict:
        """Synchronous transcription logic using faster-whisper."""
        
        # Transcription options
        options = {
            'beam_size': 5,
            'vad_filter': True,  # Voice activity detection for faster processing
            'vad_parameters': dict(min_silence_duration_ms=500),
        }
        
        # Add language hint if provided
        if language:
            options['language'] = language
        
        # Run transcription
        segments, info = self.model.transcribe(audio_path, **options)
        
        # Collect segments
        all_segments = []
        full_text_parts = []
        
        for segment in segments:
            all_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
            full_text_parts.append(segment.text.strip())
        
        full_text = " ".join(full_text_parts)
        
        transcription = {
            'text': full_text,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'segments': all_segments,
        }
        
        logger.info(
            f"Transcribed {len(full_text)} chars in {info.duration:.1f}s "
            f"(language: {info.language}, prob: {info.language_probability:.2f})"
        )
        
        return transcription
    
    async def transcribe_with_timestamps(
        self, 
        audio_path: Path
    ) -> list[dict]:
        """
        Transcribe audio and return timestamped segments.
        Useful for video subtitles or detailed analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segments: [{'start': float, 'end': float, 'text': str}, ...]
        """
        result = await self.transcribe(audio_path)
        return result.get('segments', [])
    
    def get_text_only(self, transcription_result: dict) -> str:
        """
        Extract just the text from a transcription result.
        
        Args:
            transcription_result: Result from transcribe()
            
        Returns:
            Plain text transcription
        """
        return transcription_result.get('text', '')
