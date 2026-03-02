"""
Video processing utilities for audio extraction.
Uses FFmpeg for media manipulation.
"""

import asyncio
from pathlib import Path
from typing import Optional
import logging
import subprocess
import tempfile

from ..config import settings

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processor for extracting audio from video files.
    Uses FFmpeg under the hood.
    """
    
    def __init__(self):
        """Initialize the video processor."""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("FFmpeg is available")
                return True
        except FileNotFoundError:
            logger.warning(
                "FFmpeg not found. Please install FFmpeg for video processing."
            )
        return False
    
    async def extract_audio(
        self, 
        video_path: Path,
        output_path: Optional[Path] = None,
        format: str = "wav"
    ) -> Optional[Path]:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio file
            format: Audio format (wav, mp3, m4a)
            
        Returns:
            Path to extracted audio file or None on failure
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            output_path = settings.audio_dir / f"{video_path.stem}.{format}"
        
        # Skip if already extracted
        if output_path.exists():
            logger.info(f"Audio already extracted: {output_path}")
            return output_path
        
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._extract_audio_sync, video_path, output_path, format
            )
            return result
        except Exception as e:
            logger.error(f"Audio extraction failed for {video_path}: {e}")
            return None
    
    def _extract_audio_sync(
        self, 
        video_path: Path, 
        output_path: Path,
        format: str
    ) -> Optional[Path]:
        """Synchronous audio extraction using FFmpeg."""
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', self._get_audio_codec(format),
            '-ar', '16000',  # 16kHz sample rate (good for speech)
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        logger.info(f"Extracting audio from {video_path}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
        
        if output_path.exists():
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
        
        return None
    
    def _get_audio_codec(self, format: str) -> str:
        """Get the appropriate audio codec for the format."""
        codecs = {
            'wav': 'pcm_s16le',
            'mp3': 'libmp3lame',
            'm4a': 'aac',
            'ogg': 'libvorbis',
            'flac': 'flac',
        }
        return codecs.get(format, 'pcm_s16le')
    
    async def get_video_info(self, video_path: Path) -> dict:
        """
        Get video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video info (duration, resolution, etc.)
        """
        if not video_path.exists():
            return {}
        
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._get_video_info_sync, video_path
            )
            return result
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
    
    def _get_video_info_sync(self, video_path: Path) -> dict:
        """Get video info using FFprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            try:
                data = json.loads(result.stdout)
                
                # Extract useful info
                format_info = data.get('format', {})
                streams = data.get('streams', [])
                
                video_stream = next(
                    (s for s in streams if s.get('codec_type') == 'video'),
                    {}
                )
                audio_stream = next(
                    (s for s in streams if s.get('codec_type') == 'audio'),
                    {}
                )
                
                return {
                    'duration': float(format_info.get('duration', 0)),
                    'size': int(format_info.get('size', 0)),
                    'format': format_info.get('format_name', ''),
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'fps': video_stream.get('r_frame_rate'),
                    'audio_codec': audio_stream.get('codec_name'),
                    'sample_rate': audio_stream.get('sample_rate'),
                }
            except json.JSONDecodeError:
                pass
        
        return {}
    
    async def cleanup_audio(self, audio_path: Path) -> bool:
        """
        Remove extracted audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if successfully removed
        """
        try:
            if audio_path.exists():
                audio_path.unlink()
                logger.info(f"Cleaned up audio: {audio_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to cleanup audio {audio_path}: {e}")
        return False
