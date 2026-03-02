"""
YouTube fetcher using yt-dlp for video/shorts download.
"""

import asyncio
import re
from pathlib import Path
from typing import Optional
import logging

from .base import BaseFetcher, FetchResult, MediaItem, MediaType, Platform
from ..config import settings

logger = logging.getLogger(__name__)


class YouTubeFetcher(BaseFetcher):
    """
    YouTube content fetcher using yt-dlp.
    Supports regular videos and Shorts.
    """
    
    def __init__(self):
        self.download_dir = settings.videos_dir
    
    def extract_shortcode(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'youtube\.com/watch\?v=([A-Za-z0-9_-]+)',
            r'youtube\.com/shorts/([A-Za-z0-9_-]+)',
            r'youtu\.be/([A-Za-z0-9_-]+)',
            r'youtube\.com/embed/([A-Za-z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def fetch(self, url: str) -> FetchResult:
        """Fetch YouTube video content using yt-dlp."""
        video_id = self.extract_shortcode(url)
        
        if not video_id:
            return FetchResult(
                platform=Platform.YOUTUBE,
                url=url,
                shortcode="",
                error="Could not extract video ID from URL"
            )
        
        logger.info(f"Fetching YouTube content: {video_id}")
        
        try:
            result = await self._fetch_with_ytdlp(url, video_id)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching YouTube video {video_id}: {e}")
            return FetchResult(
                platform=Platform.YOUTUBE,
                url=url,
                shortcode=video_id,
                error=str(e)
            )
    
    async def _fetch_with_ytdlp(self, url: str, video_id: str) -> FetchResult:
        """Fetch using yt-dlp."""
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
        
        loop = asyncio.get_running_loop()
        
        def extract_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        
        info = await loop.run_in_executor(None, extract_info)
        
        if not info:
            return FetchResult(
                platform=Platform.YOUTUBE,
                url=url,
                shortcode=video_id,
                error="Could not extract video info"
            )
        
        # Build media item
        media_items = [MediaItem(
            media_type=MediaType.VIDEO,
            url=url,
        )]
        
        # Extract metadata
        caption = info.get('description') or info.get('title') or ""
        title = info.get('title') or ""
        author = info.get('uploader') or info.get('channel') or ""
        
        # Include title in caption if different
        if title and title not in caption:
            caption = f"{title}\n\n{caption}"
        
        return FetchResult(
            platform=Platform.YOUTUBE,
            url=url,
            shortcode=video_id,
            author=author,
            caption=caption,
            media_items=media_items,
            raw_data=info,
        )
    
    async def download_media(self, result: FetchResult) -> FetchResult:
        """Download video using yt-dlp."""
        import yt_dlp
        
        if not result.success or not result.media_items:
            return result
        
        loop = asyncio.get_running_loop()
        
        for item in result.media_items:
            try:
                if item.media_type == MediaType.VIDEO:
                    # Use exact filename with extension to avoid ambiguity
                    output_filename = f"{result.shortcode}.mp4"
                    output_path = self.download_dir / output_filename
                    
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'outtmpl': str(output_path),  # Exact path
                        'format': 'best[ext=mp4]/best',
                        'merge_output_format': 'mp4',
                    }
                    
                    def download():
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([result.url])
                    
                    await loop.run_in_executor(None, download)
                    
                    # Verify file exists
                    if output_path.exists():
                        item.local_path = output_path
                        logger.info(f"Downloaded video to {output_path}")
                    else:
                        logger.warning(f"Expected video at {output_path} but not found")
                        # Try to find any file with the shortcode
                        for file in self.download_dir.glob(f"{result.shortcode}*"):
                            if file.name.startswith(result.shortcode) and file.suffix in ['.mp4', '.webm', '.mkv']:
                                item.local_path = file
                                logger.info(f"Found video at alternative path: {file}")
                                break
                        
            except Exception as e:
                logger.error(f"Error downloading YouTube video: {e}")
        
        return result
