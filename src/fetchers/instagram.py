"""
Instagram fetcher using yt-dlp (more reliable for public content).
Falls back to Instaloader for additional metadata.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional
import logging
import re

from .base import BaseFetcher, FetchResult, MediaItem, MediaType, Platform
from ..config import settings

logger = logging.getLogger(__name__)


class InstagramFetcher(BaseFetcher):
    """
    Instagram content fetcher using yt-dlp for video/image download
    and optional Instaloader for metadata.
    """
    
    def __init__(self):
        self.download_dir = settings.videos_dir
    
    def extract_shortcode(self, url: str) -> Optional[str]:
        """Extract shortcode from Instagram URL."""
        patterns = [
            r'instagram\.com/(?:p|reel|reels)/([A-Za-z0-9_-]+)',
            r'instagr\.am/(?:p|reel)/([A-Za-z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def fetch(self, url: str) -> FetchResult:
        """Fetch Instagram post/reel content using yt-dlp."""
        shortcode = self.extract_shortcode(url)
        
        if not shortcode:
            return FetchResult(
                platform=Platform.INSTAGRAM,
                url=url,
                shortcode="",
                error="Could not extract shortcode from URL"
            )
        
        logger.info(f"Fetching Instagram content: {shortcode}")
        
        try:
            # Use yt-dlp to get info and download
            result = await self._fetch_with_ytdlp(url, shortcode)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Instagram post {shortcode}: {e}")
            return FetchResult(
                platform=Platform.INSTAGRAM,
                url=url,
                shortcode=shortcode,
                error=str(e)
            )
    
    async def _fetch_with_ytdlp(self, url: str, shortcode: str) -> FetchResult:
        """Fetch using yt-dlp."""
        import yt_dlp
        
        # Configure yt-dlp with browser cookies for authentication
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,  # First just get info
            'cookiesfrombrowser': ('firefox',),  # Use Firefox cookies
        }
        
        # Get video info first
        loop = asyncio.get_running_loop()
        
        def extract_info():
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=False)
            except Exception as e:
                # Try without cookies as fallback
                logger.warning(f"Failed with browser cookies, trying without: {e}")
                ydl_opts_no_cookies = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'skip_download': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts_no_cookies) as ydl:
                    return ydl.extract_info(url, download=False)
        
        info = await loop.run_in_executor(None, extract_info)
        
        if not info:
            return FetchResult(
                platform=Platform.INSTAGRAM,
                url=url,
                shortcode=shortcode,
                error="Could not extract video info"
            )
        
        # Build media items
        media_items = []
        
        # Check if it's a video or image
        if info.get('ext') in ['mp4', 'webm', 'mov'] or info.get('_type') == 'video':
            media_items.append(MediaItem(
                media_type=MediaType.VIDEO,
                url=info.get('url') or url,
            ))
        elif info.get('ext') in ['jpg', 'jpeg', 'png', 'webp']:
            media_items.append(MediaItem(
                media_type=MediaType.IMAGE,
                url=info.get('url') or info.get('thumbnail'),
            ))
        else:
            # Default to video for reels
            media_items.append(MediaItem(
                media_type=MediaType.VIDEO,
                url=url,
            ))
        
        # Extract caption/description
        caption = info.get('description') or info.get('title') or ""
        author = info.get('uploader') or info.get('channel') or ""
        
        return FetchResult(
            platform=Platform.INSTAGRAM,
            url=url,
            shortcode=shortcode,
            author=author,
            caption=caption,
            media_items=media_items,
            raw_data=info,
        )
    
    async def download_media(self, result: FetchResult) -> FetchResult:
        """Download media files using yt-dlp."""
        import yt_dlp
        
        if not result.success or not result.media_items:
            return result
        
        loop = asyncio.get_running_loop()
        
        for item in result.media_items:
            try:
                if item.media_type == MediaType.VIDEO:
                    # Download video with yt-dlp
                    output_path = self.download_dir / f"{result.shortcode}.mp4"
                    
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'outtmpl': str(output_path),
                        'format': 'best[ext=mp4]/best',
                        'cookiesfrombrowser': ('firefox',),
                    }
                    
                    def download():
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                ydl.download([result.url])
                        except Exception:
                            # Try without cookies
                            ydl_opts_no_cookies = {
                                'quiet': True,
                                'no_warnings': True,
                                'outtmpl': str(output_path),
                                'format': 'best[ext=mp4]/best',
                            }
                            with yt_dlp.YoutubeDL(ydl_opts_no_cookies) as ydl:
                                ydl.download([result.url])
                    
                    await loop.run_in_executor(None, download)
                    
                    if output_path.exists():
                        item.local_path = output_path
                        logger.info(f"Downloaded video to {output_path}")
                    else:
                        # Check for different extensions
                        for ext in ['.mp4', '.webm', '.mkv']:
                            alt_path = self.download_dir / f"{result.shortcode}{ext}"
                            if alt_path.exists():
                                item.local_path = alt_path
                                logger.info(f"Downloaded video to {alt_path}")
                                break
                
                elif item.media_type == MediaType.IMAGE and item.url:
                    # Download image
                    output_path = await self._download_file(
                        item.url, 
                        settings.images_dir / f"{result.shortcode}.jpg"
                    )
                    if output_path:
                        item.local_path = output_path
                        
            except Exception as e:
                logger.error(f"Error downloading media: {e}")
        
        return result
    
    async def _download_file(self, url: str, output_path: Path) -> Optional[Path]:
        """Download a file from URL."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)
                
                return output_path
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
