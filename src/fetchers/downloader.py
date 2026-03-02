"""
Media download utilities with async support and retry logic.
"""

import asyncio
from pathlib import Path
from typing import Optional
import logging
import hashlib

import httpx
import aiofiles

logger = logging.getLogger(__name__)


class MediaDownloader:
    """Async media downloader with retry support."""
    
    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 60.0,
        chunk_size: int = 8192
    ):
        """
        Initialize the downloader.
        
        Args:
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            chunk_size: Size of chunks for streaming downloads
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
    
    async def download(
        self,
        url: str,
        target_dir: Path,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download a file from URL to target directory.
        
        Args:
            url: URL to download from
            target_dir: Directory to save the file
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            Path to downloaded file or None on failure
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = self._guess_extension(url)
            filename = f"{url_hash}{ext}"
        
        target_path = target_dir / filename
        
        # Skip if already downloaded
        if target_path.exists():
            logger.info(f"File already exists: {target_path}")
            return target_path
        
        for attempt in range(self.max_retries):
            try:
                return await self._download_with_retry(url, target_path)
            except Exception as e:
                logger.warning(
                    f"Download attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts")
                    return None
        
        return None
    
    async def _download_with_retry(self, url: str, target_path: Path) -> Path:
        """Perform the actual download."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                
                async with aiofiles.open(target_path, "wb") as f:
                    async for chunk in response.aiter_bytes(self.chunk_size):
                        await f.write(chunk)
        
        logger.info(f"Downloaded: {target_path}")
        return target_path
    
    def _guess_extension(self, url: str) -> str:
        """Guess file extension from URL."""
        url_lower = url.lower()
        
        if ".mp4" in url_lower:
            return ".mp4"
        elif ".webm" in url_lower:
            return ".webm"
        elif ".mov" in url_lower:
            return ".mov"
        elif ".png" in url_lower:
            return ".png"
        elif ".webp" in url_lower:
            return ".webp"
        elif ".gif" in url_lower:
            return ".gif"
        elif ".jpg" in url_lower or ".jpeg" in url_lower:
            return ".jpg"
        else:
            # Default based on common patterns
            if "video" in url_lower:
                return ".mp4"
            return ".jpg"
    
    async def cleanup(self, path: Path) -> bool:
        """
        Remove a downloaded file.
        
        Args:
            path: Path to file to remove
            
        Returns:
            True if removed successfully
        """
        try:
            if path.exists():
                path.unlink()
                logger.info(f"Cleaned up: {path}")
                return True
        except Exception as e:
            logger.error(f"Failed to cleanup {path}: {e}")
        return False
