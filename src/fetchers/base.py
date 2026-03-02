"""
Base fetcher interface and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import re


class Platform(str, Enum):
    """Supported social media platforms."""
    INSTAGRAM = "instagram"
    THREADS = "threads"
    YOUTUBE = "youtube"


class MediaType(str, Enum):
    """Types of media content."""
    IMAGE = "image"
    VIDEO = "video"
    CAROUSEL = "carousel"


@dataclass
class MediaItem:
    """Represents a single media item (image or video)."""
    media_type: MediaType
    url: str
    local_path: Optional[Path] = None
    extracted_text: Optional[str] = None
    raw_data: Optional[dict] = None


@dataclass
class FetchResult:
    """Result from fetching content from a social media post."""
    platform: Platform
    url: str
    shortcode: str
    caption: str = ""
    pinned_comments: list[str] = field(default_factory=list)
    media_items: list[MediaItem] = field(default_factory=list)
    author: str = ""
    timestamp: Optional[str] = None
    error: Optional[str] = None
    raw_data: Optional[dict] = None
    
    @property
    def success(self) -> bool:
        """Check if fetch was successful."""
        return self.error is None
    
    @property
    def has_video(self) -> bool:
        """Check if result contains video content."""
        return any(m.media_type == MediaType.VIDEO for m in self.media_items)
    
    @property
    def has_images(self) -> bool:
        """Check if result contains image content."""
        return any(m.media_type == MediaType.IMAGE for m in self.media_items)


class BaseFetcher(ABC):
    """Abstract base class for content fetchers."""
    
    platform: Platform
    
    @abstractmethod
    async def fetch(self, url: str) -> FetchResult:
        """
        Fetch content from the given URL.
        
        Args:
            url: The social media post URL
            
        Returns:
            FetchResult containing the fetched content
        """
        pass
    
    @abstractmethod
    def extract_shortcode(self, url: str) -> Optional[str]:
        """
        Extract the post shortcode/ID from the URL.
        
        Args:
            url: The social media post URL
            
        Returns:
            The shortcode/ID or None if invalid URL
        """
        pass
    
    @classmethod
    def detect_platform(cls, url: str) -> Optional[Platform]:
        """
        Detect which platform a URL belongs to.
        
        Args:
            url: The URL to check
            
        Returns:
            The detected Platform or None
        """
        url_lower = url.lower()
        
        if "instagram.com" in url_lower or "instagr.am" in url_lower:
            return Platform.INSTAGRAM
        elif "threads.net" in url_lower:
            return Platform.THREADS
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return Platform.YOUTUBE
        
        return None
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
