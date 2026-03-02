"""
Threads content fetcher using Playwright for browser automation.
"""

import asyncio
import re
from pathlib import Path
from typing import Optional
import logging

from ..config import settings
from .base import BaseFetcher, FetchResult, MediaItem, MediaType, Platform
from .downloader import MediaDownloader

logger = logging.getLogger(__name__)


class ThreadsFetcher(BaseFetcher):
    """Fetcher for Threads posts using Playwright browser automation."""
    
    platform = Platform.THREADS
    
    # URL pattern for Threads
    POST_PATTERN = re.compile(
        r'(?:https?://)?(?:www\.)?threads\.net/@([^/]+)/post/([A-Za-z0-9_-]+)/?'
    )
    
    def __init__(self):
        """Initialize the Threads fetcher."""
        self.downloader = MediaDownloader()
        self._browser = None
        self._playwright = None
    
    def extract_shortcode(self, url: str) -> Optional[str]:
        """Extract Threads post ID from URL."""
        match = self.POST_PATTERN.search(url)
        if match:
            return match.group(2)
        return None
    
    def _extract_username(self, url: str) -> Optional[str]:
        """Extract username from Threads URL."""
        match = self.POST_PATTERN.search(url)
        if match:
            return match.group(1)
        return None
    
    async def _ensure_browser(self):
        """Ensure Playwright browser is initialized."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True
                )
                logger.info("Playwright browser initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Playwright: {e}")
                raise
    
    async def _close_browser(self):
        """Close the browser instance."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    async def fetch(self, url: str) -> FetchResult:
        """
        Fetch content from a Threads post.
        
        Args:
            url: Threads post URL
            
        Returns:
            FetchResult with the fetched content
        """
        shortcode = self.extract_shortcode(url)
        username = self._extract_username(url)
        
        if not shortcode:
            return FetchResult(
                platform=self.platform,
                url=url,
                shortcode="",
                error="Invalid Threads URL"
            )
        
        try:
            await self._ensure_browser()
            
            context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            try:
                # Navigate to the post
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait for content to load
                await page.wait_for_timeout(2000)
                
                # Extract post content
                result = await self._extract_content(page, url, shortcode, username)
                return result
                
            finally:
                await context.close()
                
        except Exception as e:
            logger.error(f"Error fetching Threads post {shortcode}: {e}")
            return FetchResult(
                platform=self.platform,
                url=url,
                shortcode=shortcode,
                error=str(e)
            )
    
    async def _extract_content(
        self, 
        page, 
        url: str, 
        shortcode: str, 
        username: Optional[str]
    ) -> FetchResult:
        """Extract content from the loaded page."""
        
        media_items = []
        caption = ""
        pinned_comments = []
        
        try:
            # Try to extract the main post text
            # Threads uses various selectors, try common ones
            text_selectors = [
                '[data-pressable-container="true"] span',
                'article span[dir="auto"]',
                'div[data-pressable-container] span',
            ]
            
            for selector in text_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    texts = []
                    for el in elements[:5]:  # Limit to prevent noise
                        text = await el.text_content()
                        if text and len(text) > 10:
                            texts.append(text.strip())
                    if texts:
                        caption = "\n".join(texts[:3])  # Take first 3 text blocks
                        break
                except Exception:
                    continue
            
            # Try to extract images
            image_selectors = [
                'article img[src*="cdninstagram"]',
                'img[src*="threads"]',
                'article img',
            ]
            
            for selector in image_selectors:
                try:
                    images = await page.query_selector_all(selector)
                    for img in images:
                        src = await img.get_attribute("src")
                        if src and "profile" not in src.lower():
                            media_items.append(MediaItem(
                                media_type=MediaType.IMAGE,
                                url=src
                            ))
                    if media_items:
                        break
                except Exception:
                    continue
            
            # Try to extract videos
            video_selectors = [
                'article video source',
                'video source',
                'article video',
            ]
            
            for selector in video_selectors:
                try:
                    videos = await page.query_selector_all(selector)
                    for video in videos:
                        src = await video.get_attribute("src")
                        if src:
                            media_items.append(MediaItem(
                                media_type=MediaType.VIDEO,
                                url=src
                            ))
                except Exception:
                    continue
            
        except Exception as e:
            logger.warning(f"Error extracting Threads content: {e}")
        
        return FetchResult(
            platform=self.platform,
            url=url,
            shortcode=shortcode,
            caption=self.clean_text(caption),
            pinned_comments=pinned_comments,
            media_items=media_items,
            author=username or "",
        )
    
    async def download_media(self, result: FetchResult) -> FetchResult:
        """
        Download all media items in the fetch result.
        
        Args:
            result: FetchResult with media URLs
            
        Returns:
            Updated FetchResult with local file paths
        """
        for i, item in enumerate(result.media_items):
            try:
                if item.media_type == MediaType.VIDEO:
                    local_path = await self.downloader.download(
                        item.url,
                        settings.videos_dir,
                        f"threads_{result.shortcode}_{i}.mp4"
                    )
                else:
                    local_path = await self.downloader.download(
                        item.url,
                        settings.images_dir,
                        f"threads_{result.shortcode}_{i}.jpg"
                    )
                
                item.local_path = local_path
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
        
        return result
    
    def __del__(self):
        """Cleanup browser on deletion."""
        if self._browser or self._playwright:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule cleanup as a task in the running loop
                    loop.create_task(self._close_browser())
                except RuntimeError:
                    # No running event loop — run cleanup synchronously
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(self._close_browser())
                    finally:
                        loop.close()
            except Exception:
                pass
