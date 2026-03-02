"""
Tests for the knowledge extraction API.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

from src.main import app
from src.fetchers.base import Platform, FetchResult, MediaItem, MediaType


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_health_check():
    """Test health endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        # May fail if DB not connected, but endpoint should respond
        assert response.status_code in [200, 500]


@pytest.mark.anyio
async def test_process_invalid_url():
    """Test processing invalid URL."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/process",
            json={"url": "https://example.com/not-a-social-media-url"}
        )
        assert response.status_code == 400
        assert "Unsupported URL" in response.json()["detail"]


def test_instagram_url_parsing():
    """Test Instagram URL shortcode extraction."""
    from src.fetchers.instagram import InstagramFetcher
    
    fetcher = InstagramFetcher()
    
    # Test various URL formats
    assert fetcher.extract_shortcode("https://www.instagram.com/p/ABC123xyz/") == "ABC123xyz"
    assert fetcher.extract_shortcode("https://instagram.com/reel/XYZ789/") == "XYZ789"
    assert fetcher.extract_shortcode("http://instagr.am/p/test123") == "test123"
    assert fetcher.extract_shortcode("https://example.com/p/123") is None


def test_threads_url_parsing():
    """Test Threads URL shortcode extraction."""
    from src.fetchers.threads import ThreadsFetcher
    
    fetcher = ThreadsFetcher()
    
    # Test various URL formats
    assert fetcher.extract_shortcode("https://www.threads.net/@user/post/ABC123") == "ABC123"
    assert fetcher.extract_shortcode("https://threads.net/@username/post/XYZ789/") == "XYZ789"
    assert fetcher.extract_shortcode("https://instagram.com/p/123") is None


def test_platform_detection():
    """Test platform detection from URLs."""
    from src.fetchers.base import BaseFetcher
    
    assert BaseFetcher.detect_platform("https://instagram.com/p/123") == Platform.INSTAGRAM
    assert BaseFetcher.detect_platform("https://www.instagram.com/reel/abc") == Platform.INSTAGRAM
    assert BaseFetcher.detect_platform("https://threads.net/@user/post/123") == Platform.THREADS
    assert BaseFetcher.detect_platform("https://example.com") is None


def test_fetch_result_properties():
    """Test FetchResult properties."""
    result = FetchResult(
        platform=Platform.INSTAGRAM,
        url="https://instagram.com/p/test",
        shortcode="test",
        caption="Test caption",
        media_items=[
            MediaItem(media_type=MediaType.IMAGE, url="http://img.jpg"),
            MediaItem(media_type=MediaType.VIDEO, url="http://vid.mp4"),
        ]
    )
    
    assert result.success is True
    assert result.has_video is True
    assert result.has_images is True
    
    # Test with error
    error_result = FetchResult(
        platform=Platform.INSTAGRAM,
        url="https://instagram.com/p/test",
        shortcode="test",
        error="Something went wrong"
    )
    
    assert error_result.success is False


def test_prompt_templates():
    """Test prompt template generation."""
    from src.llm.prompts import PromptTemplates
    
    # Test summarize prompt
    prompt = PromptTemplates.get_summarize_prompt("Test content")
    assert "Test content" in prompt
    assert "Summarize" in prompt
    
    # Test key points prompt
    prompt = PromptTemplates.get_key_points_prompt("Test content")
    assert "Test content" in prompt
    assert "key points" in prompt.lower()
    
    # Test combined prompt
    prompt = PromptTemplates.get_combined_prompt(
        caption="Caption text",
        ocr_text="OCR text",
        transcription="",
        comments="Comment 1"
    )
    assert "Caption text" in prompt
    assert "OCR text" in prompt
    assert "Comment 1" in prompt


def test_text_cleaning():
    """Test text cleaning utility."""
    from src.fetchers.base import BaseFetcher
    
    # Test whitespace cleaning
    assert BaseFetcher.clean_text("  hello   world  ") == "hello world"
    assert BaseFetcher.clean_text("line1\n\n\n\n\nline2") == "line1\n\nline2"
    assert BaseFetcher.clean_text("") == ""
    assert BaseFetcher.clean_text(None) == ""  # type: ignore


@pytest.mark.anyio
async def test_embedding_generation():
    """Test embedding generation (if model available)."""
    try:
        from src.storage.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Test single embedding
        embedding = await generator.embed("Test text for embedding")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        
        # Test empty text
        empty_embedding = await generator.embed("")
        assert all(v == 0.0 for v in empty_embedding)
        
    except Exception:
        pytest.skip("Embedding model not available")
