"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class ProcessRequest(BaseModel):
    """Request to process a social media URL."""
    url: str = Field(..., description="Social media post URL (Instagram or Threads)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.instagram.com/p/ABC123xyz/"
            }
        }


class MediaResponse(BaseModel):
    """Response for a media item."""
    id: UUID
    media_type: str
    extracted_text: Optional[str] = None
    transcription: Optional[str] = None


class ProcessResponse(BaseModel):
    """Response after processing a URL."""
    id: UUID
    status: str = Field(..., description="Processing status: success, error")
    message: str
    shortcode: Optional[str] = None
    platform: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "message": "Post processed and added to knowledge base",
                "shortcode": "ABC123xyz",
                "platform": "instagram"
            }
        }


class PostResponse(BaseModel):
    """Response for a single post."""
    id: UUID
    platform: str
    url: str
    shortcode: str
    author: Optional[str] = None
    caption: Optional[str] = None
    pinned_comments: list[str] = []
    extracted_text: Optional[str] = None
    transcription: Optional[str] = None
    summary: Optional[str] = None
    key_points: list[str] = []
    topics: list[str] = []
    structured_knowledge: Optional[dict] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    media_count: int = 0


class PostListResponse(BaseModel):
    """Response for list of posts."""
    total: int
    posts: list[PostResponse]


class SearchRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    topics: Optional[list[str]] = Field(default=None, description="Filter by topics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "productivity tips for developers",
                "limit": 10,
                "topics": ["Productivity", "Technology"]
            }
        }


class SearchResult(BaseModel):
    """A single search result."""
    id: UUID
    score: float = Field(..., description="Similarity score (0-1)")
    platform: str
    shortcode: str
    summary: Optional[str] = None
    key_points: list[str] = []
    topics: list[str] = []
    url: str


class SearchResponse(BaseModel):
    """Response for search results."""
    query: str
    total: int
    results: list[SearchResult]


class StatsResponse(BaseModel):
    """Response for system statistics."""
    total_posts: int
    posts_by_platform: dict[str, int]
    vector_store: dict
    topics_distribution: dict[str, int]


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    database: str
    vector_store: str
    version: str
