"""
SQLAlchemy models for the knowledge base.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    String, 
    Text, 
    DateTime, 
    ForeignKey,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.dialects.postgresql import UUID

from ..fetchers.base import Platform, MediaType


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Post(Base):
    """
    Represents a processed social media post.
    Stores original content, extracted data, and LLM-generated knowledge.
    """
    __tablename__ = "posts"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Source information
    platform: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )
    url: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    shortcode: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    author: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Original content
    caption: Mapped[Optional[str]] = mapped_column(Text)
    pinned_comments: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    
    # Extracted content
    extracted_text: Mapped[Optional[str]] = mapped_column(Text)  # From OCR
    transcription: Mapped[Optional[str]] = mapped_column(Text)  # From audio
    
    # LLM-generated knowledge
    summary: Mapped[Optional[str]] = mapped_column(Text)
    key_points: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    topics: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    structured_knowledge: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    
    # Metadata
    original_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    media_items: Mapped[list["Media"]] = relationship(
        "Media",
        back_populates="post",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Post {self.shortcode} from {self.platform}>"
    
    @property
    def all_text(self) -> str:
        """Combine all text content for embedding."""
        parts = []
        
        if self.caption:
            parts.append(self.caption)
        if self.extracted_text:
            parts.append(self.extracted_text)
        if self.transcription:
            parts.append(self.transcription)
        if self.summary:
            parts.append(self.summary)
        
        return "\n\n".join(parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "platform": self.platform,
            "url": self.url,
            "shortcode": self.shortcode,
            "author": self.author,
            "caption": self.caption,
            "pinned_comments": self.pinned_comments,
            "extracted_text": self.extracted_text,
            "transcription": self.transcription,
            "summary": self.summary,
            "key_points": self.key_points,
            "topics": self.topics,
            "structured_knowledge": self.structured_knowledge,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "media_count": len(self.media_items) if self.media_items else 0,
        }


class Media(Base):
    """
    Represents a media item (image or video) from a post.
    """
    __tablename__ = "media"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Foreign key to post
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("posts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Media information
    media_type: Mapped[str] = mapped_column(String(20), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    local_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Extracted content
    extracted_text: Mapped[Optional[str]] = mapped_column(Text)  # OCR for images
    transcription: Mapped[Optional[str]] = mapped_column(Text)  # For videos
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    post: Mapped["Post"] = relationship("Post", back_populates="media_items")
    
    def __repr__(self) -> str:
        return f"<Media {self.media_type} for post {self.post_id}>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "post_id": str(self.post_id),
            "media_type": self.media_type,
            "url": self.url,
            "local_path": self.local_path,
            "extracted_text": self.extracted_text,
            "transcription": self.transcription,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
