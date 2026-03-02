"""
Configuration management for the application.
Loads settings from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://knowledge:knowledge_secret@localhost:5432/knowledge_base",
        alias="DATABASE_URL"
    )
    
    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="knowledge_base", alias="QDRANT_COLLECTION")
    
    # LLM - Gemini
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    
    # LLM - Ollama (fallback)
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3:8b", alias="OLLAMA_MODEL")
    
    # Instagram
    instagram_username: Optional[str] = Field(default=None, alias="INSTAGRAM_USERNAME")
    instagram_password: Optional[str] = Field(default=None, alias="INSTAGRAM_PASSWORD")
    
    # Processing
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    ocr_languages: str = Field(default="en,hi", alias="OCR_LANGUAGES")
    
    # Application
    debug: bool = Field(default=False, alias="DEBUG")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    
    # Embedding model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def ocr_language_list(self) -> list[str]:
        """Get OCR languages as a list."""
        return [lang.strip() for lang in self.ocr_languages.split(",")]
    
    @property
    def images_dir(self) -> Path:
        """Directory for downloaded images."""
        path = self.data_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def videos_dir(self) -> Path:
        """Directory for downloaded videos."""
        path = self.data_dir / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def audio_dir(self) -> Path:
        """Directory for extracted audio."""
        path = self.data_dir / "audio"
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
