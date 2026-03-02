"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import settings
from .api.routes import router, vector_store, embedding_generator
from .storage.database import init_db, close_db

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Knowledge Extraction API...")
    
    # Initialize database
    await init_db()
    
    # Initialize vector store collection
    await vector_store.init_collection()
    
    # Create data directories
    settings.images_dir
    settings.videos_dir
    settings.audio_dir
    
    # Preload Whisper model for faster first transcription
    logger.info("Preloading Whisper model (this may take a moment)...")
    try:
        from .processors.transcription import load_whisper_model
        load_whisper_model()
        logger.info("Whisper model loaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to preload Whisper model: {e}")

    logger.info("Preloading OCR reader...")
    try:
        from .processors.ocr import load_ocr_reader
        load_ocr_reader()
        logger.info("OCR reader loaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to preload OCR reader: {e}")

    logger.info("Preloading embedding model...")
    try:
        # Access the model property to trigger lazy load at startup
        _ = embedding_generator.model
        logger.info("Embedding model loaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to preload embedding model: {e}")

    logger.info("API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="Social Media Knowledge Extraction API",
    description="""
    A personal tool to process Instagram and Threads posts, 
    extract knowledge using OCR and transcription, 
    and build a searchable knowledge base.
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
# NOTE: allow_origins=["*"] is incompatible with allow_credentials=True per the CORS spec.
# Credentials are disabled here; set explicit origins in .env for cookie/auth-header support.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["Knowledge Base"])

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    logger.warning("Static files directory not found, skipping...")


@app.get("/")
async def root():
    """Serve the web interface."""
    try:
        return FileResponse("static/index.html")
    except Exception:
        return {
            "message": "Social Media Knowledge Extraction API",
            "docs": "/docs",
            "health": "/api/health"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
