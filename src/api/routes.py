"""
FastAPI routes for the knowledge extraction API.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from ..storage.database import get_db
from ..storage.models import Post, Media
from ..storage.vector_store import VectorStore
from ..storage.embeddings import EmbeddingGenerator
from ..processors.ocr import OCRProcessor
from ..processors.transcription import TranscriptionProcessor
from ..processors.video import VideoProcessor
from ..processing_tracker import processing_tracker, ProcessingStage
from .schemas import (
    ProcessRequest,
    ProcessResponse,
    PostResponse,
    PostListResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatsResponse,
    HealthResponse,
)
from .. import __version__

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize shared resources
vector_store = VectorStore()
embedding_generator = EmbeddingGenerator()

# Shared processor singletons — created once at module load to avoid
# per-request re-initialization cost (EasyOCR reader load takes ~30 s).
_ocr_processor = OCRProcessor()
_transcription_processor = TranscriptionProcessor()
_video_processor = VideoProcessor()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    db_status = "healthy"
    vector_status = "healthy"
    
    # Check database
    try:
        async with get_db() as session:
            await session.execute(select(func.count(Post.id)))
    except Exception as e:
        db_status = f"unhealthy: {e}"
    
    # Check vector store
    try:
        stats = await vector_store.get_stats()
        if stats and "points_count" in stats:
            vector_status = "healthy"
        else:
            vector_status = "unhealthy: no stats"
    except Exception as e:
        vector_status = f"unhealthy: {e}"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" and vector_status == "healthy" else "degraded",
        database=db_status,
        vector_store=vector_status,
        version=__version__,
    )


@router.post("/process", response_model=ProcessResponse)
async def process_url(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a social media URL and extract knowledge.
    
    This endpoint accepts an Instagram or Threads URL, fetches the content,
    processes it (OCR, transcription), and stores the extracted knowledge.
    """
    from ..fetchers.base import BaseFetcher
    
    url = request.url.strip()
    
    # Detect platform
    platform = BaseFetcher.detect_platform(url)
    if not platform:
        raise HTTPException(
            status_code=400,
            detail="Unsupported URL. Please provide an Instagram or Threads URL."
        )
    
    # Check if already processed
    async with get_db() as session:
        existing = await session.execute(
            select(Post).where(Post.url == url)
        )
        existing_post = existing.scalar_one_or_none()
        
        if existing_post:
            return ProcessResponse(
                id=existing_post.id,
                status="exists",
                message="This URL has already been processed",
                shortcode=existing_post.shortcode,
                platform=existing_post.platform,
            )
    
    # Create initial post record
    from ..fetchers.base import Platform
    
    async with get_db() as session:
        # Extract shortcode based on platform
        if platform == Platform.INSTAGRAM:
            from ..fetchers.instagram import InstagramFetcher
            fetcher = InstagramFetcher()
        elif platform == Platform.YOUTUBE:
            from ..fetchers.youtube import YouTubeFetcher
            fetcher = YouTubeFetcher()
        else:
            from ..fetchers.threads import ThreadsFetcher
            fetcher = ThreadsFetcher()
        
        shortcode = fetcher.extract_shortcode(url) or "unknown"
        
        post = Post(
            platform=platform.value,
            url=url,
            shortcode=shortcode,
        )
        session.add(post)
        await session.flush()
        post_id = post.id
    
    # Start tracking with URL
    await processing_tracker.start_job(post_id, url=url)
    
    # Queue background processing
    background_tasks.add_task(process_post_content, post_id, url, platform.value)
    
    return ProcessResponse(
        id=post_id,
        status="processing",
        message="Content is being processed. Check /api/jobs for progress.",
        shortcode=shortcode,
        platform=platform.value,
    )


@router.get("/jobs")
async def list_jobs():
    """Get all processing jobs with their current status."""
    jobs = await processing_tracker.get_all_jobs()
    return {
        "total": len(jobs),
        "jobs": [j.to_dict() for j in jobs]
    }


@router.get("/status/{post_id}")
async def get_processing_status(post_id: UUID):
    """Get the current processing status of a post."""
    status = await processing_tracker.get_status(post_id)
    
    if not status:
        # Check if post exists and is completed
        async with get_db() as session:
            post = await session.get(Post, post_id)
            if post and post.processed_at:
                return {
                    "post_id": str(post_id),
                    "stage": "completed",
                    "progress": 100,
                    "message": "Processing completed successfully",
                    "logs": [],
                    "cancelled": False
                }
            elif post:
                return {
                    "post_id": str(post_id),
                    "stage": "unknown",
                    "progress": 0,
                    "message": "Status not available",
                    "logs": [],
                    "cancelled": False
                }
        raise HTTPException(status_code=404, detail="Post not found")
    
    return status.to_dict()


@router.post("/cancel/{post_id}")
async def cancel_job(post_id: UUID):
    """Cancel a processing job."""
    status = await processing_tracker.cancel(post_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    
    return {
        "status": "cancelled",
        "post_id": str(post_id),
        "message": "Job has been cancelled"
    }


@router.get("/logs/{post_id}")
async def get_job_logs(post_id: UUID):
    """Get logs for a processing job."""
    logs = await processing_tracker.get_logs(post_id)
    
    if not logs:
        # Check if job exists
        status = await processing_tracker.get_status(post_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "post_id": str(post_id),
        "logs": logs
    }


async def process_post_content(post_id: UUID, url: str, platform: str):
    """Background task to process post content with progress tracking."""
    from ..fetchers.instagram import InstagramFetcher
    from ..fetchers.threads import ThreadsFetcher
    from ..llm.gemini import GeminiLLM
    from ..llm.ollama import OllamaLLM
    from ..llm.base import LLMFallbackChain
    from ..llm.prompts import PromptTemplates
    from ..fetchers.base import MediaType
    import json
    
    try:
        # === STAGE 1: Fetching ===
        await processing_tracker.update(
            post_id, ProcessingStage.FETCHING, 5, 
            f"Fetching content from {platform}..."
        )
        
        if platform == "instagram":
            fetcher = InstagramFetcher()
        elif platform == "youtube":
            from ..fetchers.youtube import YouTubeFetcher
            fetcher = YouTubeFetcher()
        else:
            fetcher = ThreadsFetcher()
        
        result = await fetcher.fetch(url)
        
        if not result.success:
            await processing_tracker.fail(post_id, f"Failed to fetch: {result.error}")
            return
        
        if await processing_tracker.is_cancelled(post_id):
            return
        
        await processing_tracker.update(
            post_id, ProcessingStage.FETCHING, 15,
            f"Found {len(result.media_items)} media items"
        )
        
        # === STAGE 2: Downloading Media ===
        await processing_tracker.update(
            post_id, ProcessingStage.DOWNLOADING_MEDIA, 20,
            "Downloading media files..."
        )
        
        result = await fetcher.download_media(result)
        
        await processing_tracker.update(
            post_id, ProcessingStage.DOWNLOADING_MEDIA, 30,
            "Media downloaded successfully"
        )
        
        if await processing_tracker.is_cancelled(post_id):
            return
        
        # Use shared processor singletons (avoids costly re-initialization per request)
        ocr = _ocr_processor
        transcriber = _transcription_processor
        video_processor = _video_processor
        
        all_ocr_text = []
        all_transcriptions = []
        
        total_items = len(result.media_items)
        processed_items = 0
        
        for item in result.media_items:
            if await processing_tracker.is_cancelled(post_id):
                return
            
            if item.local_path is None:
                processed_items += 1
                continue
            
            if item.media_type == MediaType.IMAGE:
                # === STAGE 3a: OCR ===
                await processing_tracker.update(
                    post_id, ProcessingStage.OCR_PROCESSING, 
                    30 + int(20 * processed_items / max(total_items, 1)),
                    f"OCR processing image {processed_items + 1}/{total_items}"
                )
                
                text = await ocr.extract_text(item.local_path)
                if text:
                    all_ocr_text.append(text)
                    item.extracted_text = text
            
            elif item.media_type == MediaType.VIDEO:
                # === STAGE 3b: Extract Audio ===
                await processing_tracker.update(
                    post_id, ProcessingStage.EXTRACTING_AUDIO,
                    35 + int(10 * processed_items / max(total_items, 1)),
                    f"Extracting audio from video {processed_items + 1}/{total_items}"
                )
                
                audio_path = await video_processor.extract_audio(item.local_path)
                
                if audio_path:
                    # === STAGE 3c: Transcribe ===
                    await processing_tracker.update(
                        post_id, ProcessingStage.TRANSCRIBING,
                        45 + int(15 * processed_items / max(total_items, 1)),
                        f"Transcribing audio {processed_items + 1}/{total_items} (this may take a moment)..."
                    )
                    
                    transcription = await transcriber.transcribe(audio_path)
                    if transcription.get('text'):
                        all_transcriptions.append(transcription['text'])
                        item.extracted_text = transcription['text']
                        
                        await processing_tracker.update(
                            post_id, ProcessingStage.TRANSCRIBING,
                            55 + int(5 * processed_items / max(total_items, 1)),
                            f"Transcribed {len(transcription['text'])} chars ({transcription.get('language', 'unknown')})"
                        )
            
            processed_items += 1
        
        if await processing_tracker.is_cancelled(post_id):
            return
        
        # === STAGE 4: LLM Summarization ===
        await processing_tracker.update(
            post_id, ProcessingStage.LLM_SUMMARIZING, 65,
            "Generating summary with LLM..."
        )
        
        # Set up LLM chain with fallback
        # Set up LLM chain with fallback to handle quota limits
        llm_chain = LLMFallbackChain([
            # Try primary configured model first (e.g., gemini-2.0-flash)
            GeminiLLM(),
            # Fallback to other variants if primary fails (e.g., due to quota)
            GeminiLLM(model="gemini-2.5-flash"),
            GeminiLLM(model="gemini-2.5-pro"),
            GeminiLLM(model="gemini-2.0-flash-lite-preview-02-05"), # Lightweight fallback
            
            # Finally fallback to local LLM
            OllamaLLM(),
        ])
        
        combined_text = PromptTemplates.get_combined_prompt(
            caption=result.caption,
            ocr_text="\n".join(all_ocr_text),
            transcription="\n".join(all_transcriptions),
            comments="\n".join(result.pinned_comments),
        )
        
        response = await llm_chain.generate(
            prompt=PromptTemplates.get_structure_prompt(combined_text),
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        await processing_tracker.update(
            post_id, ProcessingStage.LLM_SUMMARIZING, 80,
            f"LLM processing complete (model: {response.model})"
        )
        
        # Parse structured response
        structured_knowledge = {}
        summary = ""
        key_points = []
        topics = []
        
        if response.success:
            try:
                content = response.content
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    structured_knowledge = json.loads(content[start:end])
                    summary = structured_knowledge.get('summary', '')
                    key_points = structured_knowledge.get('key_points', [])
                    topics = structured_knowledge.get('topics', [])
            except json.JSONDecodeError:
                summary = response.content
        
        # === STAGE 5: Storing ===
        await processing_tracker.update(
            post_id, ProcessingStage.STORING, 85,
            "Saving to database..."
        )
        
        async with get_db() as session:
            post = await session.get(Post, post_id)
            if post:
                post.caption = result.caption
                post.pinned_comments = result.pinned_comments
                post.author = result.author
                post.extracted_text = "\n".join(all_ocr_text) if all_ocr_text else None
                post.transcription = "\n".join(all_transcriptions) if all_transcriptions else None
                post.summary = summary
                post.key_points = key_points
                post.topics = topics
                post.structured_knowledge = structured_knowledge
                post.processed_at = datetime.utcnow()
                if result.timestamp:
                    try:
                        post.original_timestamp = datetime.fromisoformat(result.timestamp)
                    except (ValueError, TypeError):
                        pass
                
                for item in result.media_items:
                    is_video = item.media_type == MediaType.VIDEO
                    media = Media(
                        post_id=post_id,
                        media_type=item.media_type.value,
                        url=item.url,
                        local_path=str(item.local_path) if item.local_path else None,
                        extracted_text=item.extracted_text if not is_video else None,
                        transcription=item.extracted_text if is_video else None,
                    )
                    session.add(media)
                # get_db() auto-commits on context exit; no explicit commit needed
        
        await processing_tracker.update(
            post_id, ProcessingStage.STORING, 95,
            "Adding to vector store..."
        )
        
        # Add to vector store
        embedding = await embedding_generator.embed(
            f"{summary}\n{' '.join(key_points)}\n{result.caption}"
        )
        await vector_store.add_document(
            post_id=post_id,
            embedding=embedding,
            metadata={
                "platform": platform,
                "shortcode": result.shortcode,
                "summary": summary,
                "topics": topics,
                "key_points": key_points,
                "url": url,
            }
        )
        
        # === COMPLETE ===
        await processing_tracker.complete(post_id)
        
    except Exception as e:
        logger.error(f"Error processing post {post_id}: {e}")
        await processing_tracker.fail(post_id, str(e))
        raise


@router.get("/posts", response_model=PostListResponse)
async def list_posts(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    platform: Optional[str] = Query(default=None),
):
    """List all processed posts."""
    async with get_db() as session:
        query = select(Post).options(selectinload(Post.media_items)).order_by(Post.created_at.desc())
        
        if platform:
            query = query.where(Post.platform == platform)
        
        count_query = select(func.count(Post.id))
        if platform:
            count_query = count_query.where(Post.platform == platform)
        
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        query = query.offset(offset).limit(limit)
        result = await session.execute(query)
        posts = result.scalars().all()
        
        # Build response inside the session context
        post_responses = []
        for p in posts:
            post_responses.append(PostResponse(
                id=p.id,
                platform=p.platform,
                url=p.url,
                shortcode=p.shortcode,
                author=p.author,
                caption=p.caption,
                pinned_comments=p.pinned_comments or [],
                extracted_text=p.extracted_text,
                transcription=p.transcription,
                summary=p.summary,
                key_points=p.key_points or [],
                topics=p.topics or [],
                structured_knowledge=p.structured_knowledge,
                created_at=p.created_at,
                processed_at=p.processed_at,
                media_count=len(p.media_items) if p.media_items else 0,
            ))
        
        return PostListResponse(total=total, posts=post_responses)


@router.get("/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: UUID):
    """Get a specific post by ID."""
    async with get_db() as session:
        result = await session.execute(
            select(Post)
            .options(selectinload(Post.media_items))
            .where(Post.id == post_id)
        )
        post = result.scalar_one_or_none()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        return PostResponse(
            id=post.id,
            platform=post.platform,
            url=post.url,
            shortcode=post.shortcode,
            author=post.author,
            caption=post.caption,
            pinned_comments=post.pinned_comments or [],
            extracted_text=post.extracted_text,
            transcription=post.transcription,
            summary=post.summary,
            key_points=post.key_points or [],
            topics=post.topics or [],
            structured_knowledge=post.structured_knowledge,
            created_at=post.created_at,
            processed_at=post.processed_at,
            media_count=len(post.media_items) if post.media_items else 0,
        )


@router.delete("/posts/{post_id}")
async def delete_post(post_id: UUID):
    """Delete a post from the knowledge base."""
    async with get_db() as session:
        post = await session.get(Post, post_id)
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        await session.delete(post)
        await session.commit()
    
    await vector_store.delete_document(post_id)
    
    return {"status": "deleted", "id": str(post_id)}


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """Semantic search across the knowledge base."""
    query_embedding = await embedding_generator.embed(request.query)
    
    results = await vector_store.search(
        query_embedding=query_embedding,
        limit=request.limit,
        filter_topics=request.topics,
        score_threshold=0.2, # Lower threshold to improved recall
    )
    
    return SearchResponse(
        query=request.query,
        total=len(results),
        results=[
            SearchResult(
                id=r["id"],
                score=r["score"],
                platform=r.get("platform", ""),
                shortcode=r.get("shortcode", ""),
                summary=r.get("summary"),
                key_points=r.get("key_points", []),
                topics=r.get("topics", []),
                url=r.get("url", ""),
            )
            for r in results
        ]
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    async with get_db() as session:
        total_result = await session.execute(select(func.count(Post.id)))
        total_posts = total_result.scalar()
        
        platform_result = await session.execute(
            select(Post.platform, func.count(Post.id))
            .group_by(Post.platform)
        )
        posts_by_platform = dict(platform_result.all())
        
        posts_result = await session.execute(select(Post.topics))
        all_topics = {}
        for (topics,) in posts_result.all():
            if topics:
                for topic in topics:
                    all_topics[topic] = all_topics.get(topic, 0) + 1
    
    vector_stats = await vector_store.get_stats()
    
    return StatsResponse(
        total_posts=total_posts,
        posts_by_platform=posts_by_platform,
        vector_store=vector_stats,
        topics_distribution=all_topics,
    )
