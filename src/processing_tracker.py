"""
Processing status tracking for real-time progress updates.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Stages of content processing."""
    QUEUED = "queued"
    FETCHING = "fetching"
    DOWNLOADING_MEDIA = "downloading_media"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    OCR_PROCESSING = "ocr_processing"
    LLM_SUMMARIZING = "llm_summarizing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingStatus:
    """Status of a processing job."""
    post_id: UUID
    stage: ProcessingStage
    url: str = ""
    progress: int = 0  # 0-100
    message: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    logs: list = field(default_factory=list)
    cancelled: bool = False
    
    def add_log(self, message: str):
        """Add a log entry with timestamp."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        # Keep only last 50 logs
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
    
    def to_dict(self) -> dict:
        return {
            "post_id": str(self.post_id),
            "url": self.url,
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error,
            "logs": self.logs,
            "cancelled": self.cancelled,
        }


class ProcessingTracker:
    """
    Tracks processing status for all jobs.
    Provides real-time progress updates.
    """
    
    def __init__(self):
        self._jobs: dict[UUID, ProcessingStatus] = {}
        self._lock = asyncio.Lock()
    
    async def start_job(self, post_id: UUID, url: str = "") -> ProcessingStatus:
        """Start tracking a new processing job."""
        async with self._lock:
            status = ProcessingStatus(
                post_id=post_id,
                url=url,
                stage=ProcessingStage.QUEUED,
                message="Job queued for processing"
            )
            status.add_log("Job started")
            self._jobs[post_id] = status
            logger.info(f"[{post_id}] Processing started")
            return status
    
    async def update(
        self,
        post_id: UUID,
        stage: ProcessingStage,
        progress: int = 0,
        message: str = ""
    ) -> Optional[ProcessingStatus]:
        """Update processing status."""
        async with self._lock:
            if post_id not in self._jobs:
                return None
            
            status = self._jobs[post_id]
            
            # Check if cancelled
            if status.cancelled:
                return None
            
            status.stage = stage
            status.progress = progress
            status.message = message
            status.updated_at = datetime.utcnow()
            status.add_log(f"{stage.value}: {message}")
            
            logger.info(f"[{post_id}] {stage.value}: {message} ({progress}%)")
            return status
    
    async def is_cancelled(self, post_id: UUID) -> bool:
        """Check if a job has been cancelled."""
        async with self._lock:
            if post_id in self._jobs:
                return self._jobs[post_id].cancelled
            return False
    
    async def cancel(self, post_id: UUID) -> Optional[ProcessingStatus]:
        """Cancel a processing job."""
        async with self._lock:
            if post_id not in self._jobs:
                return None
            
            status = self._jobs[post_id]
            status.cancelled = True
            status.stage = ProcessingStage.CANCELLED
            status.message = "Job cancelled by user"
            status.updated_at = datetime.utcnow()
            status.add_log("Job cancelled by user")
            
            logger.info(f"[{post_id}] Cancelled by user")
            return status
    
    async def fail(self, post_id: UUID, error: str) -> Optional[ProcessingStatus]:
        """Mark job as failed."""
        async with self._lock:
            if post_id not in self._jobs:
                return None
            
            status = self._jobs[post_id]
            status.stage = ProcessingStage.FAILED
            status.error = error
            status.updated_at = datetime.utcnow()
            status.add_log(f"FAILED: {error}")
            
            logger.error(f"[{post_id}] Failed: {error}")
            return status
    
    async def complete(self, post_id: UUID) -> Optional[ProcessingStatus]:
        """Mark job as completed."""
        async with self._lock:
            if post_id not in self._jobs:
                return None
            
            status = self._jobs[post_id]
            status.stage = ProcessingStage.COMPLETED
            status.progress = 100
            status.message = "Processing completed successfully"
            status.updated_at = datetime.utcnow()
            status.add_log("Completed successfully")
            
            logger.info(f"[{post_id}] Completed successfully")
            return status
    
    async def get_status(self, post_id: UUID) -> Optional[ProcessingStatus]:
        """Get current status of a job."""
        async with self._lock:
            return self._jobs.get(post_id)
    
    async def get_all_jobs(self) -> list[ProcessingStatus]:
        """Get all jobs (including completed/failed)."""
        async with self._lock:
            return list(self._jobs.values())
    
    async def get_all_active(self) -> list[ProcessingStatus]:
        """Get all active (non-completed/failed) jobs."""
        async with self._lock:
            return [
                s for s in self._jobs.values()
                if s.stage not in (ProcessingStage.COMPLETED, ProcessingStage.FAILED, ProcessingStage.CANCELLED)
            ]
    
    async def get_logs(self, post_id: UUID) -> list[str]:
        """Get logs for a job."""
        async with self._lock:
            if post_id in self._jobs:
                return self._jobs[post_id].logs.copy()
            return []
    
    async def cleanup_old(self, max_age_hours: int = 24):
        """Remove old completed/failed jobs."""
        async with self._lock:
            now = datetime.utcnow()
            to_remove = []
            
            for post_id, status in self._jobs.items():
                if status.stage in (ProcessingStage.COMPLETED, ProcessingStage.FAILED, ProcessingStage.CANCELLED):
                    age = (now - status.updated_at).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(post_id)
            
            for post_id in to_remove:
                del self._jobs[post_id]


# Global tracker instance
processing_tracker = ProcessingTracker()

