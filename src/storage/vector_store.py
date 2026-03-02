"""
Qdrant vector store integration for semantic search.
"""

import asyncio
import uuid
from typing import Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from ..config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant vector store for semantic search over knowledge base.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.dimension = settings.embedding_dimension
        
        self.client = QdrantClient(host=self.host, port=self.port)
    
    async def init_collection(self) -> bool:
        """
        Initialize the collection if it doesn't exist.
        
        Returns:
            True if collection was created or already exists
        """
        try:
            loop = asyncio.get_running_loop()
            # Check if collection exists
            collections = await loop.run_in_executor(None, self.client.get_collections)
            exists = any(
                c.name == self.collection_name 
                for c in collections.collections
            )
            
            if not exists:
                # Create collection
                await loop.run_in_executor(
                    None,
                    lambda: self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.dimension,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            return False
    
    async def add_document(
        self,
        post_id: uuid.UUID,
        embedding: list[float],
        metadata: dict,
    ) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            post_id: Unique ID for the document
            embedding: Vector embedding
            metadata: Document metadata (title, summary, topics, etc.)
            
        Returns:
            True if successful
        """
        try:
            point = qdrant_models.PointStruct(
                id=str(post_id),
                vector=embedding,
                payload=metadata
            )
            
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
            )
            
            logger.info(f"Added document {post_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            return False
    
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_topics: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filter_topics: Optional topic filter
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Build filter if needed
            query_filter = None
            if filter_topics:
                query_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="topics",
                            match=qdrant_models.MatchAny(any=filter_topics)
                        )
                    ]
                )
            
            results = (await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
            )).points
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    **result.payload
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete_document(self, post_id: uuid.UUID) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            post_id: ID of document to delete
            
        Returns:
            True if successful
        """
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_models.PointIdsList(
                        points=[str(post_id)]
                    )
                )
            )
            
            logger.info(f"Deleted document {post_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def update_metadata(
        self,
        post_id: uuid.UUID,
        metadata: dict,
    ) -> bool:
        """
        Update metadata for a document.
        
        Args:
            post_id: Document ID
            metadata: New metadata
            
        Returns:
            True if successful
        """
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=metadata,
                    points=[str(post_id)]
                )
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False
    
    async def get_stats(self) -> dict:
        """Get collection statistics."""
        try:
            info = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.get_collection(self.collection_name)
            )
            return {
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
