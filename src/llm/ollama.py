"""
Ollama local LLM integration (fallback).
"""

import asyncio
from typing import Optional
import logging

import ollama as ollama_client

from ..config import settings
from .base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama local LLM integration for offline fallback."""
    
    def __init__(
        self, 
        host: Optional[str] = None, 
        model: Optional[str] = None
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            host: Ollama server URL (uses settings if not provided)
            model: Model name (uses settings if not provided)
        """
        self.host = host or settings.ollama_host
        self._model_name = model or settings.ollama_model
        
        # Configure client
        self.client = ollama_client.Client(host=self.host)
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    async def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            # Check if server is running and model exists
            models = await asyncio.get_running_loop().run_in_executor(
                None, self.client.list
            )
            
            available_models = [m.model for m in models.models]
            
            # Check if our model is available (handle tags like :latest)
            model_base = self._model_name.split(':')[0]
            for m in available_models:
                if m.startswith(model_base):
                    logger.info(f"Ollama model {self._model_name} is available")
                    return True
            
            logger.warning(
                f"Ollama model {self._model_name} not found. "
                f"Available: {available_models}"
            )
            return False
            
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate response using Ollama.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            
        Returns:
            LLMResponse with generated content
        """
        try:
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            # Generate in thread pool (Ollama client is synchronous)
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.chat(
                    model=self._model_name,
                    messages=messages,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
            )
            
            content = response.message.content
            
            # Get token usage
            tokens_used = (
                (response.prompt_eval_count or 0) +
                (response.eval_count or 0)
            )
            
            return LLMResponse(
                content=content,
                model=self._model_name,
                tokens_used=tokens_used,
                metadata={
                    'total_duration': getattr(response, 'total_duration', None),
                    'eval_duration': getattr(response, 'eval_duration', None),
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return LLMResponse(
                content="",
                model=self._model_name,
                error=str(e)
            )
    
    async def pull_model(self) -> bool:
        """
        Pull the model if not available.
        
        Returns:
            True if model was pulled successfully
        """
        try:
            logger.info(f"Pulling Ollama model: {self._model_name}")
            
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.pull(self._model_name)
            )
            
            logger.info(f"Successfully pulled {self._model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
