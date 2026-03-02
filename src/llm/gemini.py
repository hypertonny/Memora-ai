"""
Google Gemini Pro integration.
"""

import asyncio
from typing import Optional
import logging

import google.generativeai as genai

from ..config import settings
from .base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """Google Gemini Pro LLM integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Gemini API key (uses settings if not provided)
            model: Model name (uses settings if not provided)
        """
        self.api_key = api_key or settings.gemini_api_key
        self._model_name = model or settings.gemini_model
        self._model = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    @property
    def model(self):
        """Lazy load the Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(self._model_name)
        return self._model
    
    async def is_available(self) -> bool:
        """Check if Gemini API is available (key-only check, no API call)."""
        if not self.api_key:
            logger.warning("Gemini API key not configured")
            return False
        return True
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate response using Gemini Pro.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (prepended to prompt)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            
        Returns:
            LLMResponse with generated content
        """
        if not self.api_key:
            return LLMResponse(
                content="",
                model=self._model_name,
                error="Gemini API key not configured"
            )
        
        try:
            # Combine system prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Generate in thread pool (SDK is synchronous)
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            )
            
            # Extract response text
            content = ""
            if response.parts:
                content = response.parts[0].text
            elif hasattr(response, 'text'):
                content = response.text
            
            # Get token usage if available
            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
            
            return LLMResponse(
                content=content,
                model=self._model_name,
                tokens_used=tokens_used,
                metadata={'candidate_count': len(response.candidates) if response.candidates else 0}
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return LLMResponse(
                content="",
                model=self._model_name,
                error=str(e)
            )
