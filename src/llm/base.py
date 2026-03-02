"""
Base LLM interface and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    tokens_used: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if the LLM call was successful."""
        return self.error is None and len(self.content) > 0


class BaseLLM(ABC):
    """Abstract base class for LLM integrations."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM is available and ready."""
        pass
    
    async def summarize(self, content: str, prompt_template: str) -> LLMResponse:
        """
        Summarize content using a template.
        
        Args:
            content: The content to summarize
            prompt_template: Template with {content} placeholder
            
        Returns:
            LLMResponse with the summary
        """
        prompt = prompt_template.format(content=content)
        return await self.generate(prompt)


class LLMFallbackChain:
    """Chain of LLMs with automatic fallback."""
    
    def __init__(self, llms: list[BaseLLM]):
        """
        Initialize the fallback chain.
        
        Args:
            llms: List of LLMs to try in order
        """
        self.llms = llms
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Try each LLM in order until one succeeds.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            LLMResponse from first successful LLM
        """
        errors = []
        
        for llm in self.llms:
            try:
                if not await llm.is_available():
                    logger.warning(f"{llm.model_name} is not available, trying next")
                    continue
                
                response = await llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                if response.success:
                    return response
                else:
                    errors.append(f"{llm.model_name}: {response.error}")
                    
            except Exception as e:
                errors.append(f"{llm.model_name}: {str(e)}")
                logger.error(f"LLM {llm.model_name} failed: {e}")
        
        # All LLMs failed
        return LLMResponse(
            content="",
            model="none",
            error=f"All LLMs failed: {'; '.join(errors)}"
        )
    
    async def get_available_llm(self) -> Optional[BaseLLM]:
        """Get the first available LLM."""
        for llm in self.llms:
            if await llm.is_available():
                return llm
        return None
