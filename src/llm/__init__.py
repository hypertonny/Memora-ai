"""LLM integration for summarization and knowledge extraction."""

from .base import BaseLLM, LLMResponse
from .gemini import GeminiLLM
from .ollama import OllamaLLM
from .prompts import PromptTemplates

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "GeminiLLM",
    "OllamaLLM",
    "PromptTemplates",
]
