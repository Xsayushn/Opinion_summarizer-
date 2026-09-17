"""
Generation module for Dialectical Prompt Construction and Multi-provider LLM clients.
"""

from .prompt_builder import PromptBuilder
from .llm_client import LLMClient

__all__ = ["PromptBuilder", "LLMClient"]
