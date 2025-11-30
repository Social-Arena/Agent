"""
Agent Module
"""

from .agent import Agent, RecommendationSystem
from .host import (
    LanguageModelHost,
    BackendConfig,
    BackendProvider,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    create_qwen_host,
    create_openai_host,
    create_anthropic_host
)
from .__main__ import main as cli_main

__all__ = [
    "Agent",
    "RecommendationSystem",
    "LanguageModelHost",
    "BackendConfig",
    "BackendProvider",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "create_qwen_host",
    "create_openai_host",
    "create_anthropic_host",
    "cli_main"
]

