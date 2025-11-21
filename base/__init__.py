"""
Base Agent Architecture
Core classes for all agent types
"""

from .base_agent import BaseAgent, AgentRole, LearningStage
from .agent_memory import AgentMemory, MemoryItem, Evidence
from .strategy_head import StrategyHead, StrategyDecision, ActionContext
from .content_head import ContentHead, GeneratedContent, ContentContext

__all__ = [
    'BaseAgent',
    'AgentRole',
    'LearningStage',
    'AgentMemory',
    'MemoryItem',
    'Evidence',
    'StrategyHead',
    'StrategyDecision',
    'ActionContext',
    'ContentHead',
    'GeneratedContent',
    'ContentContext',
]
