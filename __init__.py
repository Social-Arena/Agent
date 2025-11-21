"""
Agent Framework
Social media viral propagation agent simulation system.
"""

__version__ = "0.2.0"

# Import base components
from Agent.base.base_agent import (
    BaseAgent,
    AgentRole,
    LearningStage,
    AgentConfig,
    EnvironmentState,
    AgentAction,
    ActionFeedback,
    PerformanceMetrics,
    AgentState
)

from Agent.base.agent_memory import (
    AgentMemory,
    MemoryItem,
    Evidence,
    ContentMetrics
)

from Agent.base.strategy_head import (
    StrategyHead,
    StrategyConfig,
    StrategyDecision,
    ActionContext,
    ActionType
)

from Agent.base.content_head import (
    ContentHead,
    ContentConfig,
    GeneratedContent,
    ContentContext,
    ContentStyle
)

# Import agent roles
from Agent.roles.creator_agent import CreatorAgent, CreatorConfig
from Agent.roles.audience_agent import AudienceAgent, AudienceConfig
from Agent.roles.brand_agent import BrandAgent, BrandConfig
from Agent.roles.moderator_agent import ModeratorAgent, ModeratorConfig

# Initialize logging on module import
from Agent.utils.logging_utils import setup_logging

setup_logging()

__all__ = [
    # Version
    '__version__',
    
    # Base Classes
    'BaseAgent',
    'AgentRole',
    'LearningStage',
    'AgentConfig',
    'EnvironmentState',
    'AgentAction',
    'ActionFeedback',
    'PerformanceMetrics',
    'AgentState',
    
    # Memory
    'AgentMemory',
    'MemoryItem',
    'Evidence',
    'ContentMetrics',
    
    # Strategy
    'StrategyHead',
    'StrategyConfig',
    'StrategyDecision',
    'ActionContext',
    'ActionType',
    
    # Content
    'ContentHead',
    'ContentConfig',
    'GeneratedContent',
    'ContentContext',
    'ContentStyle',
    
    # Agent Roles
    'CreatorAgent',
    'CreatorConfig',
    'AudienceAgent',
    'AudienceConfig',
    'BrandAgent',
    'BrandConfig',
    'ModeratorAgent',
    'ModeratorConfig',
]
