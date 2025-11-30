"""
Social Arena Agent System
AI agent framework for realistic social media behavior simulation
"""

__version__ = "1.0.0"

# Core agent classes
from Agent.base import Actor, ActorType, AIAgent, HumanUser

# Logging utilities
from Agent.utils.logging_utils import (
    AgentLogger,
    LearningLogger,
    InteractionLogger,
    PerformanceLogger,
    EventLogger,
    get_trace_summary
)

# Log analysis
from Agent.utils.log_analyzer import (
    LogAnalyzer,
    analyze_session,
    debug_issue
)

# Decorators
from Agent.utils.trace_decorators import (
    trace_agent_action,
    measure_performance,
    trace_context
)

# Configuration
from Agent.config import (
    TRACE_DIR,
    SESSION_ID,
    get_session_dir,
    get_agent_log_file
)

# Initialize logging on import
from Agent.utils.logging_utils import setup_logging
setup_logging()

__all__ = [
    # Version
    "__version__",
    
    # Core Classes
    "Actor",
    "ActorType",
    "AIAgent",
    "HumanUser",
    
    # Logging
    "AgentLogger",
    "LearningLogger",
    "InteractionLogger",
    "PerformanceLogger",
    "EventLogger",
    "get_trace_summary",
    
    # Analysis
    "LogAnalyzer",
    "analyze_session",
    "debug_issue",
    
    # Decorators
    "trace_agent_action",
    "measure_performance",
    "trace_context",
    
    # Configuration
    "TRACE_DIR",
    "SESSION_ID",
    "get_session_dir",
    "get_agent_log_file",
]
