"""
Configuration module for Agent framework
"""

from Agent.config.logging_config import (
    TRACE_DIR,
    TRACE_SUBDIRS,
    SESSION_ID,
    get_session_dir,
    get_agent_log_file,
    get_trace_summary_file
)

__all__ = [
    "TRACE_DIR",
    "TRACE_SUBDIRS",
    "SESSION_ID",
    "get_session_dir",
    "get_agent_log_file",
    "get_trace_summary_file"
]
