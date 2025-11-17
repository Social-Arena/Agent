"""
Logging Configuration for Agent Framework
Provides centralized logging configuration with file-based output only (no console).
"""

import os
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Base trace directory
TRACE_DIR = Path(__file__).parent.parent.parent / "trace"

# Ensure trace directories exist
TRACE_SUBDIRS = {
    "agents": TRACE_DIR / "agents",
    "learning": TRACE_DIR / "learning",
    "interactions": TRACE_DIR / "interactions",
    "performance": TRACE_DIR / "performance",
    "system": TRACE_DIR / "system",
    "errors": TRACE_DIR / "errors",
    "events": TRACE_DIR / "events"
}

for subdir in TRACE_SUBDIRS.values():
    subdir.mkdir(parents=True, exist_ok=True)

# Log levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Default logging configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "detailed": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S.%f"
        },
        "performance": {
            "format": "%(asctime)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S.%f"
        }
    },

    "handlers": {
        # System-wide logs
        "system_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["system"] / "system.log"),
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": "INFO"
        },

        # Error logs
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["errors"] / "errors.log"),
            "maxBytes": 50 * 1024 * 1024,  # 50MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": "ERROR"
        },

        # Agent activity logs
        "agent_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["agents"] / "agents.log"),
            "maxBytes": 200 * 1024 * 1024,  # 200MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": "DEBUG"
        },

        # Learning system logs
        "learning_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["learning"] / "learning.log"),
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": "DEBUG"
        },

        # Interaction logs
        "interaction_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["interactions"] / "interactions.log"),
            "maxBytes": 200 * 1024 * 1024,  # 200MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": "INFO"
        },

        # Performance metrics
        "performance_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["performance"] / "performance.log"),
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 10,
            "formatter": "performance",
            "level": "INFO"
        },

        # Event logs
        "event_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(TRACE_SUBDIRS["events"] / "events.log"),
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": "INFO"
        }
    },

    "loggers": {
        # Root logger - catches everything
        "": {
            "handlers": ["system_file", "error_file"],
            "level": "INFO",
            "propagate": False
        },

        # Agent loggers
        "agent": {
            "handlers": ["agent_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "agent.creator": {
            "handlers": ["agent_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "agent.audience": {
            "handlers": ["agent_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "agent.brand": {
            "handlers": ["agent_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "agent.moderator": {
            "handlers": ["agent_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },

        # Learning loggers
        "learning": {
            "handlers": ["learning_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "learning.bandit": {
            "handlers": ["learning_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "learning.rl": {
            "handlers": ["learning_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "learning.evolution": {
            "handlers": ["learning_file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        },

        # Interaction logger
        "interaction": {
            "handlers": ["interaction_file", "error_file"],
            "level": "INFO",
            "propagate": False
        },

        # Performance logger
        "performance": {
            "handlers": ["performance_file"],
            "level": "INFO",
            "propagate": False
        },

        # Event logger
        "event": {
            "handlers": ["event_file", "error_file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Session tracking
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = TRACE_DIR / f"session_{SESSION_ID}"

def get_session_dir() -> Path:
    """Get the current session directory"""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR

def get_agent_log_file(agent_id: str) -> Path:
    """Get the log file path for a specific agent"""
    session_dir = get_session_dir()
    agent_dir = session_dir / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir / f"{agent_id}.log"

def get_trace_summary_file() -> Path:
    """Get the trace summary file for the current session"""
    session_dir = get_session_dir()
    return session_dir / "trace_summary.json"
