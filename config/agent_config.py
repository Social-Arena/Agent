"""
Agent Configuration Management
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GlobalAgentConfig:
    """Global configuration for all agents"""
    
    # Logging
    log_level: str = "INFO"
    trace_dir: Path = field(default_factory=lambda: Path("trace"))
    log_retention_days: int = 30
    
    # Learning
    default_learning_stage: str = "cold_start"
    enable_evolution: bool = True
    evolution_rate: float = 0.1
    
    # Performance
    max_actions_per_agent: int = 1000
    action_timeout_seconds: int = 30
    
    # Content
    default_content_model: str = "gpt-4"
    content_temperature: float = 0.7
    max_content_length: int = 280
    
    # Memory
    memory_retention_days: int = 30
    max_memory_items: int = 10000
    
    # Monitoring
    enable_performance_tracking: bool = True
    metrics_collection_interval: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "log_level": self.log_level,
            "trace_dir": str(self.trace_dir),
            "log_retention_days": self.log_retention_days,
            "default_learning_stage": self.default_learning_stage,
            "enable_evolution": self.enable_evolution,
            "evolution_rate": self.evolution_rate,
            "max_actions_per_agent": self.max_actions_per_agent,
            "action_timeout_seconds": self.action_timeout_seconds,
            "default_content_model": self.default_content_model,
            "content_temperature": self.content_temperature,
            "max_content_length": self.max_content_length,
            "memory_retention_days": self.memory_retention_days,
            "max_memory_items": self.max_memory_items,
            "enable_performance_tracking": self.enable_performance_tracking,
            "metrics_collection_interval": self.metrics_collection_interval
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GlobalAgentConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Default global configuration
DEFAULT_CONFIG = GlobalAgentConfig()


def get_default_config() -> GlobalAgentConfig:
    """Get default global configuration"""
    return DEFAULT_CONFIG


def load_config_from_file(config_path: Path) -> GlobalAgentConfig:
    """Load configuration from file"""
    import json
    
    if not config_path.exists():
        return DEFAULT_CONFIG
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return GlobalAgentConfig.from_dict(config_dict)


def save_config_to_file(config: GlobalAgentConfig, config_path: Path) -> None:
    """Save configuration to file"""
    import json
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

