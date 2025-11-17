"""
Logging Utilities for Agent Framework
Provides custom formatters, loggers, and utilities for structured logging.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path
import threading
from contextlib import contextmanager

from Agent.config.logging_config import (
    DEFAULT_CONFIG,
    get_agent_log_file,
    get_session_dir,
    get_trace_summary_file,
    SESSION_ID
)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "session_id": SESSION_ID,
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


class TraceLogger:
    """Base trace logger with structured logging capabilities"""

    def __init__(self, name: str, extra_fields: Optional[Dict[str, Any]] = None):
        """
        Initialize trace logger

        Args:
            name: Logger name
            extra_fields: Extra fields to include in all log entries
        """
        self.logger = logging.getLogger(name)
        self.extra_fields = extra_fields or {}
        self._local = threading.local()

    def _add_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add extra fields to log data"""
        log_extra = self.extra_fields.copy()
        if extra:
            log_extra.update(extra)
        return {"extra_data": log_extra}

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(message, extra=self._add_extra(extra))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(message, extra=self._add_extra(extra))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(message, extra=self._add_extra(extra))

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, extra=self._add_extra(extra), exc_info=exc_info)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message"""
        self.logger.critical(message, extra=self._add_extra(extra), exc_info=exc_info)

    @contextmanager
    def context(self, **kwargs):
        """Add contextual information to all logs within this context"""
        old_fields = self.extra_fields.copy()
        self.extra_fields.update(kwargs)
        try:
            yield self
        finally:
            self.extra_fields = old_fields


class AgentLogger(TraceLogger):
    """Logger for agent activities"""

    def __init__(self, agent_id: str, agent_role: str):
        """
        Initialize agent logger

        Args:
            agent_id: Unique agent identifier
            agent_role: Agent role (creator, audience, brand, moderator)
        """
        super().__init__(
            f"agent.{agent_role}",
            extra_fields={
                "agent_id": agent_id,
                "agent_role": agent_role
            }
        )
        self.agent_id = agent_id
        self.agent_role = agent_role

        # Create agent-specific log file
        self.agent_log_file = get_agent_log_file(agent_id)
        self._setup_agent_file_handler()

    def _setup_agent_file_handler(self):
        """Setup file handler for this specific agent"""
        file_handler = logging.FileHandler(self.agent_log_file)
        file_handler.setFormatter(JsonFormatter())
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def log_action(self, action_type: str, action_data: Dict[str, Any], decision_context: Optional[Dict[str, Any]] = None):
        """Log agent action"""
        self.info(
            f"Agent action: {action_type}",
            extra={
                "event_type": "agent_action",
                "action_type": action_type,
                "action_data": action_data,
                "decision_context": decision_context
            }
        )

    def log_decision(self, decision_type: str, decision_data: Dict[str, Any], reasoning: Optional[str] = None):
        """Log agent decision"""
        self.info(
            f"Agent decision: {decision_type}",
            extra={
                "event_type": "agent_decision",
                "decision_type": decision_type,
                "decision_data": decision_data,
                "reasoning": reasoning
            }
        )

    def log_state_change(self, old_state: Dict[str, Any], new_state: Dict[str, Any], trigger: str):
        """Log agent state change"""
        self.info(
            f"Agent state change: {trigger}",
            extra={
                "event_type": "state_change",
                "old_state": old_state,
                "new_state": new_state,
                "trigger": trigger
            }
        )

    def log_learning_update(self, learning_stage: str, update_data: Dict[str, Any]):
        """Log learning update"""
        self.debug(
            f"Learning update: {learning_stage}",
            extra={
                "event_type": "learning_update",
                "learning_stage": learning_stage,
                "update_data": update_data
            }
        )

    def log_content_generation(self, content_type: str, content_id: str, metadata: Dict[str, Any]):
        """Log content generation"""
        self.info(
            f"Content generated: {content_type}",
            extra={
                "event_type": "content_generation",
                "content_type": content_type,
                "content_id": content_id,
                "metadata": metadata
            }
        )

    def log_interaction(self, interaction_type: str, target_id: str, interaction_data: Dict[str, Any]):
        """Log interaction with another agent or content"""
        self.info(
            f"Interaction: {interaction_type} with {target_id}",
            extra={
                "event_type": "interaction",
                "interaction_type": interaction_type,
                "target_id": target_id,
                "interaction_data": interaction_data
            }
        )


class LearningLogger(TraceLogger):
    """Logger for learning system activities"""

    def __init__(self, learning_type: str, agent_id: Optional[str] = None):
        """
        Initialize learning logger

        Args:
            learning_type: Type of learning (bandit, rl, evolution)
            agent_id: Associated agent ID (if applicable)
        """
        super().__init__(
            f"learning.{learning_type}",
            extra_fields={
                "learning_type": learning_type,
                "agent_id": agent_id
            }
        )
        self.learning_type = learning_type

    def log_training_step(self, step: int, metrics: Dict[str, float], model_state: Optional[Dict[str, Any]] = None):
        """Log training step"""
        self.debug(
            f"Training step {step}",
            extra={
                "event_type": "training_step",
                "step": step,
                "metrics": metrics,
                "model_state": model_state
            }
        )

    def log_reward(self, action_id: str, reward: float, cumulative_reward: float):
        """Log reward"""
        self.debug(
            f"Reward: {reward}",
            extra={
                "event_type": "reward",
                "action_id": action_id,
                "reward": reward,
                "cumulative_reward": cumulative_reward
            }
        )

    def log_policy_update(self, update_type: str, update_data: Dict[str, Any]):
        """Log policy update"""
        self.info(
            f"Policy update: {update_type}",
            extra={
                "event_type": "policy_update",
                "update_type": update_type,
                "update_data": update_data
            }
        )

    def log_exploration_exploitation(self, action_selected: str, exploration: bool, epsilon: Optional[float] = None):
        """Log exploration vs exploitation decision"""
        self.debug(
            f"Action selection: {'exploration' if exploration else 'exploitation'}",
            extra={
                "event_type": "exploration_exploitation",
                "action_selected": action_selected,
                "exploration": exploration,
                "epsilon": epsilon
            }
        )

    def log_convergence(self, converged: bool, convergence_metrics: Dict[str, float]):
        """Log convergence status"""
        self.info(
            f"Convergence: {converged}",
            extra={
                "event_type": "convergence",
                "converged": converged,
                "convergence_metrics": convergence_metrics
            }
        )


class InteractionLogger(TraceLogger):
    """Logger for agent interactions and social dynamics"""

    def __init__(self):
        super().__init__("interaction")

    def log_content_interaction(
        self,
        agent_id: str,
        content_id: str,
        interaction_type: str,
        interaction_metadata: Dict[str, Any]
    ):
        """Log content interaction"""
        self.info(
            f"Content interaction: {agent_id} {interaction_type} {content_id}",
            extra={
                "event_type": "content_interaction",
                "agent_id": agent_id,
                "content_id": content_id,
                "interaction_type": interaction_type,
                "metadata": interaction_metadata
            }
        )

    def log_social_connection(self, agent_id_1: str, agent_id_2: str, connection_type: str):
        """Log social connection formation"""
        self.info(
            f"Social connection: {agent_id_1} -> {agent_id_2} ({connection_type})",
            extra={
                "event_type": "social_connection",
                "agent_id_1": agent_id_1,
                "agent_id_2": agent_id_2,
                "connection_type": connection_type
            }
        )

    def log_viral_cascade(self, content_id: str, cascade_data: Dict[str, Any]):
        """Log viral cascade event"""
        self.info(
            f"Viral cascade: {content_id}",
            extra={
                "event_type": "viral_cascade",
                "content_id": content_id,
                "cascade_data": cascade_data
            }
        )


class PerformanceLogger(TraceLogger):
    """Logger for performance metrics"""

    def __init__(self):
        super().__init__("performance")
        self.metrics_buffer = []
        self.buffer_size = 100

    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Log performance metric"""
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {}
        }

        self.metrics_buffer.append(metric_data)

        if len(self.metrics_buffer) >= self.buffer_size:
            self._flush_metrics()

        self.info(
            f"Metric: {metric_name} = {value}",
            extra={"event_type": "metric", **metric_data}
        )

    def log_latency(self, operation: str, latency_ms: float, tags: Optional[Dict[str, str]] = None):
        """Log operation latency"""
        self.log_metric(f"latency.{operation}", latency_ms, tags)

    def log_throughput(self, operation: str, count: int, duration_s: float):
        """Log throughput metric"""
        throughput = count / duration_s if duration_s > 0 else 0
        self.log_metric(f"throughput.{operation}", throughput, {"count": str(count)})

    def _flush_metrics(self):
        """Flush metrics buffer to file"""
        if not self.metrics_buffer:
            return

        # Write to performance log
        self.debug("Metrics batch flushed", extra={"metrics_count": len(self.metrics_buffer)})
        self.metrics_buffer.clear()


class EventLogger(TraceLogger):
    """Logger for system-wide events"""

    def __init__(self):
        super().__init__("event")

    def log_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log system event"""
        self.info(
            f"System event: {event_type}",
            extra={
                "event_type": event_type,
                "event_data": event_data
            }
        )

    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation start"""
        self.info(
            "Simulation started",
            extra={
                "event_type": "simulation_start",
                "config": config
            }
        )

    def log_simulation_end(self, summary: Dict[str, Any]):
        """Log simulation end"""
        self.info(
            "Simulation ended",
            extra={
                "event_type": "simulation_end",
                "summary": summary
            }
        )

    def log_checkpoint(self, checkpoint_id: str, checkpoint_data: Dict[str, Any]):
        """Log checkpoint"""
        self.info(
            f"Checkpoint: {checkpoint_id}",
            extra={
                "event_type": "checkpoint",
                "checkpoint_id": checkpoint_id,
                "checkpoint_data": checkpoint_data
            }
        )


class TraceSummary:
    """Utility for generating trace summaries"""

    def __init__(self):
        self.summary_file = get_trace_summary_file()
        self.summary_data = {
            "session_id": SESSION_ID,
            "start_time": datetime.now().isoformat(),
            "agents": {},
            "events": [],
            "errors": [],
            "performance_summary": {}
        }
        # Save initial summary
        self._save()

    def add_agent(self, agent_id: str, agent_role: str, metadata: Dict[str, Any]):
        """Add agent to summary"""
        self.summary_data["agents"][agent_id] = {
            "role": agent_role,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        self._save()

    def add_event(self, event_type: str, event_data: Dict[str, Any]):
        """Add event to summary"""
        self.summary_data["events"].append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data
        })
        self._save()

    def add_error(self, error_type: str, error_message: str, stacktrace: str):
        """Add error to summary"""
        self.summary_data["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "stacktrace": stacktrace
        })
        self._save()

    def update_performance_summary(self, metrics: Dict[str, Any]):
        """Update performance summary"""
        self.summary_data["performance_summary"].update(metrics)
        self._save()

    def finalize(self):
        """Finalize summary"""
        self.summary_data["end_time"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Save summary to file"""
        with open(self.summary_file, 'w') as f:
            json.dump(self.summary_data, f, indent=2)


def setup_logging():
    """Setup logging configuration"""
    import logging.config
    logging.config.dictConfig(DEFAULT_CONFIG)


# Initialize logging on module import
setup_logging()

# Global trace summary
_trace_summary = TraceSummary()

def get_trace_summary() -> TraceSummary:
    """Get global trace summary"""
    return _trace_summary
