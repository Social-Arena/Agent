"""
Log Analyzer Utilities
Provides tools for analyzing trace logs and debugging issues.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict, Counter

from Agent.config.logging_config import TRACE_DIR, SESSION_ID, get_session_dir


class LogEntry:
    """Represents a single log entry"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.timestamp = datetime.fromisoformat(data.get("timestamp", ""))
        self.level = data.get("level", "")
        self.logger = data.get("logger", "")
        self.message = data.get("message", "")
        self.extra = data.get("extra", {})
        self.exception = data.get("exception")

    def __repr__(self):
        return f"<LogEntry {self.timestamp} {self.level} {self.logger}: {self.message}>"


class LogAnalyzer:
    """Analyzer for trace logs"""

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize log analyzer

        Args:
            session_id: Session ID to analyze (defaults to current session)
        """
        self.session_id = session_id or SESSION_ID
        self.session_dir = TRACE_DIR / f"session_{self.session_id}"

        if not self.session_dir.exists():
            raise ValueError(f"Session directory not found: {self.session_dir}")

        self.logs: List[LogEntry] = []
        self._load_logs()

    def _load_logs(self):
        """Load all logs from session directory"""
        # Load from all log files in session directory
        for log_file in self.session_dir.rglob("*.log"):
            self._load_log_file(log_file)

        # Sort by timestamp
        self.logs.sort(key=lambda x: x.timestamp)

    def _load_log_file(self, log_file: Path):
        """Load logs from a single file"""
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Try to parse as JSON
                        data = json.loads(line)
                        self.logs.append(LogEntry(data))
                    except json.JSONDecodeError:
                        # Plain text log entry - skip or parse differently
                        pass
        except Exception as e:
            print(f"Error loading {log_file}: {e}")

    def filter_by_agent(self, agent_id: str) -> List[LogEntry]:
        """Filter logs by agent ID"""
        return [
            log for log in self.logs
            if log.extra.get("agent_id") == agent_id
        ]

    def filter_by_level(self, level: str) -> List[LogEntry]:
        """Filter logs by level"""
        return [log for log in self.logs if log.level == level.upper()]

    def filter_by_logger(self, logger_name: str) -> List[LogEntry]:
        """Filter logs by logger name"""
        return [log for log in self.logs if logger_name in log.logger]

    def filter_by_event_type(self, event_type: str) -> List[LogEntry]:
        """Filter logs by event type"""
        return [
            log for log in self.logs
            if log.extra.get("event_type") == event_type
        ]

    def filter_by_time_range(self, start_time: datetime, end_time: datetime) -> List[LogEntry]:
        """Filter logs by time range"""
        return [
            log for log in self.logs
            if start_time <= log.timestamp <= end_time
        ]

    def get_errors(self) -> List[LogEntry]:
        """Get all error and critical logs"""
        return [log for log in self.logs if log.level in ["ERROR", "CRITICAL"]]

    def get_exceptions(self) -> List[LogEntry]:
        """Get all logs with exceptions"""
        return [log for log in self.logs if log.exception is not None]

    def analyze_agent_activity(self, agent_id: str) -> Dict[str, Any]:
        """Analyze activity for a specific agent"""
        agent_logs = self.filter_by_agent(agent_id)

        if not agent_logs:
            return {"agent_id": agent_id, "error": "No logs found for this agent"}

        # Count events by type
        event_counts = Counter()
        action_counts = Counter()
        decision_counts = Counter()

        for log in agent_logs:
            event_type = log.extra.get("event_type")
            if event_type:
                event_counts[event_type] += 1

            if event_type == "agent_action":
                action_type = log.extra.get("action_type")
                if action_type:
                    action_counts[action_type] += 1

            if event_type == "agent_decision":
                decision_type = log.extra.get("decision_type")
                if decision_type:
                    decision_counts[decision_type] += 1

        return {
            "agent_id": agent_id,
            "total_logs": len(agent_logs),
            "first_activity": agent_logs[0].timestamp.isoformat(),
            "last_activity": agent_logs[-1].timestamp.isoformat(),
            "event_counts": dict(event_counts),
            "action_counts": dict(action_counts),
            "decision_counts": dict(decision_counts),
            "errors": len([log for log in agent_logs if log.level == "ERROR"])
        }

    def analyze_learning_progress(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze learning system progress"""
        learning_logs = self.filter_by_logger("learning")

        if agent_id:
            learning_logs = [log for log in learning_logs if log.extra.get("agent_id") == agent_id]

        # Extract training steps
        training_steps = []
        rewards = []
        policy_updates = []

        for log in learning_logs:
            event_type = log.extra.get("event_type")

            if event_type == "training_step":
                training_steps.append({
                    "step": log.extra.get("step"),
                    "timestamp": log.timestamp.isoformat(),
                    "metrics": log.extra.get("metrics", {})
                })

            if event_type == "reward":
                rewards.append({
                    "timestamp": log.timestamp.isoformat(),
                    "reward": log.extra.get("reward"),
                    "cumulative_reward": log.extra.get("cumulative_reward")
                })

            if event_type == "policy_update":
                policy_updates.append({
                    "timestamp": log.timestamp.isoformat(),
                    "update_type": log.extra.get("update_type")
                })

        return {
            "agent_id": agent_id,
            "total_training_steps": len(training_steps),
            "total_rewards": len(rewards),
            "total_policy_updates": len(policy_updates),
            "training_steps": training_steps,
            "rewards": rewards,
            "policy_updates": policy_updates
        }

    def analyze_interactions(self) -> Dict[str, Any]:
        """Analyze interaction patterns"""
        interaction_logs = self.filter_by_logger("interaction")

        # Count interactions by type
        interaction_counts = Counter()
        content_interactions = defaultdict(int)
        social_connections = []

        for log in interaction_logs:
            event_type = log.extra.get("event_type")
            if event_type:
                interaction_counts[event_type] += 1

            if event_type == "content_interaction":
                content_id = log.extra.get("content_id")
                if content_id:
                    content_interactions[content_id] += 1

            if event_type == "social_connection":
                social_connections.append({
                    "agent_1": log.extra.get("agent_id_1"),
                    "agent_2": log.extra.get("agent_id_2"),
                    "connection_type": log.extra.get("connection_type"),
                    "timestamp": log.timestamp.isoformat()
                })

        # Find viral content
        viral_content = sorted(
            content_interactions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_interactions": len(interaction_logs),
            "interaction_counts": dict(interaction_counts),
            "total_social_connections": len(social_connections),
            "social_connections": social_connections,
            "top_viral_content": [
                {"content_id": content_id, "interaction_count": count}
                for content_id, count in viral_content
            ]
        }

    def find_error_patterns(self) -> Dict[str, Any]:
        """Find patterns in errors"""
        error_logs = self.get_errors()

        if not error_logs:
            return {"total_errors": 0, "message": "No errors found"}

        # Count errors by logger
        errors_by_logger = Counter()
        errors_by_message = Counter()
        exception_types = Counter()

        for log in error_logs:
            errors_by_logger[log.logger] += 1
            errors_by_message[log.message] += 1

            if log.exception:
                exception_type = log.exception.get("type")
                if exception_type:
                    exception_types[exception_type] += 1

        return {
            "total_errors": len(error_logs),
            "first_error": error_logs[0].timestamp.isoformat() if error_logs else None,
            "last_error": error_logs[-1].timestamp.isoformat() if error_logs else None,
            "errors_by_logger": dict(errors_by_logger),
            "most_common_messages": errors_by_message.most_common(10),
            "exception_types": dict(exception_types),
            "recent_errors": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "logger": log.logger,
                    "message": log.message,
                    "exception": log.exception
                }
                for log in error_logs[-10:]
            ]
        }

    def trace_agent_journey(self, agent_id: str) -> List[Dict[str, Any]]:
        """Trace the complete journey of an agent"""
        agent_logs = self.filter_by_agent(agent_id)

        journey = []
        for log in agent_logs:
            journey.append({
                "timestamp": log.timestamp.isoformat(),
                "event_type": log.extra.get("event_type"),
                "message": log.message,
                "level": log.level,
                "extra": log.extra
            })

        return journey

    def find_causality_chain(self, target_event: LogEntry, lookback_seconds: int = 60) -> List[LogEntry]:
        """
        Find events that may have caused a target event

        Args:
            target_event: The event to trace back from
            lookback_seconds: How many seconds to look back

        Returns:
            List of potentially related events
        """
        # Get time window
        end_time = target_event.timestamp
        start_time = end_time - timedelta(seconds=lookback_seconds)

        # Get all events in time window
        window_logs = self.filter_by_time_range(start_time, end_time)

        # Filter to relevant loggers or agents
        relevant_logs = []
        target_agent_id = target_event.extra.get("agent_id")

        for log in window_logs:
            # Include if same agent or related system component
            if log.extra.get("agent_id") == target_agent_id:
                relevant_logs.append(log)
            elif log.logger.startswith("learning") and target_event.logger.startswith("agent"):
                relevant_logs.append(log)

        return relevant_logs

    def generate_debug_report(self, issue_description: str) -> str:
        """
        Generate a debug report for a specific issue

        Args:
            issue_description: Description of the issue

        Returns:
            Formatted debug report
        """
        report = [
            "=" * 80,
            "DEBUG REPORT",
            "=" * 80,
            f"Session ID: {self.session_id}",
            f"Issue: {issue_description}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "=" * 80,
            "ERROR SUMMARY",
            "=" * 80,
        ]

        # Add error patterns
        error_patterns = self.find_error_patterns()
        report.append(json.dumps(error_patterns, indent=2))

        report.extend([
            "",
            "=" * 80,
            "RECENT ERRORS",
            "=" * 80,
        ])

        # Add recent errors with context
        recent_errors = self.get_errors()[-5:]
        for error in recent_errors:
            report.append(f"\nTimestamp: {error.timestamp.isoformat()}")
            report.append(f"Logger: {error.logger}")
            report.append(f"Message: {error.message}")

            if error.exception:
                report.append("Exception:")
                report.append(json.dumps(error.exception, indent=2))

            # Add causality chain
            causality = self.find_causality_chain(error, lookback_seconds=30)
            if causality:
                report.append("\nPotential causes:")
                for cause in causality[-5:]:
                    report.append(f"  - {cause.timestamp.isoformat()} [{cause.level}] {cause.message}")

        return "\n".join(report)

    def export_analysis(self, output_file: Path):
        """Export complete analysis to file"""
        analysis = {
            "session_id": self.session_id,
            "generated_at": datetime.now().isoformat(),
            "total_logs": len(self.logs),
            "error_patterns": self.find_error_patterns(),
            "interaction_analysis": self.analyze_interactions(),
        }

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)


from datetime import timedelta


def analyze_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to analyze a session

    Args:
        session_id: Session ID to analyze (defaults to current)

    Returns:
        Analysis results
    """
    analyzer = LogAnalyzer(session_id)

    return {
        "session_id": analyzer.session_id,
        "total_logs": len(analyzer.logs),
        "errors": analyzer.find_error_patterns(),
        "interactions": analyzer.analyze_interactions(),
    }


def debug_issue(issue_description: str, session_id: Optional[str] = None) -> str:
    """
    Generate debug report for an issue

    Args:
        issue_description: Description of the issue
        session_id: Session ID (defaults to current)

    Returns:
        Debug report
    """
    analyzer = LogAnalyzer(session_id)
    return analyzer.generate_debug_report(issue_description)
