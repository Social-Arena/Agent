"""
Unit tests for the logging system
"""

import asyncio
import json
from pathlib import Path
import pytest

from Agent.utils import (
    AgentLogger,
    LearningLogger,
    InteractionLogger,
    PerformanceLogger,
    EventLogger,
    LogAnalyzer,
    get_trace_summary
)


class TestAgentLogger:
    """Test AgentLogger functionality"""

    def test_logger_initialization(self):
        """Test logger can be initialized"""
        logger = AgentLogger(agent_id="test_001", agent_role="creator")

        assert logger.agent_id == "test_001"
        assert logger.agent_role == "creator"

    def test_log_action(self):
        """Test logging actions"""
        logger = AgentLogger(agent_id="test_002", agent_role="creator")

        # Should not raise exception
        logger.log_action(
            action_type="create_content",
            action_data={"content_type": "video"},
            decision_context={"trend": "AI"}
        )

    def test_log_decision(self):
        """Test logging decisions"""
        logger = AgentLogger(agent_id="test_003", agent_role="creator")

        logger.log_decision(
            decision_type="content_timing",
            decision_data={"time": "18:00"},
            reasoning="Peak activity"
        )

    def test_log_state_change(self):
        """Test logging state changes"""
        logger = AgentLogger(agent_id="test_004", agent_role="creator")

        logger.log_state_change(
            old_state={"followers": 100},
            new_state={"followers": 200},
            trigger="viral_content"
        )

    def test_log_content_generation(self):
        """Test logging content generation"""
        logger = AgentLogger(agent_id="test_005", agent_role="creator")

        logger.log_content_generation(
            content_type="video",
            content_id="content_123",
            metadata={"duration": 30}
        )


class TestLearningLogger:
    """Test LearningLogger functionality"""

    def test_logger_initialization(self):
        """Test logger can be initialized"""
        logger = LearningLogger(learning_type="bandit", agent_id="test_001")

        assert logger.learning_type == "bandit"

    def test_log_training_step(self):
        """Test logging training steps"""
        logger = LearningLogger(learning_type="rl", agent_id="test_001")

        logger.log_training_step(
            step=100,
            metrics={"loss": 0.5, "reward": 1.0},
            model_state={"lr": 0.001}
        )

    def test_log_reward(self):
        """Test logging rewards"""
        logger = LearningLogger(learning_type="bandit", agent_id="test_001")

        logger.log_reward(
            action_id="action_001",
            reward=1.5,
            cumulative_reward=10.0
        )

    def test_log_exploration_exploitation(self):
        """Test logging exploration vs exploitation"""
        logger = LearningLogger(learning_type="bandit", agent_id="test_001")

        logger.log_exploration_exploitation(
            action_selected="create_video",
            exploration=True,
            epsilon=0.1
        )


class TestInteractionLogger:
    """Test InteractionLogger functionality"""

    def test_logger_initialization(self):
        """Test logger can be initialized"""
        logger = InteractionLogger()
        assert logger is not None

    def test_log_content_interaction(self):
        """Test logging content interactions"""
        logger = InteractionLogger()

        logger.log_content_interaction(
            agent_id="agent_001",
            content_id="content_001",
            interaction_type="like",
            interaction_metadata={"timestamp": "now"}
        )

    def test_log_social_connection(self):
        """Test logging social connections"""
        logger = InteractionLogger()

        logger.log_social_connection(
            agent_id_1="agent_001",
            agent_id_2="agent_002",
            connection_type="follow"
        )

    def test_log_viral_cascade(self):
        """Test logging viral cascades"""
        logger = InteractionLogger()

        logger.log_viral_cascade(
            content_id="content_001",
            cascade_data={"shares": 1000, "reach": 10000}
        )


class TestPerformanceLogger:
    """Test PerformanceLogger functionality"""

    def test_logger_initialization(self):
        """Test logger can be initialized"""
        logger = PerformanceLogger()
        assert logger is not None

    def test_log_metric(self):
        """Test logging metrics"""
        logger = PerformanceLogger()

        logger.log_metric(
            metric_name="test_metric",
            value=100.0,
            tags={"env": "test"}
        )

    def test_log_latency(self):
        """Test logging latency"""
        logger = PerformanceLogger()

        logger.log_latency(
            operation="test_operation",
            latency_ms=50.5,
            tags={"service": "test"}
        )

    def test_log_throughput(self):
        """Test logging throughput"""
        logger = PerformanceLogger()

        logger.log_throughput(
            operation="test_operation",
            count=100,
            duration_s=10.0
        )


class TestEventLogger:
    """Test EventLogger functionality"""

    def test_logger_initialization(self):
        """Test logger can be initialized"""
        logger = EventLogger()
        assert logger is not None

    def test_log_system_event(self):
        """Test logging system events"""
        logger = EventLogger()

        logger.log_system_event(
            event_type="test_event",
            event_data={"key": "value"}
        )

    def test_log_simulation_start(self):
        """Test logging simulation start"""
        logger = EventLogger()

        logger.log_simulation_start(
            config={"num_agents": 100}
        )

    def test_log_checkpoint(self):
        """Test logging checkpoints"""
        logger = EventLogger()

        logger.log_checkpoint(
            checkpoint_id="cp_001",
            checkpoint_data={"step": 1000}
        )


class TestTraceSummary:
    """Test TraceSummary functionality"""

    def test_trace_summary_singleton(self):
        """Test trace summary is accessible"""
        summary = get_trace_summary()
        assert summary is not None
        assert hasattr(summary, 'summary_data')

    def test_add_agent(self):
        """Test adding agent to summary"""
        summary = get_trace_summary()

        summary.add_agent(
            agent_id="test_agent",
            agent_role="creator",
            metadata={"test": True}
        )

        assert "test_agent" in summary.summary_data["agents"]

    def test_add_event(self):
        """Test adding event to summary"""
        summary = get_trace_summary()

        summary.add_event(
            event_type="test_event",
            event_data={"data": "test"}
        )

        # Should have at least one event
        assert len(summary.summary_data["events"]) > 0


class TestLogAnalyzer:
    """Test LogAnalyzer functionality"""

    @pytest.fixture
    def setup_test_logs(self):
        """Setup test logs"""
        # Create some test logs
        logger = AgentLogger(agent_id="analyzer_test", agent_role="creator")
        logger.log_action(
            action_type="test_action",
            action_data={"test": True},
            decision_context=None
        )
        logger.info("Test log message")

    def test_analyzer_initialization(self, setup_test_logs):
        """Test analyzer can be initialized"""
        analyzer = LogAnalyzer()
        assert analyzer is not None
        assert analyzer.session_id is not None

    def test_filter_by_agent(self, setup_test_logs):
        """Test filtering logs by agent"""
        analyzer = LogAnalyzer()

        agent_logs = analyzer.filter_by_agent("analyzer_test")
        # Should have at least some logs
        assert isinstance(agent_logs, list)

    def test_filter_by_level(self, setup_test_logs):
        """Test filtering logs by level"""
        analyzer = LogAnalyzer()

        info_logs = analyzer.filter_by_level("INFO")
        assert isinstance(info_logs, list)

    def test_analyze_agent_activity(self, setup_test_logs):
        """Test analyzing agent activity"""
        analyzer = LogAnalyzer()

        analysis = analyzer.analyze_agent_activity("analyzer_test")
        assert "agent_id" in analysis


@pytest.mark.asyncio
class TestAsyncLogging:
    """Test async logging functionality"""

    async def test_async_context_logging(self):
        """Test logging in async context"""
        logger = AgentLogger(agent_id="async_test", agent_role="creator")

        logger.info("Async test started")
        await asyncio.sleep(0.01)
        logger.info("Async test completed")

        # Should complete without errors


def test_log_files_created():
    """Test that log files are being created"""
    from Agent.config import TRACE_SUBDIRS

    # Check that trace directories exist
    assert TRACE_SUBDIRS["agents"].exists()
    assert TRACE_SUBDIRS["learning"].exists()
    assert TRACE_SUBDIRS["interactions"].exists()
    assert TRACE_SUBDIRS["performance"].exists()
    assert TRACE_SUBDIRS["system"].exists()
    assert TRACE_SUBDIRS["errors"].exists()
    assert TRACE_SUBDIRS["events"].exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
