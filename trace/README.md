# Agent Framework Trace System

## Overview

The trace system provides comprehensive logging and debugging capabilities for the Agent framework. All runtime logs are stored in files (no console output) for efficient debugging and analysis.

## Directory Structure

```
trace/
├── agents/           # Agent-specific activity logs
├── learning/         # Learning system logs (bandit, RL, evolution)
├── interactions/     # Agent-to-agent and agent-to-content interactions
├── performance/      # Performance metrics and latency measurements
├── system/           # System-wide logs
├── errors/           # Error and exception logs
├── events/           # High-level system events
└── session_YYYYMMDD_HHMMSS/  # Session-specific logs
    ├── agents/       # Individual agent log files
    │   ├── agent_001.log
    │   ├── agent_002.log
    │   └── ...
    └── trace_summary.json  # Session summary
```

## Log Types

### 1. Agent Logs (`agents/`)
- Agent actions and decisions
- State changes
- Content generation
- Strategy updates
- Individual agent trace files in session directories

**Format**: Structured JSON

**Example**:
```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "session_id": "20250115_103045",
  "level": "INFO",
  "logger": "agent.creator",
  "message": "Agent action: create_content",
  "extra": {
    "agent_id": "creator_001",
    "agent_role": "creator",
    "event_type": "agent_action",
    "action_type": "create_content",
    "action_data": {...}
  }
}
```

### 2. Learning Logs (`learning/`)
- Training steps and iterations
- Reward signals
- Policy updates
- Exploration vs exploitation decisions
- Convergence metrics

**Format**: Structured JSON

### 3. Interaction Logs (`interactions/`)
- Content interactions (likes, shares, comments)
- Social connections between agents
- Viral cascades
- Collaboration events

**Format**: Structured JSON

### 4. Performance Logs (`performance/`)
- Operation latencies
- Throughput metrics
- Resource utilization
- Bottleneck identification

**Format**: Timestamped metrics

### 5. System Logs (`system/`)
- Framework initialization
- Configuration changes
- Module loading
- General system events

**Format**: Detailed text with timestamps

### 6. Error Logs (`errors/`)
- All ERROR and CRITICAL level messages
- Exception traces
- Error context
- Debugging information

**Format**: Detailed text with stack traces

### 7. Event Logs (`events/`)
- Simulation start/end
- Checkpoints
- Phase transitions
- High-level milestones

**Format**: Structured JSON

## Usage

### Basic Logging

```python
from Agent.utils import AgentLogger, LearningLogger, PerformanceLogger

# Create an agent logger
agent_logger = AgentLogger(agent_id="creator_001", agent_role="creator")

# Log an action
agent_logger.log_action(
    action_type="create_content",
    action_data={"content_type": "video", "duration": 30},
    decision_context={"trending_topic": "AI"}
)

# Log a decision
agent_logger.log_decision(
    decision_type="content_timing",
    decision_data={"scheduled_time": "2025-01-15 18:00"},
    reasoning="Peak audience activity time"
)

# Log a learning update
agent_logger.log_learning_update(
    learning_stage="bandit",
    update_data={"epsilon": 0.1, "best_arm": "video_content"}
)
```

### Learning System Logging

```python
learning_logger = LearningLogger(learning_type="bandit", agent_id="creator_001")

# Log training step
learning_logger.log_training_step(
    step=100,
    metrics={"loss": 0.25, "accuracy": 0.85},
    model_state={"learning_rate": 0.001}
)

# Log reward
learning_logger.log_reward(
    action_id="action_123",
    reward=1.5,
    cumulative_reward=150.0
)

# Log exploration/exploitation
learning_logger.log_exploration_exploitation(
    action_selected="create_video",
    exploration=True,
    epsilon=0.1
)
```

### Performance Tracking

```python
from Agent.utils import measure_performance, trace_performance

# Using context manager
with measure_performance("content_generation"):
    content = await generate_content()

# Using decorator
@trace_performance("model_inference")
async def run_inference(model, input_data):
    return await model.predict(input_data)
```

### Using Decorators

```python
from Agent.utils import trace_agent_action, trace_learning_update, trace_errors

class CreatorAgent:
    @trace_agent_action(action_type="create_content")
    async def create_content(self, context):
        # Action is automatically logged
        content = await self._generate_content(context)
        return content

    @trace_learning_update(update_type="bandit_update")
    async def update_strategy(self, feedback):
        # Learning update is automatically logged
        await self.bandit_learner.update(feedback)

    @trace_errors()
    async def risky_operation(self):
        # Errors are automatically logged with full context
        result = await self._complex_operation()
        return result
```

### Context-Based Logging

```python
from Agent.utils import trace_context

with trace_context(agent_logger, "viral_optimization", content_id="content_123"):
    # All logs within this context include content_id
    optimized_content = optimize_for_virality(content)

    # Automatically logs entry/exit and elapsed time
```

## Log Analysis

### Using LogAnalyzer

```python
from Agent.utils import LogAnalyzer

# Analyze current session
analyzer = LogAnalyzer()

# Get agent activity summary
agent_summary = analyzer.analyze_agent_activity("creator_001")
print(f"Total logs: {agent_summary['total_logs']}")
print(f"Actions: {agent_summary['action_counts']}")
print(f"Errors: {agent_summary['errors']}")

# Analyze learning progress
learning_progress = analyzer.analyze_learning_progress("creator_001")
print(f"Training steps: {learning_progress['total_training_steps']}")
print(f"Total rewards: {learning_progress['total_rewards']}")

# Find error patterns
error_patterns = analyzer.find_error_patterns()
print(f"Total errors: {error_patterns['total_errors']}")
print(f"Most common: {error_patterns['most_common_messages']}")

# Analyze interactions
interactions = analyzer.analyze_interactions()
print(f"Total interactions: {interactions['total_interactions']}")
print(f"Viral content: {interactions['top_viral_content']}")

# Trace agent journey
journey = analyzer.trace_agent_journey("creator_001")
for event in journey:
    print(f"{event['timestamp']}: {event['message']}")
```

### Quick Analysis Functions

```python
from Agent.utils import analyze_session, debug_issue

# Quick session analysis
analysis = analyze_session()
print(analysis)

# Generate debug report for an issue
report = debug_issue("Agent creator_001 not generating content")
print(report)
```

### Export Analysis

```python
from pathlib import Path

analyzer = LogAnalyzer()

# Export complete analysis
analyzer.export_analysis(Path("trace/analysis_report.json"))
```

## Debugging Workflow

When debugging an issue:

1. **Identify the issue**: Note the error message, agent ID, and approximate time
2. **Generate debug report**:
   ```python
   report = debug_issue("Description of the issue")
   print(report)
   ```
3. **Analyze error patterns**:
   ```python
   analyzer = LogAnalyzer()
   errors = analyzer.find_error_patterns()
   ```
4. **Trace causality**:
   ```python
   recent_errors = analyzer.get_errors()[-1]
   causes = analyzer.find_causality_chain(recent_errors, lookback_seconds=60)
   ```
5. **Examine agent journey**:
   ```python
   journey = analyzer.trace_agent_journey("problematic_agent_id")
   ```

## Best Practices

### 1. Use Appropriate Log Levels
- **DEBUG**: Detailed diagnostic information (training steps, exploration decisions)
- **INFO**: General informational messages (actions, decisions, interactions)
- **WARNING**: Warning messages (deprecated features, potential issues)
- **ERROR**: Error messages (exceptions, failures)
- **CRITICAL**: Critical issues (system failures)

### 2. Include Context
Always include relevant context in logs:
```python
agent_logger.log_action(
    action_type="create_content",
    action_data={
        "content_type": "video",
        "target_audience": "tech_enthusiasts",
        "estimated_reach": 10000
    },
    decision_context={
        "trending_topics": ["AI", "ML"],
        "audience_activity": "high",
        "time_of_day": "evening"
    }
)
```

### 3. Use Structured Data
Log structured data (dicts) rather than strings for better analysis:
```python
# Good
agent_logger.info("Content generated", extra={
    "content_id": "c123",
    "content_type": "video",
    "duration": 30
})

# Avoid
agent_logger.info(f"Generated video content c123 with duration 30")
```

### 4. Log State Transitions
Always log when agent state changes:
```python
agent_logger.log_state_change(
    old_state={"learning_stage": "bandit", "follower_count": 100},
    new_state={"learning_stage": "reinforcement", "follower_count": 500},
    trigger="threshold_reached"
)
```

### 5. Use Decorators for Automatic Tracing
Leverage decorators to ensure consistent logging:
```python
@trace_agent_action()
async def perform_action(self, context):
    # Action automatically logged
    pass
```

## Configuration

### Log Rotation
Logs are automatically rotated when they reach size limits:
- Agent logs: 200MB per file, 10 backups
- Learning logs: 100MB per file, 10 backups
- Error logs: 50MB per file, 5 backups

### Session Management
Each session gets a unique ID based on start timestamp:
- Format: `YYYYMMDD_HHMMSS`
- Session directory: `trace/session_{SESSION_ID}/`
- Individual agent logs stored in session directory

### Custom Configuration
Modify `Agent/config/logging_config.py` to customize:
- Log file sizes and rotation
- Log levels
- Formatters
- Additional handlers

## Performance Considerations

### Buffer Flushing
Performance metrics are buffered (default: 100 entries) before flushing to disk to reduce I/O overhead.

### Async Logging
Use async logging for high-throughput scenarios to avoid blocking.

### Log Level Filtering
Set appropriate log levels in production to reduce log volume:
```python
# In production, consider INFO level instead of DEBUG
"logger": {
    "handlers": ["agent_file"],
    "level": "INFO"  # Instead of DEBUG
}
```

## Troubleshooting

### No logs appearing
1. Check that logging is initialized: `from Agent import *` or call `setup_logging()`
2. Verify trace directory exists and is writable
3. Check log levels are appropriate

### Logs too large
1. Increase rotation settings
2. Raise log level (INFO instead of DEBUG)
3. Implement custom filtering

### Missing context
1. Ensure logger is properly initialized with extra fields
2. Use context managers for block-level context
3. Check decorator usage

## Example: Complete Agent Implementation

```python
from Agent.utils import AgentLogger, PerformanceLogger, trace_agent_action, measure_performance

class CreatorAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = AgentLogger(agent_id, "creator")
        self.performance_logger = PerformanceLogger()

        self.logger.info("Agent initialized", extra={
            "agent_type": "creator",
            "initial_state": {"followers": 0, "content_count": 0}
        })

    @trace_agent_action(action_type="create_content")
    async def create_content(self, context):
        """Create content - automatically traced"""

        with measure_performance("content_generation"):
            content = await self._generate_content(context)

        self.logger.log_content_generation(
            content_type=content.type,
            content_id=content.id,
            metadata=content.metadata
        )

        return content

    async def update_from_feedback(self, feedback):
        """Update strategy based on feedback"""

        self.logger.log_learning_update(
            learning_stage=self.learning_stage,
            update_data={
                "reward": feedback.reward,
                "action": feedback.action
            }
        )

        await self.evolve_strategy()
```

## Support

For issues or questions:
1. Check the debug report: `debug_issue("your issue description")`
2. Analyze error patterns: `analyzer.find_error_patterns()`
3. Review agent journey: `analyzer.trace_agent_journey(agent_id)`
