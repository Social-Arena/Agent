# Trace System Quick Start Guide

## Overview

The Agent framework includes a comprehensive trace/logging system that stores all runtime logs to files for debugging. **No console output is used** - all logs go to structured files.

## Directory Structure

```
trace/
├── agents/              # Agent activity logs
├── learning/            # Learning system logs
├── interactions/        # Agent interactions
├── performance/         # Performance metrics
├── system/              # System-wide logs
├── errors/              # Error logs
├── events/              # System events
└── session_YYYYMMDD_HHMMSS/
    ├── agents/          # Individual agent logs
    │   ├── agent_001.log
    │   └── agent_002.log
    └── trace_summary.json
```

## Basic Usage

### 1. Import Logging Utilities

```python
from Agent.utils import AgentLogger, PerformanceLogger, trace_agent_action
```

### 2. Create Logger in Your Agent

```python
class MyAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = AgentLogger(agent_id, "creator")

        # Log initialization
        self.logger.info("Agent initialized")
```

### 3. Log Actions

```python
# Manual logging
self.logger.log_action(
    action_type="create_content",
    action_data={"type": "video"},
    decision_context={"trend": "AI"}
)

# Or use decorator for automatic logging
@trace_agent_action(action_type="create_content")
async def create_content(self):
    # Action automatically logged
    pass
```

### 4. Log Decisions

```python
self.logger.log_decision(
    decision_type="content_timing",
    decision_data={"scheduled_time": "18:00"},
    reasoning="Peak audience activity"
)
```

### 5. Log Learning Updates

```python
self.logger.log_learning_update(
    learning_stage="bandit",
    update_data={"epsilon": 0.1, "reward": 1.5}
)
```

## Debugging with Logs

### Quick Analysis

```python
from Agent.utils import analyze_session, debug_issue

# Analyze current session
analysis = analyze_session()
print(analysis)

# Generate debug report for an issue
report = debug_issue("Agent not generating content")
print(report)
```

### Detailed Analysis

```python
from Agent.utils import LogAnalyzer

# Create analyzer
analyzer = LogAnalyzer()

# Analyze specific agent
agent_summary = analyzer.analyze_agent_activity("agent_001")
print(f"Total actions: {agent_summary['action_counts']}")
print(f"Errors: {agent_summary['errors']}")

# Find error patterns
errors = analyzer.find_error_patterns()
print(f"Total errors: {errors['total_errors']}")
print(f"Common errors: {errors['most_common_messages']}")

# Trace agent journey
journey = analyzer.trace_agent_journey("agent_001")
for event in journey:
    print(f"{event['timestamp']}: {event['message']}")
```

### When Debugging an Issue

1. **Note the error details**: agent ID, error message, approximate time
2. **Generate debug report**:
   ```python
   report = debug_issue("Description of your issue")
   ```
3. **Examine error patterns**:
   ```python
   analyzer = LogAnalyzer()
   errors = analyzer.find_error_patterns()
   ```
4. **Trace the agent's journey**:
   ```python
   journey = analyzer.trace_agent_journey("problematic_agent_id")
   ```
5. **Look at causality chain**:
   ```python
   recent_error = analyzer.get_errors()[-1]
   causes = analyzer.find_causality_chain(recent_error)
   ```

## Common Patterns

### Pattern 1: Agent with Full Logging

```python
from Agent.utils import AgentLogger, trace_agent_action

class MyAgent:
    def __init__(self, agent_id: str):
        self.logger = AgentLogger(agent_id, "creator")
        self.logger.info("Initialized")

    @trace_agent_action()
    async def act(self, environment):
        # Automatically logged
        decision = await self.decide(environment)
        return await self.execute(decision)
```

### Pattern 2: Learning System Logging

```python
from Agent.utils import LearningLogger

class BanditLearner:
    def __init__(self, agent_id: str):
        self.logger = LearningLogger("bandit", agent_id)

    def update(self, action: str, reward: float):
        self.logger.log_reward(action, reward, self.cumulative)
        self.logger.log_exploration_exploitation(
            action,
            exploration=True,
            epsilon=0.1
        )
```

### Pattern 3: Performance Monitoring

```python
from Agent.utils import measure_performance

with measure_performance("content_generation"):
    content = await generate_content()
```

## Key Features

### 1. Automatic Session Management
Each run gets a unique session ID and directory:
- `trace/session_20250115_103045/`
- All logs for that session stored together
- Easy to analyze specific runs

### 2. Structured JSON Logs
Logs are in JSON format for easy parsing:
```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "level": "INFO",
  "logger": "agent.creator",
  "message": "Agent action: create_content",
  "extra": {
    "agent_id": "creator_001",
    "event_type": "agent_action"
  }
}
```

### 3. Individual Agent Logs
Each agent gets its own log file:
- `session_XXX/agents/creator_001.log`
- Easy to trace single agent's behavior
- No mixing of agent activities

### 4. Error Tracking
All errors automatically logged with:
- Full stack traces
- Error context
- Time of occurrence
- Related events

### 5. Performance Metrics
Track performance with:
- Operation latencies
- Throughput metrics
- Resource usage

## Best Practices

1. **Always initialize logger in __init__**:
   ```python
   self.logger = AgentLogger(agent_id, role)
   ```

2. **Use decorators for actions**:
   ```python
   @trace_agent_action()
   async def my_action(self):
       pass
   ```

3. **Include context in logs**:
   ```python
   self.logger.log_action(
       action_type="create_content",
       action_data={...},
       decision_context={...}  # Include decision context
   )
   ```

4. **Log state changes**:
   ```python
   self.logger.log_state_change(old_state, new_state, trigger)
   ```

5. **Use structured data**:
   ```python
   # Good
   logger.info("Content created", extra={"content_id": "123"})

   # Avoid
   logger.info(f"Content 123 created")
   ```

## Example: Complete Agent

```python
from Agent.utils import AgentLogger, trace_agent_action, measure_performance

class CreatorAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = AgentLogger(agent_id, "creator")
        self.logger.info("Agent initialized", extra={"followers": 0})

    @trace_agent_action(action_type="create_content")
    async def create_content(self, topic: str):
        with measure_performance("content_generation"):
            content = await self._generate(topic)

        self.logger.log_content_generation(
            content_type="video",
            content_id=content.id,
            metadata={"topic": topic}
        )

        return content

    async def update_strategy(self, feedback):
        self.logger.log_learning_update(
            learning_stage="bandit",
            update_data={"reward": feedback.reward}
        )
```

## Troubleshooting

### Issue: No logs appearing
- Check that logging is initialized (happens automatically on `from Agent import *`)
- Verify trace directory is writable
- Check log levels in config

### Issue: Logs too large
- Logs auto-rotate at configured sizes
- Consider raising log level (INFO instead of DEBUG)
- Check rotation settings in `Agent/config/logging_config.py`

### Issue: Can't find specific logs
- Use LogAnalyzer to search:
  ```python
  analyzer = LogAnalyzer()
  logs = analyzer.filter_by_agent("agent_id")
  ```

### Issue: Need to analyze old session
```python
analyzer = LogAnalyzer(session_id="20250115_103045")
```

## See Also

- Full documentation: `trace/README.md`
- Example usage: `examples/trace_example.py`
- Tests: `tests/test_logging.py`

## Quick Reference

```python
# Import
from Agent.utils import AgentLogger, LogAnalyzer, debug_issue

# Create logger
logger = AgentLogger(agent_id, role)

# Log action
logger.log_action(action_type, action_data, decision_context)

# Log decision
logger.log_decision(decision_type, decision_data, reasoning)

# Log learning
logger.log_learning_update(stage, update_data)

# Analyze
analyzer = LogAnalyzer()
summary = analyzer.analyze_agent_activity(agent_id)
errors = analyzer.find_error_patterns()
report = debug_issue("issue description")
```
