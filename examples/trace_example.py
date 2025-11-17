"""
Example: Using the Trace System

This example demonstrates how to use the comprehensive trace logging system
for debugging the Agent framework.
"""

import asyncio
from pathlib import Path

# Import logging utilities
from Agent.utils import (
    AgentLogger,
    LearningLogger,
    InteractionLogger,
    PerformanceLogger,
    EventLogger,
    trace_agent_action,
    trace_learning_update,
    measure_performance,
    trace_context,
    LogAnalyzer,
    debug_issue,
    get_trace_summary
)


# Example 1: Basic Agent Logging
class ExampleCreatorAgent:
    """Example agent with integrated logging"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Initialize logger
        self.logger = AgentLogger(agent_id=agent_id, agent_role="creator")
        self.performance_logger = PerformanceLogger()

        # Log initialization
        self.logger.info(
            "Agent initialized",
            extra={
                "initial_state": {
                    "followers": 0,
                    "content_count": 0,
                    "learning_stage": "cold_start"
                }
            }
        )

        # Add to trace summary
        trace_summary = get_trace_summary()
        trace_summary.add_agent(
            agent_id=agent_id,
            agent_role="creator",
            metadata={"niche": "tech", "experience_level": "beginner"}
        )

    @trace_agent_action(action_type="create_content")
    async def create_content(self, content_type: str, topic: str):
        """Create content - automatically traced by decorator"""

        self.logger.debug(f"Starting content creation: {content_type} about {topic}")

        # Simulate content generation with performance tracking
        with measure_performance("llm_generation"):
            await asyncio.sleep(0.1)  # Simulate LLM call
            content_id = f"content_{self.agent_id}_{content_type}"

        # Log content generation
        self.logger.log_content_generation(
            content_type=content_type,
            content_id=content_id,
            metadata={
                "topic": topic,
                "word_count": 500,
                "estimated_reach": 1000
            }
        )

        return content_id

    @trace_agent_action(action_type="engage_audience")
    async def engage_audience(self, audience_id: str, engagement_type: str):
        """Engage with audience"""

        self.logger.log_interaction(
            interaction_type=engagement_type,
            target_id=audience_id,
            interaction_data={
                "timestamp": "now",
                "context": "viral_content"
            }
        )

        return True

    async def make_decision(self, decision_type: str, options: list):
        """Make a strategic decision"""

        # Log decision with reasoning
        selected_option = options[0]  # Simple selection

        self.logger.log_decision(
            decision_type=decision_type,
            decision_data={
                "options": options,
                "selected": selected_option,
                "confidence": 0.75
            },
            reasoning=f"Selected {selected_option} based on historical performance"
        )

        return selected_option

    async def update_learning(self, reward: float, action: str):
        """Update learning based on reward"""

        learning_logger = LearningLogger(learning_type="bandit", agent_id=self.agent_id)

        # Log reward
        learning_logger.log_reward(
            action_id=action,
            reward=reward,
            cumulative_reward=reward * 10  # Simplified
        )

        # Log exploration/exploitation decision
        learning_logger.log_exploration_exploitation(
            action_selected=action,
            exploration=True,
            epsilon=0.1
        )

        self.logger.log_learning_update(
            learning_stage="bandit",
            update_data={
                "action": action,
                "reward": reward,
                "epsilon": 0.1
            }
        )


# Example 2: Interaction Logging
async def example_interactions():
    """Example of logging interactions between agents"""

    interaction_logger = InteractionLogger()

    # Log content interaction
    interaction_logger.log_content_interaction(
        agent_id="audience_001",
        content_id="content_creator_001_video",
        interaction_type="like",
        interaction_metadata={
            "timestamp": "2025-01-15T10:30:00",
            "engagement_score": 0.8
        }
    )

    # Log social connection
    interaction_logger.log_social_connection(
        agent_id_1="creator_001",
        agent_id_2="creator_002",
        connection_type="collaboration"
    )

    # Log viral cascade
    interaction_logger.log_viral_cascade(
        content_id="content_creator_001_video",
        cascade_data={
            "total_shares": 5000,
            "total_likes": 15000,
            "reach": 100000,
            "cascade_depth": 4,
            "viral_coefficient": 2.5
        }
    )


# Example 3: System Events
async def example_system_events():
    """Example of logging system-wide events"""

    event_logger = EventLogger()

    # Log simulation start
    event_logger.log_simulation_start(
        config={
            "num_agents": 100,
            "simulation_duration": 3600,
            "learning_stages": ["bandit", "rl", "evolution"]
        }
    )

    # Simulate some work
    await asyncio.sleep(0.1)

    # Log checkpoint
    event_logger.log_checkpoint(
        checkpoint_id="checkpoint_001",
        checkpoint_data={
            "timestamp": "2025-01-15T10:35:00",
            "agents_active": 100,
            "total_content": 500,
            "total_interactions": 2000
        }
    )

    # Log simulation end
    event_logger.log_simulation_end(
        summary={
            "total_duration": 3600,
            "total_agents": 100,
            "total_content": 5000,
            "total_interactions": 50000,
            "viral_cascades": 25
        }
    )


# Example 4: Error Handling and Tracing
async def example_error_handling():
    """Example of error logging and tracing"""

    agent = ExampleCreatorAgent("creator_error_test")

    try:
        # Simulate an error
        raise ValueError("Content generation failed: insufficient context")
    except Exception as e:
        agent.logger.error(
            f"Error during content generation: {str(e)}",
            extra={
                "error_type": type(e).__name__,
                "context": "content_generation",
                "attempted_action": "create_video"
            },
            exc_info=True
        )

        # Add to trace summary
        trace_summary = get_trace_summary()
        trace_summary.add_error(
            error_type=type(e).__name__,
            error_message=str(e),
            stacktrace="<traceback here>"
        )


# Example 5: Performance Tracking
async def example_performance_tracking():
    """Example of performance metric logging"""

    perf_logger = PerformanceLogger()

    # Track operation latency
    with measure_performance("database_query"):
        await asyncio.sleep(0.05)  # Simulate DB query

    # Log custom metrics
    perf_logger.log_metric(
        metric_name="agents.active",
        value=100,
        tags={"environment": "simulation"}
    )

    perf_logger.log_throughput(
        operation="content_generation",
        count=500,
        duration_s=60.0
    )

    perf_logger.log_latency(
        operation="model_inference",
        latency_ms=125.5,
        tags={"model": "gpt-4", "agent_type": "creator"}
    )


# Example 6: Log Analysis
async def example_log_analysis():
    """Example of analyzing logs for debugging"""

    # Create and run some agents
    agents = [ExampleCreatorAgent(f"creator_{i:03d}") for i in range(5)]

    for agent in agents:
        await agent.create_content("video", "AI trends")
        await agent.engage_audience("audience_001", "comment")
        await agent.update_learning(reward=1.5, action="create_video")

    # Analyze the session
    print("\n" + "="*80)
    print("LOG ANALYSIS")
    print("="*80)

    analyzer = LogAnalyzer()

    # Analyze specific agent
    print("\n--- Agent Activity Analysis ---")
    agent_analysis = analyzer.analyze_agent_activity("creator_000")
    print(f"Agent: {agent_analysis['agent_id']}")
    print(f"Total logs: {agent_analysis['total_logs']}")
    print(f"Event counts: {agent_analysis['event_counts']}")
    print(f"Action counts: {agent_analysis['action_counts']}")

    # Analyze learning progress
    print("\n--- Learning Progress Analysis ---")
    learning_analysis = analyzer.analyze_learning_progress("creator_000")
    print(f"Training steps: {learning_analysis['total_training_steps']}")
    print(f"Rewards: {learning_analysis['total_rewards']}")
    print(f"Policy updates: {learning_analysis['total_policy_updates']}")

    # Analyze interactions
    print("\n--- Interaction Analysis ---")
    interaction_analysis = analyzer.analyze_interactions()
    print(f"Total interactions: {interaction_analysis['total_interactions']}")
    print(f"Interaction types: {interaction_analysis['interaction_counts']}")

    # Find error patterns
    print("\n--- Error Pattern Analysis ---")
    error_analysis = analyzer.find_error_patterns()
    print(f"Total errors: {error_analysis['total_errors']}")

    # Generate debug report
    print("\n--- Debug Report ---")
    report = debug_issue("Testing trace system functionality")
    print(report[:500] + "...")  # Print first 500 chars


# Example 7: Context-Based Logging
async def example_context_logging():
    """Example of using context for enhanced logging"""

    agent = ExampleCreatorAgent("creator_context")

    # Use context manager for automatic context inclusion
    with trace_context(
        agent.logger,
        "viral_optimization",
        content_id="content_123",
        optimization_stage="hashtag_selection"
    ):
        # All logs within this block include the context
        agent.logger.info("Analyzing trending hashtags")
        await asyncio.sleep(0.05)
        agent.logger.info("Selected optimal hashtags")

    # Context automatically logged with entry/exit and timing


# Main execution
async def main():
    """Run all examples"""

    print("="*80)
    print("AGENT FRAMEWORK TRACE SYSTEM EXAMPLES")
    print("="*80)

    # Example 1: Basic agent logging
    print("\n[1] Basic Agent Logging")
    agent = ExampleCreatorAgent("creator_001")
    await agent.create_content("video", "Machine Learning Basics")
    await agent.engage_audience("audience_001", "reply")
    decision = await agent.make_decision("content_type", ["video", "article", "podcast"])
    await agent.update_learning(reward=2.5, action="create_video")

    # Example 2: Interaction logging
    print("\n[2] Interaction Logging")
    await example_interactions()

    # Example 3: System events
    print("\n[3] System Events Logging")
    await example_system_events()

    # Example 4: Error handling
    print("\n[4] Error Handling and Tracing")
    await example_error_handling()

    # Example 5: Performance tracking
    print("\n[5] Performance Tracking")
    await example_performance_tracking()

    # Example 6: Context-based logging
    print("\n[6] Context-Based Logging")
    await example_context_logging()

    # Example 7: Log analysis
    print("\n[7] Log Analysis and Debugging")
    await example_log_analysis()

    # Finalize trace summary
    trace_summary = get_trace_summary()
    trace_summary.finalize()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print(f"\nCheck the trace/ directory for detailed logs")
    print(f"Session directory: trace/session_{trace_summary.summary_data['session_id']}/")


if __name__ == "__main__":
    asyncio.run(main())
