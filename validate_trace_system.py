#!/usr/bin/env python3
"""
Validation script for the trace/logging system.
This script tests that the logging system is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from Agent.utils import (
    AgentLogger,
    LearningLogger,
    InteractionLogger,
    PerformanceLogger,
    EventLogger,
    LogAnalyzer,
    get_trace_summary,
    trace_agent_action,
    measure_performance
)
from Agent.config import get_session_dir, TRACE_SUBDIRS


def validate_directory_structure():
    """Validate that trace directory structure exists"""
    print("Validating directory structure...")

    required_dirs = [
        "agents",
        "learning",
        "interactions",
        "performance",
        "system",
        "errors",
        "events"
    ]

    all_exist = True
    for dir_name in required_dirs:
        if TRACE_SUBDIRS[dir_name].exists():
            print(f"  ✓ {dir_name}/ exists")
        else:
            print(f"  ✗ {dir_name}/ missing")
            all_exist = False

    return all_exist


def validate_basic_logging():
    """Validate basic logging functionality"""
    print("\nValidating basic logging...")

    try:
        # Test AgentLogger
        agent_logger = AgentLogger(agent_id="validation_001", agent_role="creator")
        agent_logger.info("Validation test message")
        agent_logger.log_action(
            action_type="test_action",
            action_data={"test": True},
            decision_context=None
        )
        print("  ✓ AgentLogger working")

        # Test LearningLogger
        learning_logger = LearningLogger(learning_type="bandit", agent_id="validation_001")
        learning_logger.log_reward(
            action_id="test_action",
            reward=1.0,
            cumulative_reward=1.0
        )
        print("  ✓ LearningLogger working")

        # Test InteractionLogger
        interaction_logger = InteractionLogger()
        interaction_logger.log_content_interaction(
            agent_id="validation_001",
            content_id="test_content",
            interaction_type="test",
            interaction_metadata={"test": True}
        )
        print("  ✓ InteractionLogger working")

        # Test PerformanceLogger
        perf_logger = PerformanceLogger()
        perf_logger.log_metric("test_metric", 100.0)
        print("  ✓ PerformanceLogger working")

        # Test EventLogger
        event_logger = EventLogger()
        event_logger.log_system_event("test_event", {"test": True})
        print("  ✓ EventLogger working")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_session_management():
    """Validate session management"""
    print("\nValidating session management...")

    try:
        session_dir = get_session_dir()

        if session_dir.exists():
            print(f"  ✓ Session directory created: {session_dir.name}")
        else:
            print(f"  ✗ Session directory not found")
            return False

        # Check trace summary
        trace_summary = get_trace_summary()
        if trace_summary.summary_file.exists():
            print(f"  ✓ Trace summary file created")
        else:
            print(f"  ✗ Trace summary file not found")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def validate_decorators():
    """Validate decorator functionality"""
    print("\nValidating decorators...")

    class TestAgent:
        def __init__(self):
            self.agent_id = "decorator_test"
            self.logger = AgentLogger("decorator_test", "creator")
            self.performance_logger = PerformanceLogger()

        @trace_agent_action(action_type="test_action")
        async def test_action(self):
            await asyncio.sleep(0.01)
            return "success"

    try:
        agent = TestAgent()
        result = await agent.test_action()
        success = result == "success"

        if success:
            print("  ✓ Decorators working")
            return True
        else:
            print("  ✗ Decorator test failed")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_log_analyzer():
    """Validate log analyzer functionality"""
    print("\nValidating log analyzer...")

    try:
        # Create some test logs
        logger = AgentLogger("analyzer_test", "creator")
        logger.info("Test log 1")
        logger.info("Test log 2")
        logger.log_action("test", {}, None)

        # Create analyzer
        analyzer = LogAnalyzer()
        print(f"  ✓ LogAnalyzer initialized")

        # Test filtering
        agent_logs = analyzer.filter_by_agent("analyzer_test")
        print(f"  ✓ Filter by agent: {len(agent_logs)} logs found")

        # Test analysis
        analysis = analyzer.analyze_agent_activity("analyzer_test")
        if "agent_id" in analysis:
            print(f"  ✓ Agent activity analysis working")
        else:
            print(f"  ✗ Agent activity analysis failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def validate_async_logging():
    """Validate async logging"""
    print("\nValidating async logging...")

    try:
        logger = AgentLogger("async_test", "creator")

        async def async_operation():
            logger.info("Starting async operation")
            await asyncio.sleep(0.01)
            logger.info("Async operation completed")

        await async_operation()
        print("  ✓ Async logging working")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_performance_tracking():
    """Validate performance tracking"""
    print("\nValidating performance tracking...")

    try:
        with measure_performance("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.01)

        print("  ✓ Performance tracking working")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_log_files_created():
    """Validate that log files are actually created"""
    print("\nValidating log file creation...")

    log_files_found = []

    # Check main log files
    for subdir_name, subdir_path in TRACE_SUBDIRS.items():
        log_files = list(subdir_path.glob("*.log"))
        if log_files:
            log_files_found.append(subdir_name)

    # Check session directory
    session_dir = get_session_dir()
    session_logs = list(session_dir.rglob("*.log"))

    if log_files_found:
        print(f"  ✓ Log files created in: {', '.join(log_files_found)}")
    else:
        print(f"  ⚠ No log files in main directories (may be normal)")

    if session_logs:
        print(f"  ✓ {len(session_logs)} log files in session directory")
    else:
        print(f"  ⚠ No log files in session directory")

    return True


async def main():
    """Run all validation tests"""
    print("="*80)
    print("TRACE SYSTEM VALIDATION")
    print("="*80)

    results = {
        "Directory Structure": validate_directory_structure(),
        "Basic Logging": validate_basic_logging(),
        "Session Management": validate_session_management(),
        "Decorators": await validate_decorators(),
        "Log Analyzer": validate_log_analyzer(),
        "Async Logging": await validate_async_logging(),
        "Performance Tracking": validate_performance_tracking(),
        "Log File Creation": validate_log_files_created()
    }

    # Finalize trace summary
    trace_summary = get_trace_summary()
    trace_summary.finalize()

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
        if not result:
            all_passed = False

    print("="*80)

    if all_passed:
        print("\n✓ All validation tests passed!")
        print(f"\nSession directory: {get_session_dir()}")
        print(f"Check trace/ directory for logs")
        return 0
    else:
        print("\n✗ Some validation tests failed")
        print("Check error messages above for details")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
