"""
Trace Decorators
Provides decorators for automatic tracing of function calls, performance, and errors.
"""

import functools
import time
from typing import Callable, Any, Optional
from contextlib import contextmanager

from Agent.utils.logging_utils import (
    AgentLogger,
    LearningLogger,
    PerformanceLogger,
    TraceLogger,
    get_trace_summary
)


def trace_agent_action(action_type: Optional[str] = None):
    """
    Decorator to automatically trace agent actions

    Args:
        action_type: Type of action (if None, uses function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get agent logger from self
            if not hasattr(self, 'logger') or not isinstance(self.logger, AgentLogger):
                # Call function without tracing if no logger
                return await func(self, *args, **kwargs)

            _action_type = action_type or func.__name__

            # Log action start
            self.logger.log_action(
                action_type=_action_type,
                action_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                decision_context=getattr(self, 'last_decision_context', None)
            )

            start_time = time.time()
            error_occurred = False

            try:
                result = await func(self, *args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                self.logger.error(
                    f"Error in action {_action_type}: {str(e)}",
                    extra={"action_type": _action_type},
                    exc_info=True
                )
                raise
            finally:
                elapsed_ms = (time.time() - start_time) * 1000

                # Log performance
                if hasattr(self, 'performance_logger'):
                    self.performance_logger.log_latency(
                        operation=f"action.{_action_type}",
                        latency_ms=elapsed_ms,
                        tags={
                            "agent_id": self.agent_id,
                            "success": str(not error_occurred)
                        }
                    )

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Get agent logger from self
            if not hasattr(self, 'logger') or not isinstance(self.logger, AgentLogger):
                return func(self, *args, **kwargs)

            _action_type = action_type or func.__name__

            self.logger.log_action(
                action_type=_action_type,
                action_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                decision_context=getattr(self, 'last_decision_context', None)
            )

            start_time = time.time()
            error_occurred = False

            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                self.logger.error(
                    f"Error in action {_action_type}: {str(e)}",
                    extra={"action_type": _action_type},
                    exc_info=True
                )
                raise
            finally:
                elapsed_ms = (time.time() - start_time) * 1000

                if hasattr(self, 'performance_logger'):
                    self.performance_logger.log_latency(
                        operation=f"action.{_action_type}",
                        latency_ms=elapsed_ms,
                        tags={
                            "agent_id": self.agent_id,
                            "success": str(not error_occurred)
                        }
                    )

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def trace_learning_update(update_type: Optional[str] = None):
    """
    Decorator to trace learning updates

    Args:
        update_type: Type of learning update
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            if not hasattr(self, 'logger') or not isinstance(self.logger, LearningLogger):
                return await func(self, *args, **kwargs)

            _update_type = update_type or func.__name__

            start_time = time.time()

            try:
                result = await func(self, *args, **kwargs)

                # Log the update
                self.logger.log_policy_update(
                    update_type=_update_type,
                    update_data={"result": str(result)[:200]}
                )

                return result
            except Exception as e:
                self.logger.error(
                    f"Error in learning update {_update_type}: {str(e)}",
                    exc_info=True
                )
                raise
            finally:
                elapsed_ms = (time.time() - start_time) * 1000

                if hasattr(self, 'performance_logger'):
                    self.performance_logger.log_latency(
                        operation=f"learning.{_update_type}",
                        latency_ms=elapsed_ms
                    )

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            if not hasattr(self, 'logger') or not isinstance(self.logger, LearningLogger):
                return func(self, *args, **kwargs)

            _update_type = update_type or func.__name__

            start_time = time.time()

            try:
                result = func(self, *args, **kwargs)

                self.logger.log_policy_update(
                    update_type=_update_type,
                    update_data={"result": str(result)[:200]}
                )

                return result
            except Exception as e:
                self.logger.error(
                    f"Error in learning update {_update_type}: {str(e)}",
                    exc_info=True
                )
                raise
            finally:
                elapsed_ms = (time.time() - start_time) * 1000

                if hasattr(self, 'performance_logger'):
                    self.performance_logger.log_latency(
                        operation=f"learning.{_update_type}",
                        latency_ms=elapsed_ms
                    )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def trace_performance(operation_name: Optional[str] = None):
    """
    Decorator to trace performance of any function

    Args:
        operation_name: Name of the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _operation_name = operation_name or f"{func.__module__}.{func.__name__}"

            perf_logger = PerformanceLogger()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                perf_logger.log_latency(
                    operation=_operation_name,
                    latency_ms=elapsed_ms
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _operation_name = operation_name or f"{func.__module__}.{func.__name__}"

            perf_logger = PerformanceLogger()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                perf_logger.log_latency(
                    operation=_operation_name,
                    latency_ms=elapsed_ms
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def trace_errors(logger: Optional[TraceLogger] = None):
    """
    Decorator to automatically trace errors

    Args:
        logger: Logger to use (if None, creates a default logger)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _logger = logger or TraceLogger("error_trace")

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                _logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    },
                    exc_info=True
                )

                # Add to trace summary
                trace_summary = get_trace_summary()
                trace_summary.add_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stacktrace=traceback.format_exc()
                )

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _logger = logger or TraceLogger("error_trace")

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                _logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    },
                    exc_info=True
                )

                trace_summary = get_trace_summary()
                trace_summary.add_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stacktrace=traceback.format_exc()
                )

                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_context(logger: TraceLogger, context_name: str, **context_data):
    """
    Context manager for tracing a block of code

    Args:
        logger: Logger to use
        context_name: Name of the context
        **context_data: Additional context data

    Example:
        with trace_context(agent.logger, "content_generation", content_type="video"):
            # Generate content
            pass
    """
    logger.debug(f"Entering context: {context_name}", extra=context_data)
    start_time = time.time()

    try:
        with logger.context(**context_data):
            yield
    except Exception as e:
        logger.error(
            f"Error in context {context_name}: {str(e)}",
            extra=context_data,
            exc_info=True
        )
        raise
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Exiting context: {context_name}",
            extra={**context_data, "elapsed_ms": elapsed_ms}
        )


@contextmanager
def measure_performance(operation: str, logger: Optional[PerformanceLogger] = None):
    """
    Context manager for measuring performance

    Args:
        operation: Operation name
        logger: Performance logger (creates new one if None)

    Example:
        with measure_performance("model_inference"):
            result = model.predict(input_data)
    """
    _logger = logger or PerformanceLogger()
    start_time = time.time()

    try:
        yield
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        _logger.log_latency(operation, elapsed_ms)


import asyncio
import traceback
