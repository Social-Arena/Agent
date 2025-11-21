"""
Base Agent - Abstract base class for all agent types
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from Agent.utils.logging_utils import get_logger


class AgentRole(Enum):
    """Agent role types"""
    CREATOR = "creator"
    AUDIENCE = "audience"
    BRAND = "brand"
    MODERATOR = "moderator"


class LearningStage(Enum):
    """Progressive learning stages"""
    COLD_START = "cold_start"          # Offline imitation learning
    BANDIT = "bandit"                   # Multi-armed bandit
    REINFORCEMENT = "reinforcement"     # Reinforcement learning (PPO/DDPG)
    EVOLUTION = "evolution"             # Evolution strategy


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_id: str
    role: AgentRole
    learning_stage: LearningStage = LearningStage.COLD_START
    
    # Strategy configuration
    strategy_config: Optional[Dict[str, Any]] = None
    
    # Content generation configuration
    content_config: Optional[Dict[str, Any]] = None
    
    # Norm learning configuration
    norm_config: Optional[Dict[str, Any]] = None
    
    # Role-specific configuration
    role_specific_config: Optional[Dict[str, Any]] = None


@dataclass
class EnvironmentState:
    """Environment state information"""
    timestamp: datetime
    trending_topics: List[str]
    audience_activity: Dict[str, Any]
    platform_metrics: Dict[str, float]
    recommended_content: Optional[List[Any]] = None
    social_signals: Optional[Dict[str, Any]] = None


@dataclass
class AgentAction:
    """Agent action representation"""
    action_type: str
    agent_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    expected_outcome: Optional[Dict[str, float]] = None


@dataclass
class ActionFeedback:
    """Feedback from action execution"""
    action_id: str
    action_type: str
    success: bool
    metrics: Dict[str, float]
    timestamp: datetime
    
    # Content-related feedback
    content_id: Optional[str] = None
    engagement_metrics: Optional[Dict[str, int]] = None
    
    # Audience feedback
    audience_reaction: Optional[Dict[str, Any]] = None
    audience_insights: Optional[Dict[str, Any]] = None
    
    # Social context
    social_context: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Agent performance metrics"""
    agent_id: str
    learning_stage: LearningStage
    total_actions: int
    successful_actions: int
    average_reward: float
    cumulative_reward: float
    learning_progress: float
    timestamp: datetime


@dataclass
class AgentState:
    """Complete agent state"""
    agent_id: str
    role: AgentRole
    learning_stage: LearningStage
    current_strategy: Optional[Any]
    performance_metrics: PerformanceMetrics
    memory_summary: Dict[str, Any]
    timestamp: datetime


class BaseAgent(ABC):
    """Base agent abstract class - common interface for all agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, config: AgentConfig):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        
        # Initialize logger
        self.logger = get_logger(
            f"{self.__class__.__name__}",
            component=f"agent_{role.value}"
        )
        
        # Core components (will be initialized in subclasses)
        self.memory = None
        self.strategy_head = None
        self.content_head = None
        self.norm_learner = None
        
        # Learning related
        self.learning_stage = config.learning_stage
        self.bandit_learner = None
        self.rl_trainer = None
        self.evolution_strategy = None
        
        # Performance tracking
        self.performance_history = []
        self.current_strategy = None
        self.action_count = 0
        self.total_reward = 0.0
        
        # State
        self.is_initialized = False
        self.last_environment_state = None
        
        self.logger.info(f"Agent {agent_id} created", extra={
            "agent_id": agent_id,
            "role": role.value,
            "learning_stage": config.learning_stage.value
        })
    
    async def initialize(self) -> None:
        """Initialize agent components"""
        if self.is_initialized:
            return
        
        self.logger.info(f"Initializing agent {self.agent_id}")
        
        # Initialize components (implemented in subclasses)
        await self._initialize_components()
        
        # Load initial strategy if in cold start
        if self.learning_stage == LearningStage.COLD_START:
            await self._load_initial_strategy()
        
        self.is_initialized = True
        
        self.logger.info(f"Agent {self.agent_id} initialized successfully")
    
    @abstractmethod
    async def _initialize_components(self) -> None:
        """Initialize agent-specific components - implemented by subclasses"""
        pass
    
    @abstractmethod
    async def act(self, environment_state: EnvironmentState) -> AgentAction:
        """
        Agent action - implemented by specific roles
        
        Args:
            environment_state: Current environment state
            
        Returns:
            AgentAction: Action to be executed
        """
        pass
    
    @abstractmethod
    async def update_from_feedback(self, feedback: ActionFeedback) -> None:
        """
        Learn from feedback - implemented by specific roles
        
        Args:
            feedback: Feedback from action execution
        """
        pass
    
    async def evolve_strategy(self) -> None:
        """Strategy evolution based on current learning stage"""
        if self.learning_stage == LearningStage.BANDIT:
            await self._bandit_update()
        elif self.learning_stage == LearningStage.REINFORCEMENT:
            await self._rl_update()
        elif self.learning_stage == LearningStage.EVOLUTION:
            await self._evolution_update()
        
        self.logger.debug(f"Strategy evolved", extra={
            "agent_id": self.agent_id,
            "learning_stage": self.learning_stage.value
        })
    
    async def _bandit_update(self) -> None:
        """Multi-armed bandit update"""
        if self.bandit_learner:
            # Update will be implemented when BanditLearner is ready
            pass
    
    async def _rl_update(self) -> None:
        """Reinforcement learning update"""
        if self.rl_trainer:
            # Update will be implemented when RLTrainer is ready
            pass
    
    async def _evolution_update(self) -> None:
        """Evolution strategy update"""
        if self.evolution_strategy:
            # Update will be implemented when EvolutionStrategy is ready
            pass
    
    def transition_learning_stage(self, new_stage: LearningStage) -> None:
        """
        Transition to new learning stage
        
        Args:
            new_stage: New learning stage
        """
        old_stage = self.learning_stage
        self.learning_stage = new_stage
        
        if self.memory:
            self.memory.log_stage_transition(new_stage)
        
        self.logger.info(f"Learning stage transition", extra={
            "agent_id": self.agent_id,
            "old_stage": old_stage.value,
            "new_stage": new_stage.value
        })
    
    async def _load_initial_strategy(self) -> None:
        """Load initial strategy for cold start"""
        # Will be implemented based on role
        pass
    
    def get_current_state(self) -> AgentState:
        """Get current agent state"""
        performance = PerformanceMetrics(
            agent_id=self.agent_id,
            learning_stage=self.learning_stage,
            total_actions=self.action_count,
            successful_actions=len([r for r in self.performance_history if r > 0]),
            average_reward=sum(self.performance_history) / max(len(self.performance_history), 1),
            cumulative_reward=self.total_reward,
            learning_progress=self._calculate_learning_progress(),
            timestamp=datetime.now()
        )
        
        memory_summary = {}
        if self.memory:
            memory_summary = self.memory.get_summary()
        
        return AgentState(
            agent_id=self.agent_id,
            role=self.role,
            learning_stage=self.learning_stage,
            current_strategy=self.current_strategy,
            performance_metrics=performance,
            memory_summary=memory_summary,
            timestamp=datetime.now()
        )
    
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress (0-1)"""
        if not self.performance_history:
            return 0.0
        
        # Simple progress based on reward trend
        if len(self.performance_history) < 10:
            return 0.1
        
        recent_avg = sum(self.performance_history[-10:]) / 10
        early_avg = sum(self.performance_history[:10]) / 10
        
        if early_avg == 0:
            return 0.5
        
        improvement = (recent_avg - early_avg) / abs(early_avg)
        return min(max(improvement, 0.0), 1.0)
    
    def record_action(self, action: AgentAction, reward: float) -> None:
        """Record action and reward"""
        self.action_count += 1
        self.total_reward += reward
        self.performance_history.append(reward)
        
        self.logger.debug(f"Action recorded", extra={
            "agent_id": self.agent_id,
            "action_type": action.action_type,
            "reward": reward,
            "total_actions": self.action_count
        })
    
    async def receive_environment_update(self, environment_state: EnvironmentState) -> None:
        """Receive environment updates"""
        self.last_environment_state = environment_state
        
        # Adapt to environment changes if needed
        await self._adapt_to_environment(environment_state)
    
    async def _adapt_to_environment(self, environment_state: EnvironmentState) -> None:
        """Adapt behavior to environment changes"""
        # To be implemented in subclasses
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role.value}, stage={self.learning_stage.value})"

