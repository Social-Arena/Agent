"""
Strategy Head - Decision engine for agents
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from Agent.utils.logging_utils import get_logger


class ActionType(Enum):
    """Types of actions agents can take"""
    CREATE_CONTENT = "create_content"
    ENGAGE_AUDIENCE = "engage_audience"
    COLLABORATE = "collaborate"
    CONSUME_CONTENT = "consume_content"
    MODERATE_CONTENT = "moderate_content"
    LAUNCH_CAMPAIGN = "launch_campaign"
    IDLE = "idle"


@dataclass
class ActionContext:
    """Context for action decision"""
    timestamp: datetime
    trends: List[str]
    audience_state: Dict[str, Any]
    available_actions: List[ActionType]
    agent_state: Dict[str, Any]
    platform_metrics: Optional[Dict[str, float]] = None


@dataclass
class StrategyDecision:
    """Decision output from strategy head"""
    action: ActionType
    parameters: Dict[str, Any]
    confidence: float  # 0-1
    rationale: Dict[str, Any]
    expected_reward: float = 0.0


@dataclass
class TimingDecision:
    """Decision about when to post content"""
    scheduled_time: datetime
    confidence: float
    rationale: str


@dataclass
class CollaborationOpp:
    """Collaboration opportunity"""
    partner_id: str
    partner_role: str
    opportunity_type: str
    estimated_benefit: float
    terms: Dict[str, Any]


@dataclass
class CollaborationDecision:
    """Decision on collaboration"""
    accept: bool
    terms: Dict[str, Any]
    rationale: str


@dataclass
class StrategyConfig:
    """Configuration for strategy head"""
    role: str
    exploration_rate: float = 0.1
    risk_tolerance: float = 0.5
    collaboration_threshold: float = 0.6
    planning_horizon: int = 24  # hours


class StrategyHead:
    """Strategy head - decision engine"""
    
    def __init__(self, role, config: StrategyConfig):
        self.role = role
        self.config = config
        self.current_strategy = None
        self.decision_history: List[StrategyDecision] = []
        
        self.logger = get_logger(f"StrategyHead_{role}", component="agent_strategy")
        
        self.logger.info(f"StrategyHead initialized", extra={
            "role": str(role),
            "exploration_rate": config.exploration_rate
        })
    
    async def decide_action(self, context: ActionContext) -> StrategyDecision:
        """
        Decide on action considering environment, history, goals
        
        Args:
            context: Current action context
            
        Returns:
            StrategyDecision: Chosen action and parameters
        """
        # Assess situation
        situation_assessment = self._assess_situation(context)
        
        # Evaluate available actions
        action_values = {}
        for action in context.available_actions:
            value = self._evaluate_action(action, situation_assessment, context)
            action_values[action] = value
        
        # Select action (with exploration)
        selected_action = self._select_action(action_values)
        
        # Generate parameters for action
        parameters = self._generate_action_parameters(selected_action, context)
        
        decision = StrategyDecision(
            action=selected_action,
            parameters=parameters,
            confidence=action_values[selected_action],
            rationale=situation_assessment,
            expected_reward=self._estimate_reward(selected_action, parameters)
        )
        
        self.decision_history.append(decision)
        
        self.logger.debug(f"Action decided", extra={
            "action": selected_action.value,
            "confidence": decision.confidence,
            "expected_reward": decision.expected_reward
        })
        
        return decision
    
    async def decide_content_timing(self, content: Any) -> TimingDecision:
        """
        Decide optimal posting time
        
        Args:
            content: Content to be posted
            
        Returns:
            TimingDecision: When to post
        """
        # Analyze audience activity patterns
        audience_activity = self._analyze_audience_activity()
        
        # Consider competition
        competition_level = self._assess_competition_by_time()
        
        # Optimize timing
        optimal_time = self._optimize_timing(audience_activity, competition_level)
        
        decision = TimingDecision(
            scheduled_time=optimal_time,
            confidence=0.7,  # Fixed confidence for now
            rationale="Based on audience activity and competition analysis"
        )
        
        self.logger.debug(f"Timing decided", extra={
            "scheduled_time": optimal_time.isoformat(),
            "confidence": decision.confidence
        })
        
        return decision
    
    async def decide_collaboration(self, opportunity: CollaborationOpp) -> CollaborationDecision:
        """
        Decide on collaboration opportunity
        
        Args:
            opportunity: Collaboration opportunity
            
        Returns:
            CollaborationDecision: Accept or reject
        """
        # Evaluate partner compatibility
        compatibility = self._evaluate_partner_compatibility(opportunity.partner_id)
        
        # Estimate mutual benefit
        benefit_estimate = opportunity.estimated_benefit
        
        # Make decision
        should_collaborate = (
            compatibility > self.config.collaboration_threshold and
            benefit_estimate > 0.5
        )
        
        decision = CollaborationDecision(
            accept=should_collaborate,
            terms=opportunity.terms if should_collaborate else {},
            rationale=f"Compatibility: {compatibility:.2f}, Benefit: {benefit_estimate:.2f}"
        )
        
        self.logger.info(f"Collaboration decision", extra={
            "partner_id": opportunity.partner_id,
            "accept": should_collaborate,
            "compatibility": compatibility,
            "benefit": benefit_estimate
        })
        
        return decision
    
    def _assess_situation(self, context: ActionContext) -> Dict[str, Any]:
        """Assess current situation"""
        assessment = {
            "trend_alignment": self._calculate_trend_alignment(context.trends),
            "audience_engagement": self._calculate_audience_engagement(context.audience_state),
            "platform_health": self._calculate_platform_health(context.platform_metrics),
            "agent_readiness": self._calculate_agent_readiness(context.agent_state),
            "timestamp": context.timestamp
        }
        
        return assessment
    
    def _evaluate_action(self, action: ActionType, assessment: Dict[str, Any], context: ActionContext) -> float:
        """Evaluate action value"""
        # Base value
        value = 0.5
        
        # Adjust based on situation
        if action == ActionType.CREATE_CONTENT:
            value += assessment.get("trend_alignment", 0) * 0.3
            value += assessment.get("agent_readiness", 0) * 0.2
        elif action == ActionType.ENGAGE_AUDIENCE:
            value += assessment.get("audience_engagement", 0) * 0.4
        elif action == ActionType.IDLE:
            value = 0.1  # Low value for idle
        
        return min(max(value, 0.0), 1.0)
    
    def _select_action(self, action_values: Dict[ActionType, float]) -> ActionType:
        """Select action with exploration"""
        import random
        
        # Exploration vs exploitation
        if random.random() < self.config.exploration_rate:
            # Explore: random action
            return random.choice(list(action_values.keys()))
        else:
            # Exploit: best action
            return max(action_values, key=action_values.get)
    
    def _generate_action_parameters(self, action: ActionType, context: ActionContext) -> Dict[str, Any]:
        """Generate parameters for action"""
        parameters = {
            "action_type": action.value,
            "timestamp": context.timestamp.isoformat()
        }
        
        if action == ActionType.CREATE_CONTENT:
            parameters.update({
                "trending_topics": context.trends[:3],
                "target_audience": "general"
            })
        elif action == ActionType.ENGAGE_AUDIENCE:
            parameters.update({
                "engagement_type": "like_and_comment",
                "target_count": 10
            })
        
        return parameters
    
    def _estimate_reward(self, action: ActionType, parameters: Dict[str, Any]) -> float:
        """Estimate expected reward for action"""
        # Simple estimation
        base_rewards = {
            ActionType.CREATE_CONTENT: 0.7,
            ActionType.ENGAGE_AUDIENCE: 0.5,
            ActionType.COLLABORATE: 0.8,
            ActionType.CONSUME_CONTENT: 0.3,
            ActionType.IDLE: 0.0
        }
        
        return base_rewards.get(action, 0.5)
    
    def _calculate_trend_alignment(self, trends: List[str]) -> float:
        """Calculate alignment with current trends"""
        # Placeholder: return based on number of trends
        return min(len(trends) / 10, 1.0)
    
    def _calculate_audience_engagement(self, audience_state: Dict[str, Any]) -> float:
        """Calculate audience engagement level"""
        # Placeholder: extract from audience state
        return audience_state.get("engagement_level", 0.5)
    
    def _calculate_platform_health(self, platform_metrics: Optional[Dict[str, float]]) -> float:
        """Calculate platform health"""
        if not platform_metrics:
            return 0.5
        
        return platform_metrics.get("health_score", 0.5)
    
    def _calculate_agent_readiness(self, agent_state: Dict[str, Any]) -> float:
        """Calculate agent readiness"""
        return agent_state.get("readiness", 0.7)
    
    def _analyze_audience_activity(self) -> Dict[str, float]:
        """Analyze audience activity patterns"""
        # Placeholder: return mock data
        return {
            "morning": 0.3,
            "afternoon": 0.6,
            "evening": 0.9,
            "night": 0.4
        }
    
    def _assess_competition_by_time(self) -> Dict[str, float]:
        """Assess competition level by time"""
        # Placeholder: return mock data
        return {
            "morning": 0.5,
            "afternoon": 0.8,
            "evening": 0.9,
            "night": 0.3
        }
    
    def _optimize_timing(self, activity: Dict[str, float], competition: Dict[str, float]) -> datetime:
        """Optimize posting time"""
        # Simple heuristic: high activity, low competition
        scores = {}
        for time_slot in activity.keys():
            score = activity[time_slot] / (competition[time_slot] + 0.1)
            scores[time_slot] = score
        
        best_slot = max(scores, key=scores.get)
        
        # Return a datetime (placeholder: current time)
        return datetime.now()
    
    def _evaluate_partner_compatibility(self, partner_id: str) -> float:
        """Evaluate compatibility with potential partner"""
        # Placeholder: return random compatibility
        import random
        return random.uniform(0.5, 0.9)
    
    def adapt_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Adapt strategy based on performance feedback"""
        # Update exploration rate based on performance
        if performance_feedback.get("success_rate", 0) > 0.7:
            # Performing well, reduce exploration
            self.config.exploration_rate = max(0.05, self.config.exploration_rate * 0.9)
        else:
            # Not performing well, increase exploration
            self.config.exploration_rate = min(0.3, self.config.exploration_rate * 1.1)
        
        self.logger.debug(f"Strategy adapted", extra={
            "new_exploration_rate": self.config.exploration_rate,
            "success_rate": performance_feedback.get("success_rate", 0)
        })

