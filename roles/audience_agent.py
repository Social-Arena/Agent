"""
Audience Agent - Simulates real user behavior for content consumption
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from Agent.base.base_agent import BaseAgent, AgentRole, AgentConfig, EnvironmentState, AgentAction, ActionFeedback
from Agent.base.agent_memory import AgentMemory, Interaction
from Agent.base.strategy_head import StrategyHead, StrategyConfig
from Agent.utils.logging_utils import get_logger


@dataclass
class AudienceConfig(AgentConfig):
    """Configuration for audience agent"""
    interests: List[str] = None
    daily_time_budget: int = 60  # minutes
    drift_rate: float = 0.05  # preference drift rate
    social_proof_sensitivity: float = 0.7


@dataclass
class PreferenceDrift:
    """Models preference changes over time"""
    drift_rate: float
    current_preferences: Dict[str, float]
    
    def __init__(self, drift_rate: float = 0.05):
        self.drift_rate = drift_rate
        self.current_preferences = {}
    
    def update_preferences(self, feedback: ActionFeedback) -> None:
        """Update preferences based on feedback"""
        if feedback.engagement_metrics:
            for key, value in feedback.engagement_metrics.items():
                if isinstance(value, (int, float)):
                    current = self.current_preferences.get(key, 0.5)
                    # Drift towards observed value
                    self.current_preferences[key] = current + self.drift_rate * (value - current)


@dataclass
class EngageWithContentAction(AgentAction):
    """Content engagement action"""
    content_id: str = ""
    engagement_type: str = "view"  # "view", "like", "share", "comment"
    dwell_time: float = 0.0  # seconds


@dataclass
class IdleAction(AgentAction):
    """Idle action when no content interests the agent"""
    reason: str = "no_reason"


class AudienceAgent(BaseAgent):
    """Audience agent - simulates realistic user behavior"""
    
    def __init__(self, agent_id: str, config: AudienceConfig):
        super().__init__(agent_id, AgentRole.AUDIENCE, config)
        
        self.interests = config.interests or ["general"]
        self.time_budget = config.daily_time_budget * 60  # convert to seconds
        self.time_spent_today = 0
        self.preference_drift = PreferenceDrift(config.drift_rate)
        self.social_proof_sensitivity = config.social_proof_sensitivity
        
        # Social network
        self.friends: List[str] = []
        self.friend_activities: Dict[str, List[Any]] = {}
        
        self.logger = get_logger(f"AudienceAgent_{agent_id}", component="agent_audience")
        
        self.logger.info(f"AudienceAgent created", extra={
            "agent_id": agent_id,
            "interests": self.interests,
            "time_budget": self.time_budget
        })
    
    async def _initialize_components(self) -> None:
        """Initialize audience-specific components"""
        # Initialize memory
        self.memory = AgentMemory(self.agent_id)
        
        # Initialize strategy head
        strategy_config = StrategyConfig(
            role="audience",
            exploration_rate=0.2,
            risk_tolerance=0.3
        )
        self.strategy_head = StrategyHead(self.role, strategy_config)
        
        self.logger.info(f"AudienceAgent components initialized")
    
    async def act(self, environment_state: EnvironmentState) -> AgentAction:
        """
        Audience agent action
        
        Args:
            environment_state: Current environment state
            
        Returns:
            AgentAction: Chosen action
        """
        # Store environment state
        self.last_environment_state = environment_state
        
        # Check time budget
        if self.time_spent_today >= self.time_budget:
            return IdleAction(
                action_type="idle",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={},
                reason="time_budget_exceeded"
            )
        
        # Get available content
        available_content = environment_state.recommended_content
        
        if not available_content:
            return IdleAction(
                action_type="idle",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={},
                reason="no_content_available"
            )
        
        # Analyze social proof
        social_signals = self._analyze_social_proof(available_content, environment_state)
        
        # Make probabilistic content selection
        selected_content = self._make_probabilistic_choice(available_content, social_signals)
        
        if selected_content:
            # Decide engagement type
            engagement_type = self._decide_engagement_type(selected_content, social_signals)
            
            # Estimate dwell time
            dwell_time = self._estimate_dwell_time(selected_content)
            
            action = EngageWithContentAction(
                action_type="engage_content",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={
                    "interests": self.interests,
                    "social_influence": social_signals.get("influence_score", 0)
                },
                content_id=selected_content.get("id", "unknown"),
                engagement_type=engagement_type,
                dwell_time=dwell_time
            )
            
            # Update time spent
            self.time_spent_today += dwell_time
            
            self.logger.info(f"Content engaged", extra={
                "agent_id": self.agent_id,
                "content_id": selected_content.get("id"),
                "engagement_type": engagement_type,
                "dwell_time": dwell_time
            })
            
            return action
        else:
            return IdleAction(
                action_type="idle",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={},
                reason="no_interesting_content"
            )
    
    def _analyze_social_proof(self, content_list: List[Any], environment_state: EnvironmentState) -> Dict[str, Any]:
        """Analyze social proof signals"""
        social_signals = {}
        
        # Check friend activities
        if environment_state.social_signals:
            friend_liked = environment_state.social_signals.get("friend_likes", [])
            friend_shared = environment_state.social_signals.get("friend_shares", [])
            
            social_signals["friend_engagement"] = len(friend_liked) + len(friend_shared)
            social_signals["influence_score"] = min(len(friend_liked) * 0.1, 1.0)
        else:
            social_signals["friend_engagement"] = 0
            social_signals["influence_score"] = 0
        
        return social_signals
    
    def _make_probabilistic_choice(
        self,
        content_list: List[Any],
        social_signals: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Make probabilistic content selection
        
        Args:
            content_list: Available content
            social_signals: Social proof signals
            
        Returns:
            Selected content or None
        """
        if not content_list:
            return None
        
        # Calculate scores for each content
        scores = []
        for content in content_list:
            score = self._calculate_content_score(content, social_signals)
            scores.append((content, score))
        
        # Normalize scores to probabilities
        total_score = sum(s[1] for s in scores)
        if total_score == 0:
            return None
        
        probabilities = [(c, s / total_score) for c, s in scores]
        
        # Probabilistic selection
        rand_val = random.random()
        cumulative = 0
        
        for content, prob in probabilities:
            cumulative += prob
            if rand_val <= cumulative:
                return content
        
        # Fallback to first item
        return content_list[0] if content_list else None
    
    def _calculate_content_score(self, content: Dict[str, Any], social_signals: Dict[str, Any]) -> float:
        """Calculate content attractiveness score"""
        score = 0.5  # Base score
        
        # Interest matching
        content_topics = content.get("topics", [])
        interest_match = len(set(content_topics) & set(self.interests)) / max(len(self.interests), 1)
        score += interest_match * 0.3
        
        # Social proof influence
        social_influence = social_signals.get("influence_score", 0)
        score += social_influence * self.social_proof_sensitivity * 0.3
        
        # Content popularity
        engagement = content.get("engagement_count", 0)
        popularity_bonus = min(engagement / 1000, 0.2)
        score += popularity_bonus
        
        return min(score, 1.0)
    
    def _decide_engagement_type(self, content: Dict[str, Any], social_signals: Dict[str, Any]) -> str:
        """Decide what type of engagement to perform"""
        # Probabilistic engagement based on content quality and social signals
        rand = random.random()
        
        # Higher engagement for highly rated content
        content_quality = content.get("quality_score", 0.5)
        social_boost = social_signals.get("influence_score", 0)
        
        engagement_threshold = content_quality + social_boost * 0.2
        
        if rand < 0.1 * engagement_threshold:
            return "share"  # Rare, high commitment
        elif rand < 0.3 * engagement_threshold:
            return "comment"  # Medium commitment
        elif rand < 0.6 * engagement_threshold:
            return "like"  # Common, low commitment
        else:
            return "view"  # Just viewing
    
    def _estimate_dwell_time(self, content: Dict[str, Any]) -> float:
        """Estimate how long to spend on content (seconds)"""
        # Base time on content type and length
        content_length = content.get("length", 100)
        
        # Simple heuristic: ~3 seconds per 100 characters
        base_time = (content_length / 100) * 3
        
        # Add randomness
        dwell_time = base_time * random.uniform(0.5, 1.5)
        
        # Clip to reasonable range
        return max(1.0, min(dwell_time, 60.0))
    
    async def update_from_feedback(self, feedback: ActionFeedback) -> None:
        """
        Learn from feedback
        
        Args:
            feedback: Feedback from action execution
        """
        # Update preference drift
        self.preference_drift.update_preferences(feedback)
        
        # Social learning
        if feedback.social_context:
            self._update_social_preferences(feedback.social_context)
        
        # Remember interaction
        if feedback.content_id:
            interaction = Interaction(
                interaction_id=f"int_{self.agent_id}_{feedback.content_id}",
                interaction_type=feedback.action_type,
                timestamp=feedback.timestamp,
                participants=[self.agent_id, feedback.content_id],
                outcome=feedback.metrics,
                context=feedback.social_context or {}
            )
            self.memory.remember_interaction(interaction)
        
        # Calculate reward
        reward = self._calculate_reward(feedback)
        
        # Record action
        action = AgentAction(
            action_type=feedback.action_type,
            agent_id=self.agent_id,
            timestamp=feedback.timestamp,
            parameters={}
        )
        self.record_action(action, reward)
        
        self.logger.info(f"Feedback processed", extra={
            "agent_id": self.agent_id,
            "action_type": feedback.action_type,
            "reward": reward
        })
    
    def _update_social_preferences(self, social_context: Dict[str, Any]) -> None:
        """Update preferences based on social context"""
        # Learn from friend activities
        friend_preferences = social_context.get("friend_preferences", {})
        
        for preference, value in friend_preferences.items():
            if isinstance(value, (int, float)):
                current = self.preference_drift.current_preferences.get(preference, 0.5)
                # Social influence on preferences
                influence = self.social_proof_sensitivity * 0.1
                self.preference_drift.current_preferences[preference] = \
                    current + influence * (value - current)
    
    def _calculate_reward(self, feedback: ActionFeedback) -> float:
        """Calculate reward from feedback"""
        # Audience agents get reward from content quality
        reward = 0.5
        
        if feedback.success:
            # Positive reward for successful engagement
            reward += 0.2
            
            # Bonus for content matching interests
            if feedback.metrics.get("interest_match", 0) > 0.7:
                reward += 0.2
        
        return min(reward, 1.0)
    
    def reset_daily_budget(self) -> None:
        """Reset daily time budget"""
        self.time_spent_today = 0
        self.logger.info(f"Daily budget reset", extra={
            "agent_id": self.agent_id
        })
    
    def add_friend(self, friend_id: str) -> None:
        """Add a friend to social network"""
        if friend_id not in self.friends:
            self.friends.append(friend_id)
            self.logger.info(f"Friend added", extra={
                "agent_id": self.agent_id,
                "friend_id": friend_id,
                "total_friends": len(self.friends)
            })

