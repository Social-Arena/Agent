"""
Moderator Agent - Platform governance and rule enforcement
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from Agent.base.base_agent import BaseAgent, AgentRole, AgentConfig, EnvironmentState, AgentAction, ActionFeedback
from Agent.base.agent_memory import AgentMemory
from Agent.base.strategy_head import StrategyHead, StrategyConfig
from Agent.utils.logging_utils import get_logger


@dataclass
class ModeratorConfig(AgentConfig):
    """Configuration for moderator agent"""
    platform_rules: Dict[str, Any] = None
    toxicity_threshold: float = 0.7
    crisis_sensitivity: float = 0.8


@dataclass
class ContentClassifier:
    """Classify content for moderation"""
    
    def classify_toxicity(self, content: str) -> float:
        """Classify toxicity level (0-1)"""
        # Mock implementation - in production use ML model
        toxic_words = ["hate", "violence", "abuse", "attack"]
        toxicity = sum(1 for word in toxic_words if word in content.lower())
        return min(toxicity / len(toxic_words), 1.0)
    
    def classify_misinformation(self, content: Dict[str, Any]) -> float:
        """Classify misinformation risk (0-1)"""
        # Mock implementation
        suspicious_patterns = ["fake", "hoax", "conspiracy", "cover-up"]
        text = content.get("text", "")
        risk = sum(1 for pattern in suspicious_patterns if pattern in text.lower())
        return min(risk / len(suspicious_patterns), 1.0)
    
    def classify_spam(self, content: Dict[str, Any]) -> float:
        """Classify spam likelihood (0-1)"""
        # Mock implementation
        spam_indicators = ["click here", "buy now", "limited time", "act fast"]
        text = content.get("text", "")
        score = sum(1 for indicator in spam_indicators if indicator in text.lower())
        return min(score / len(spam_indicators), 1.0)


@dataclass
class CrisisDetector:
    """Detect potential platform crises"""
    
    def detect_potential_crises(self, environment_state: EnvironmentState) -> List[Dict[str, Any]]:
        """Detect potential crises"""
        crises = []
        
        # Check for viral negative content
        if environment_state.platform_metrics:
            negative_sentiment = environment_state.platform_metrics.get("negative_sentiment", 0)
            if negative_sentiment > 0.7:
                crises.append({
                    "type": "negative_sentiment_spike",
                    "severity": negative_sentiment,
                    "timestamp": datetime.now()
                })
        
        # Check for coordinated attacks
        trending = environment_state.trending_topics
        if any("attack" in topic.lower() or "raid" in topic.lower() for topic in trending):
            crises.append({
                "type": "coordinated_attack",
                "severity": 0.8,
                "timestamp": datetime.now()
            })
        
        return crises


@dataclass
class ModerationAction(AgentAction):
    """Content moderation action"""
    flagged_content: List[str] = None
    action_taken: str = "review"  # "remove", "warn", "review"
    
    def __post_init__(self):
        if self.flagged_content is None:
            self.flagged_content = []


@dataclass
class CrisisResponseAction(AgentAction):
    """Crisis response action"""
    crisis_type: str = "unknown"
    response_measures: List[str] = None
    
    def __post_init__(self):
        if self.response_measures is None:
            self.response_measures = []


@dataclass
class TuneExplorationAction(AgentAction):
    """Tune platform exploration parameters"""
    new_parameters: Dict[str, float] = None
    rationale: str = ""
    
    def __post_init__(self):
        if self.new_parameters is None:
            self.new_parameters = {}


class ModeratorAgent(BaseAgent):
    """Moderator agent - maintains platform health and enforces rules"""
    
    def __init__(self, agent_id: str, config: ModeratorConfig):
        super().__init__(agent_id, AgentRole.MODERATOR, config)
        
        self.platform_rules = config.platform_rules or self._default_rules()
        self.toxicity_threshold = config.toxicity_threshold
        self.crisis_sensitivity = config.crisis_sensitivity
        
        self.content_classifier = ContentClassifier()
        self.crisis_detector = CrisisDetector()
        
        # Moderation statistics
        self.moderation_stats = {
            "content_reviewed": 0,
            "content_removed": 0,
            "warnings_issued": 0,
            "crises_handled": 0
        }
        
        # Platform health metrics
        self.platform_health = {
            "toxicity_level": 0.0,
            "misinformation_level": 0.0,
            "user_satisfaction": 0.5
        }
        
        self.logger = get_logger(f"ModeratorAgent_{agent_id}", component="agent_moderator")
        
        self.logger.info(f"ModeratorAgent created", extra={
            "agent_id": agent_id,
            "toxicity_threshold": self.toxicity_threshold
        })
    
    def _default_rules(self) -> Dict[str, Any]:
        """Default platform rules"""
        return {
            "no_hate_speech": True,
            "no_misinformation": True,
            "no_spam": True,
            "no_harassment": True,
            "max_toxicity": 0.7
        }
    
    async def _initialize_components(self) -> None:
        """Initialize moderator-specific components"""
        # Initialize memory
        self.memory = AgentMemory(self.agent_id)
        
        # Initialize strategy head
        strategy_config = StrategyConfig(
            role="moderator",
            exploration_rate=0.05,  # Low exploration for consistency
            risk_tolerance=0.2  # Low risk tolerance for safety
        )
        self.strategy_head = StrategyHead(self.role, strategy_config)
        
        self.logger.info(f"ModeratorAgent components initialized")
    
    async def act(self, environment_state: EnvironmentState) -> AgentAction:
        """
        Moderator agent action
        
        Args:
            environment_state: Current environment state
            
        Returns:
            AgentAction: Chosen action
        """
        # Store environment state
        self.last_environment_state = environment_state
        
        # Scan for content violations
        flagged_content = await self._scan_for_violations(environment_state)
        
        # Detect potential crises
        potential_crises = self.crisis_detector.detect_potential_crises(environment_state)
        
        # Priority: handle crises first
        if potential_crises:
            action = await self._handle_crisis(potential_crises[0])
        elif flagged_content:
            action = await self._moderate_content(flagged_content)
        else:
            # No immediate issues, tune exploration parameters
            action = await self._tune_exploration_params(environment_state)
        
        self.logger.info(f"Moderator action taken", extra={
            "agent_id": self.agent_id,
            "action_type": action.action_type,
            "flagged_content_count": len(flagged_content),
            "crisis_count": len(potential_crises)
        })
        
        return action
    
    async def _scan_for_violations(self, environment_state: EnvironmentState) -> List[Dict[str, Any]]:
        """Scan content for policy violations"""
        flagged = []
        
        # Mock content scanning - in production would scan actual content
        if environment_state.recommended_content:
            for content in environment_state.recommended_content[:10]:
                violations = []
                
                # Check toxicity
                text = content.get("text", "")
                toxicity = self.content_classifier.classify_toxicity(text)
                if toxicity > self.toxicity_threshold:
                    violations.append("toxicity")
                
                # Check misinformation
                misinfo = self.content_classifier.classify_misinformation(content)
                if misinfo > 0.7:
                    violations.append("misinformation")
                
                # Check spam
                spam = self.content_classifier.classify_spam(content)
                if spam > 0.7:
                    violations.append("spam")
                
                if violations:
                    flagged.append({
                        "content_id": content.get("id", "unknown"),
                        "violations": violations,
                        "severity": max(toxicity, misinfo, spam)
                    })
        
        return flagged
    
    async def _moderate_content(self, flagged_content: List[Dict[str, Any]]) -> ModerationAction:
        """Take moderation action on flagged content"""
        content_ids = [c["content_id"] for c in flagged_content]
        
        # Determine action based on severity
        max_severity = max(c["severity"] for c in flagged_content)
        
        if max_severity > 0.9:
            action_taken = "remove"
            self.moderation_stats["content_removed"] += len(content_ids)
        elif max_severity > 0.7:
            action_taken = "warn"
            self.moderation_stats["warnings_issued"] += len(content_ids)
        else:
            action_taken = "review"
        
        self.moderation_stats["content_reviewed"] += len(content_ids)
        
        action = ModerationAction(
            action_type="moderate_content",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters={
                "severity": max_severity,
                "violation_types": list(set(v for c in flagged_content for v in c["violations"]))
            },
            flagged_content=content_ids,
            action_taken=action_taken
        )
        
        self.logger.warning(f"Content moderated", extra={
            "agent_id": self.agent_id,
            "content_count": len(content_ids),
            "action_taken": action_taken,
            "max_severity": max_severity
        })
        
        return action
    
    async def _handle_crisis(self, crisis: Dict[str, Any]) -> CrisisResponseAction:
        """Handle platform crisis"""
        crisis_type = crisis["type"]
        severity = crisis["severity"]
        
        # Determine response measures
        response_measures = []
        
        if severity > 0.8:
            response_measures.extend([
                "increase_moderation",
                "rate_limit_content",
                "alert_admin"
            ])
        elif severity > 0.6:
            response_measures.extend([
                "increase_monitoring",
                "review_trending"
            ])
        else:
            response_measures.append("monitor_situation")
        
        self.moderation_stats["crises_handled"] += 1
        
        action = CrisisResponseAction(
            action_type="crisis_response",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters={
                "severity": severity
            },
            crisis_type=crisis_type,
            response_measures=response_measures
        )
        
        self.logger.critical(f"Crisis handled", extra={
            "agent_id": self.agent_id,
            "crisis_type": crisis_type,
            "severity": severity,
            "measures": response_measures
        })
        
        return action
    
    async def _tune_exploration_params(self, environment_state: EnvironmentState) -> TuneExplorationAction:
        """Tune platform exploration parameters"""
        # Calculate platform health
        platform_health = self._calculate_platform_health(environment_state)
        
        # Adjust exploration based on health
        new_parameters = {}
        
        if platform_health < 0.5:
            # Low health, reduce exploration
            new_parameters["exploration_rate"] = 0.05
            new_parameters["diversity_weight"] = 0.3
            rationale = "Reducing exploration due to low platform health"
        elif platform_health > 0.8:
            # High health, increase exploration
            new_parameters["exploration_rate"] = 0.15
            new_parameters["diversity_weight"] = 0.5
            rationale = "Increasing exploration due to good platform health"
        else:
            # Maintain current settings
            new_parameters["exploration_rate"] = 0.1
            new_parameters["diversity_weight"] = 0.4
            rationale = "Maintaining current exploration parameters"
        
        action = TuneExplorationAction(
            action_type="tune_exploration",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters={
                "platform_health": platform_health
            },
            new_parameters=new_parameters,
            rationale=rationale
        )
        
        self.logger.info(f"Exploration parameters tuned", extra={
            "agent_id": self.agent_id,
            "platform_health": platform_health,
            "new_params": new_parameters
        })
        
        return action
    
    def _calculate_platform_health(self, environment_state: EnvironmentState) -> float:
        """Calculate overall platform health (0-1)"""
        health = 0.5  # Base health
        
        if environment_state.platform_metrics:
            # Factor in various metrics
            negative_sentiment = environment_state.platform_metrics.get("negative_sentiment", 0.3)
            engagement = environment_state.platform_metrics.get("engagement", 0.5)
            
            # Lower negative sentiment = better health
            health += (1 - negative_sentiment) * 0.3
            
            # Higher engagement = better health
            health += engagement * 0.2
        
        # Factor in moderation statistics
        if self.moderation_stats["content_reviewed"] > 0:
            violation_rate = self.moderation_stats["content_removed"] / self.moderation_stats["content_reviewed"]
            health -= violation_rate * 0.2
        
        return max(0.0, min(1.0, health))
    
    async def update_from_feedback(self, feedback: ActionFeedback) -> None:
        """
        Learn from moderation feedback
        
        Args:
            feedback: Feedback from action execution
        """
        # Update platform health metrics
        if "platform_health" in feedback.metrics:
            for key, value in feedback.metrics["platform_health"].items():
                self.platform_health[key] = value
        
        # Calculate reward based on platform improvement
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
            "reward": reward,
            "platform_health": self._calculate_platform_health(self.last_environment_state) if self.last_environment_state else 0.5
        })
    
    def _calculate_reward(self, feedback: ActionFeedback) -> float:
        """Calculate reward based on moderation effectiveness"""
        reward = 0.5
        
        if feedback.success:
            # Base success reward
            reward += 0.2
            
            # Platform health improvement
            if "platform_health" in feedback.metrics:
                health_improvement = feedback.metrics["platform_health"].get("improvement", 0)
                reward += health_improvement * 0.3
        
        return min(reward, 1.0)
    
    def get_moderation_report(self) -> Dict[str, Any]:
        """Generate moderation report"""
        return {
            "agent_id": self.agent_id,
            "statistics": self.moderation_stats.copy(),
            "platform_health": self.platform_health.copy(),
            "rules": self.platform_rules,
            "timestamp": datetime.now().isoformat()
        }

