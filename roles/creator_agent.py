"""
Creator Agent - Content creator focused on engagement and growth
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from Agent.base.base_agent import BaseAgent, AgentRole, AgentConfig, EnvironmentState, AgentAction, ActionFeedback
from Agent.base.agent_memory import AgentMemory, ContentMetrics
from Agent.base.strategy_head import StrategyHead, StrategyConfig, ActionContext, ActionType
from Agent.base.content_head import ContentHead, ContentConfig, ContentContext, ContentStyle
from Agent.utils.logging_utils import get_logger


@dataclass
class CreatorConfig(AgentConfig):
    """Configuration for creator agent"""
    niche_specialty: str = "general"
    target_follower_growth: float = 0.1  # 10% per period
    content_frequency: int = 5  # posts per day
    collaboration_openness: float = 0.7


@dataclass
class ContentCalendar:
    """Content scheduling calendar"""
    scheduled_posts: List[Dict[str, Any]]
    
    def __init__(self):
        self.scheduled_posts = []
    
    def schedule_post(self, content: str, scheduled_time: datetime) -> None:
        """Schedule a post"""
        self.scheduled_posts.append({
            "content": content,
            "scheduled_time": scheduled_time,
            "status": "scheduled"
        })
    
    def get_next_post(self) -> Optional[Dict[str, Any]]:
        """Get next scheduled post"""
        if not self.scheduled_posts:
            return None
        
        # Sort by time and get earliest
        self.scheduled_posts.sort(key=lambda x: x["scheduled_time"])
        return self.scheduled_posts[0]


@dataclass
class CreateContentAction(AgentAction):
    """Content creation action"""
    content: str = ""
    hashtags: List[str] = None
    scheduled_time: datetime = None
    
    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.scheduled_time is None:
            self.scheduled_time = datetime.now()


@dataclass
class EngageAudienceAction(AgentAction):
    """Audience engagement action"""
    engagement_type: str = "like"
    target_users: List[str] = None
    
    def __post_init__(self):
        if self.target_users is None:
            self.target_users = []


@dataclass
class CollaborateAction(AgentAction):
    """Collaboration action"""
    partner_id: str = ""
    collaboration_type: str = "cross_promotion"


class CreatorAgent(BaseAgent):
    """Creator agent - focuses on content creation and audience growth"""
    
    def __init__(self, agent_id: str, config: CreatorConfig):
        super().__init__(agent_id, AgentRole.CREATOR, config)
        
        self.follower_count = 0
        self.niche_specialty = config.niche_specialty
        self.content_calendar = ContentCalendar()
        self.brand_partnerships = []
        
        self.logger = get_logger(f"CreatorAgent_{agent_id}", component="agent_creator")
        
        self.logger.info(f"CreatorAgent created", extra={
            "agent_id": agent_id,
            "niche": self.niche_specialty
        })
    
    async def _initialize_components(self) -> None:
        """Initialize creator-specific components"""
        # Initialize memory
        self.memory = AgentMemory(self.agent_id)
        
        # Initialize strategy head
        strategy_config = StrategyConfig(
            role="creator",
            exploration_rate=0.15,
            risk_tolerance=0.6,
            collaboration_threshold=0.7
        )
        self.strategy_head = StrategyHead(self.role, strategy_config)
        
        # Initialize content head
        content_config = ContentConfig(
            model_name="gpt-4",
            temperature=0.8,
            style=ContentStyle.ENTERTAINING,
            enable_hashtags=True
        )
        self.content_head = ContentHead(self.role, content_config)
        
        self.logger.info(f"CreatorAgent components initialized")
    
    async def act(self, environment_state: EnvironmentState) -> AgentAction:
        """
        Creator agent action
        
        Args:
            environment_state: Current environment state
            
        Returns:
            AgentAction: Chosen action
        """
        # Store environment state
        self.last_environment_state = environment_state
        
        # Analyze current environment
        trends = environment_state.trending_topics
        audience_state = environment_state.audience_activity
        
        # Strategy decision
        action_context = ActionContext(
            timestamp=datetime.now(),
            trends=trends,
            audience_state=audience_state,
            available_actions=[
                ActionType.CREATE_CONTENT,
                ActionType.ENGAGE_AUDIENCE,
                ActionType.COLLABORATE
            ],
            agent_state={"readiness": 0.8},
            platform_metrics=environment_state.platform_metrics
        )
        
        strategy_decision = await self.strategy_head.decide_action(action_context)
        
        # Execute decision
        if strategy_decision.action == ActionType.CREATE_CONTENT:
            action = await self._create_content(strategy_decision.parameters, environment_state)
        elif strategy_decision.action == ActionType.ENGAGE_AUDIENCE:
            action = await self._engage_audience(strategy_decision.parameters)
        elif strategy_decision.action == ActionType.COLLABORATE:
            action = await self._initiate_collaboration(strategy_decision.parameters)
        else:
            action = AgentAction(
                action_type="idle",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={}
            )
        
        self.logger.info(f"Creator action taken", extra={
            "agent_id": self.agent_id,
            "action_type": action.action_type,
            "timestamp": action.timestamp.isoformat()
        })
        
        return action
    
    async def _create_content(self, params: Dict[str, Any], environment_state: EnvironmentState) -> CreateContentAction:
        """Create content"""
        # Build content context
        content_context = ContentContext(
            target_audience=self.memory.fan_segment_memory.get_primary_audience(),
            trending_topics=environment_state.trending_topics,
            personal_brand=self.niche_specialty,
            previous_performance=self.memory.content_performance,
            content_style=ContentStyle.ENTERTAINING,
            max_length=280
        )
        
        # Generate content
        generated = await self.content_head.generate_content(content_context)
        
        # Optimize for virality
        optimized_content = await self.content_head.optimize_for_virality(
            generated.text,
            content_context.target_audience
        )
        
        # Generate hashtags
        hashtags = self.content_head.generate_hashtags(
            optimized_content,
            environment_state.trending_topics
        )
        
        # Decide timing
        timing_decision = await self.strategy_head.decide_content_timing(optimized_content)
        
        # Create action
        action = CreateContentAction(
            action_type="create_content",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters={
                "content_length": len(optimized_content),
                "hashtag_count": len(hashtags),
                "niche": self.niche_specialty
            },
            content=optimized_content,
            hashtags=hashtags,
            scheduled_time=timing_decision.scheduled_time
        )
        
        # Schedule in calendar
        self.content_calendar.schedule_post(optimized_content, timing_decision.scheduled_time)
        
        self.logger.info(f"Content created", extra={
            "agent_id": self.agent_id,
            "content_length": len(optimized_content),
            "hashtags": len(hashtags),
            "scheduled_time": timing_decision.scheduled_time.isoformat()
        })
        
        return action
    
    async def _engage_audience(self, params: Dict[str, Any]) -> EngageAudienceAction:
        """Engage with audience"""
        engagement_type = params.get("engagement_type", "like_and_comment")
        target_count = params.get("target_count", 10)
        
        action = EngageAudienceAction(
            action_type="engage_audience",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters=params,
            engagement_type=engagement_type,
            target_users=[]  # Would be populated from actual audience
        )
        
        self.logger.info(f"Engaging audience", extra={
            "agent_id": self.agent_id,
            "engagement_type": engagement_type,
            "target_count": target_count
        })
        
        return action
    
    async def _initiate_collaboration(self, params: Dict[str, Any]) -> CollaborateAction:
        """Initiate collaboration"""
        action = CollaborateAction(
            action_type="collaborate",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters=params,
            partner_id=params.get("partner_id", "unknown"),
            collaboration_type=params.get("collaboration_type", "cross_promotion")
        )
        
        self.logger.info(f"Collaboration initiated", extra={
            "agent_id": self.agent_id,
            "partner_id": action.partner_id,
            "type": action.collaboration_type
        })
        
        return action
    
    async def update_from_feedback(self, feedback: ActionFeedback) -> None:
        """
        Learn from feedback
        
        Args:
            feedback: Feedback from action execution
        """
        # Update content performance memory
        if feedback.action_type == "create_content" and feedback.content_id:
            metrics = ContentMetrics(
                content_id=feedback.content_id,
                views=feedback.engagement_metrics.get("views", 0) if feedback.engagement_metrics else 0,
                likes=feedback.engagement_metrics.get("likes", 0) if feedback.engagement_metrics else 0,
                shares=feedback.engagement_metrics.get("shares", 0) if feedback.engagement_metrics else 0,
                comments=feedback.engagement_metrics.get("comments", 0) if feedback.engagement_metrics else 0,
                engagement_rate=feedback.metrics.get("engagement_rate", 0.0),
                virality_score=feedback.metrics.get("virality_score", 0.0)
            )
            self.memory.remember_content_performance(feedback.content_id, metrics)
            
            # Adapt content style if we have audience feedback
            if feedback.audience_reaction:
                from Agent.base.content_head import AudienceFeedback
                audience_feedback = AudienceFeedback(
                    content_id=feedback.content_id,
                    likes=metrics.likes,
                    shares=metrics.shares,
                    comments=metrics.comments,
                    sentiment=feedback.audience_reaction.get("sentiment", 0.0),
                    engagement_rate=metrics.engagement_rate
                )
                await self.content_head.adapt_style(audience_feedback)
        
        # Update fan segment memory
        if feedback.audience_reaction:
            self.memory.fan_segment_memory.update_audience_preferences(feedback.audience_reaction)
        
        # Calculate reward
        reward = self._calculate_reward(feedback)
        
        # Record action and reward
        action = AgentAction(
            action_type=feedback.action_type,
            agent_id=self.agent_id,
            timestamp=feedback.timestamp,
            parameters={}
        )
        self.record_action(action, reward)
        
        # Trigger strategy evolution
        await self.evolve_strategy()
        
        self.logger.info(f"Feedback processed", extra={
            "agent_id": self.agent_id,
            "action_type": feedback.action_type,
            "reward": reward,
            "success": feedback.success
        })
    
    def _calculate_reward(self, feedback: ActionFeedback) -> float:
        """Calculate reward from feedback"""
        if not feedback.success:
            return -0.1
        
        # Base reward
        reward = 0.5
        
        # Add bonus for engagement
        if feedback.engagement_metrics:
            likes = feedback.engagement_metrics.get("likes", 0)
            shares = feedback.engagement_metrics.get("shares", 0)
            
            # Normalize to 0-1 range
            engagement_bonus = min((likes + shares * 2) / 100, 0.5)
            reward += engagement_bonus
        
        # Add bonus for virality
        if "virality_score" in feedback.metrics:
            virality_bonus = feedback.metrics["virality_score"] * 0.3
            reward += virality_bonus
        
        return min(reward, 1.0)
    
    def update_follower_count(self, new_count: int) -> None:
        """Update follower count"""
        growth = new_count - self.follower_count
        growth_rate = growth / max(self.follower_count, 1)
        
        self.follower_count = new_count
        
        self.logger.info(f"Follower count updated", extra={
            "agent_id": self.agent_id,
            "new_count": new_count,
            "growth": growth,
            "growth_rate": growth_rate
        })

