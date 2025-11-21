"""
Brand Agent - Marketing focused on ROI and campaign optimization
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
class BrandConfig(AgentConfig):
    """Configuration for brand agent"""
    brand_name: str = "BrandCo"
    marketing_budget: float = 10000.0
    target_demographics: Dict[str, Any] = None
    campaign_objectives: List[str] = None


@dataclass
class KPITracker:
    """Track Key Performance Indicators"""
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    revenue: float = 0.0
    
    def get_ctr(self) -> float:
        """Calculate Click-Through Rate"""
        return self.clicks / max(self.impressions, 1)
    
    def get_cpc(self) -> float:
        """Calculate Cost Per Click"""
        return self.spend / max(self.clicks, 1)
    
    def get_roi(self) -> float:
        """Calculate Return on Investment"""
        return (self.revenue - self.spend) / max(self.spend, 1)
    
    def update_metrics(self, marketing_metrics: Dict[str, Any]) -> None:
        """Update metrics from feedback"""
        self.impressions += marketing_metrics.get("impressions", 0)
        self.clicks += marketing_metrics.get("clicks", 0)
        self.conversions += marketing_metrics.get("conversions", 0)
        self.spend += marketing_metrics.get("spend", 0.0)
        self.revenue += marketing_metrics.get("revenue", 0.0)


@dataclass
class CreatorOpportunity:
    """Creator partnership opportunity"""
    creator_id: str
    follower_count: int
    engagement_rate: float
    niche: str
    estimated_cost: float
    estimated_reach: int


@dataclass
class SponsorContentAction(AgentAction):
    """Sponsor content action"""
    creator_id: str = ""
    budget_allocated: float = 0.0
    campaign_id: str = ""


@dataclass
class LaunchCampaignAction(AgentAction):
    """Launch marketing campaign action"""
    campaign_name: str = ""
    budget: float = 0.0
    target_audience: Dict[str, Any] = None
    duration_days: int = 7
    
    def __post_init__(self):
        if self.target_audience is None:
            self.target_audience = {}


class BrandAgent(BaseAgent):
    """Brand agent - focuses on marketing ROI and campaign optimization"""
    
    def __init__(self, agent_id: str, config: BrandConfig):
        super().__init__(agent_id, AgentRole.BRAND, config)
        
        self.brand_name = config.brand_name
        self.marketing_budget = config.marketing_budget
        self.remaining_budget = config.marketing_budget
        self.target_demographics = config.target_demographics or {}
        self.campaign_objectives = config.campaign_objectives or ["awareness", "conversions"]
        
        self.kpi_tracker = KPITracker()
        self.active_campaigns: Dict[str, Dict[str, Any]] = {}
        self.creator_partnerships: List[str] = []
        
        self.logger = get_logger(f"BrandAgent_{agent_id}", component="agent_brand")
        
        self.logger.info(f"BrandAgent created", extra={
            "agent_id": agent_id,
            "brand_name": self.brand_name,
            "budget": self.marketing_budget
        })
    
    async def _initialize_components(self) -> None:
        """Initialize brand-specific components"""
        # Initialize memory
        self.memory = AgentMemory(self.agent_id)
        
        # Initialize strategy head
        strategy_config = StrategyConfig(
            role="brand",
            exploration_rate=0.1,
            risk_tolerance=0.4,
            collaboration_threshold=0.7
        )
        self.strategy_head = StrategyHead(self.role, strategy_config)
        
        self.logger.info(f"BrandAgent components initialized")
    
    async def act(self, environment_state: EnvironmentState) -> AgentAction:
        """
        Brand agent action
        
        Args:
            environment_state: Current environment state
            
        Returns:
            AgentAction: Chosen action
        """
        # Store environment state
        self.last_environment_state = environment_state
        
        # Check budget
        if self.remaining_budget < 100:
            self.logger.warning(f"Low budget", extra={
                "agent_id": self.agent_id,
                "remaining_budget": self.remaining_budget
            })
            return AgentAction(
                action_type="idle",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={"reason": "insufficient_budget"}
            )
        
        # Analyze market opportunities
        creator_opportunities = self._identify_creator_partnerships(environment_state)
        
        # Decide on budget allocation
        from Agent.base.strategy_head import ActionContext, ActionType
        action_context = ActionContext(
            timestamp=datetime.now(),
            trends=environment_state.trending_topics,
            audience_state=environment_state.audience_activity,
            available_actions=[ActionType.LAUNCH_CAMPAIGN, ActionType.COLLABORATE],
            agent_state={
                "budget_remaining": self.remaining_budget,
                "active_campaigns": len(self.active_campaigns)
            },
            platform_metrics=environment_state.platform_metrics
        )
        
        strategy_decision = await self.strategy_head.decide_action(action_context)
        
        # Execute decision
        if strategy_decision.action == ActionType.COLLABORATE and creator_opportunities:
            action = await self._sponsor_content(creator_opportunities[0])
        elif strategy_decision.action == ActionType.LAUNCH_CAMPAIGN:
            action = await self._launch_campaign(environment_state)
        else:
            action = AgentAction(
                action_type="monitor",
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                parameters={"reason": "no_opportunities"}
            )
        
        self.logger.info(f"Brand action taken", extra={
            "agent_id": self.agent_id,
            "action_type": action.action_type,
            "remaining_budget": self.remaining_budget
        })
        
        return action
    
    def _identify_creator_partnerships(self, environment_state: EnvironmentState) -> List[CreatorOpportunity]:
        """Identify potential creator partnerships"""
        opportunities = []
        
        # Mock creator opportunities based on trending topics
        for i, topic in enumerate(environment_state.trending_topics[:3]):
            opportunity = CreatorOpportunity(
                creator_id=f"creator_{topic}_{i}",
                follower_count=10000 + i * 5000,
                engagement_rate=0.05 + i * 0.01,
                niche=topic,
                estimated_cost=500.0 + i * 200,
                estimated_reach=50000 + i * 20000
            )
            
            # Check if fits target demographics
            if self._fits_target_demographics(opportunity):
                opportunities.append(opportunity)
        
        # Sort by estimated ROI
        opportunities.sort(
            key=lambda x: x.estimated_reach / max(x.estimated_cost, 1),
            reverse=True
        )
        
        return opportunities
    
    def _fits_target_demographics(self, opportunity: CreatorOpportunity) -> bool:
        """Check if opportunity fits target demographics"""
        if not self.target_demographics:
            return True
        
        # Simple check: niche matching
        target_niches = self.target_demographics.get("niches", [])
        if target_niches:
            return opportunity.niche in target_niches
        
        return True
    
    async def _sponsor_content(self, opportunity: CreatorOpportunity) -> SponsorContentAction:
        """Sponsor creator content"""
        # Allocate budget
        budget_allocated = min(opportunity.estimated_cost, self.remaining_budget * 0.2)
        self.remaining_budget -= budget_allocated
        
        # Create campaign
        campaign_id = f"campaign_{self.agent_id}_{len(self.active_campaigns)}"
        self.active_campaigns[campaign_id] = {
            "creator_id": opportunity.creator_id,
            "budget": budget_allocated,
            "start_time": datetime.now(),
            "estimated_reach": opportunity.estimated_reach
        }
        
        # Add to partnerships
        if opportunity.creator_id not in self.creator_partnerships:
            self.creator_partnerships.append(opportunity.creator_id)
        
        action = SponsorContentAction(
            action_type="sponsor_content",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters={
                "niche": opportunity.niche,
                "estimated_reach": opportunity.estimated_reach
            },
            creator_id=opportunity.creator_id,
            budget_allocated=budget_allocated,
            campaign_id=campaign_id
        )
        
        self.logger.info(f"Content sponsored", extra={
            "agent_id": self.agent_id,
            "creator_id": opportunity.creator_id,
            "budget": budget_allocated,
            "campaign_id": campaign_id
        })
        
        return action
    
    async def _launch_campaign(self, environment_state: EnvironmentState) -> LaunchCampaignAction:
        """Launch marketing campaign"""
        # Determine campaign budget
        campaign_budget = min(self.remaining_budget * 0.3, 2000.0)
        self.remaining_budget -= campaign_budget
        
        # Create campaign
        campaign_id = f"campaign_{self.agent_id}_{len(self.active_campaigns)}"
        campaign_name = f"{self.brand_name}_campaign_{len(self.active_campaigns)}"
        
        self.active_campaigns[campaign_id] = {
            "name": campaign_name,
            "budget": campaign_budget,
            "start_time": datetime.now(),
            "status": "active"
        }
        
        action = LaunchCampaignAction(
            action_type="launch_campaign",
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            parameters={
                "objectives": self.campaign_objectives
            },
            campaign_name=campaign_name,
            budget=campaign_budget,
            target_audience=self.target_demographics,
            duration_days=7
        )
        
        self.logger.info(f"Campaign launched", extra={
            "agent_id": self.agent_id,
            "campaign_name": campaign_name,
            "budget": campaign_budget,
            "campaign_id": campaign_id
        })
        
        return action
    
    async def update_from_feedback(self, feedback: ActionFeedback) -> None:
        """
        Learn from marketing feedback
        
        Args:
            feedback: Feedback from action execution
        """
        # Update KPI tracker
        if "marketing_metrics" in feedback.metrics:
            self.kpi_tracker.update_metrics(feedback.metrics["marketing_metrics"])
        
        # Refine target demographics
        if feedback.audience_insights:
            self._refine_target_demographics(feedback.audience_insights)
        
        # Calculate reward based on ROI
        reward = self._calculate_reward(feedback)
        
        # Record action
        action = AgentAction(
            action_type=feedback.action_type,
            agent_id=self.agent_id,
            timestamp=feedback.timestamp,
            parameters={}
        )
        self.record_action(action, reward)
        
        # Evolve strategy
        await self.evolve_strategy()
        
        self.logger.info(f"Feedback processed", extra={
            "agent_id": self.agent_id,
            "action_type": feedback.action_type,
            "reward": reward,
            "roi": self.kpi_tracker.get_roi()
        })
    
    def _refine_target_demographics(self, audience_insights: Dict[str, Any]) -> None:
        """Refine target demographics based on insights"""
        # Update demographics with insights
        for key, value in audience_insights.items():
            if key in self.target_demographics:
                # Weighted update
                current = self.target_demographics[key]
                if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                    self.target_demographics[key] = current * 0.8 + value * 0.2
            else:
                self.target_demographics[key] = value
        
        self.logger.debug(f"Demographics refined", extra={
            "agent_id": self.agent_id,
            "updated_fields": list(audience_insights.keys())
        })
    
    def _calculate_reward(self, feedback: ActionFeedback) -> float:
        """Calculate reward based on marketing performance"""
        reward = 0.5
        
        if feedback.success:
            # Base success reward
            reward += 0.2
            
            # ROI-based reward
            if "marketing_metrics" in feedback.metrics:
                metrics = feedback.metrics["marketing_metrics"]
                revenue = metrics.get("revenue", 0)
                spend = metrics.get("spend", 1)
                roi = (revenue - spend) / max(spend, 1)
                
                # Normalize ROI to 0-0.3 range
                roi_reward = min(max(roi, 0), 1.0) * 0.3
                reward += roi_reward
        
        return min(reward, 1.0)
    
    def get_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign performance metrics"""
        if campaign_id not in self.active_campaigns:
            return {}
        
        campaign = self.active_campaigns[campaign_id]
        
        return {
            "campaign_id": campaign_id,
            "budget": campaign["budget"],
            "ctr": self.kpi_tracker.get_ctr(),
            "cpc": self.kpi_tracker.get_cpc(),
            "roi": self.kpi_tracker.get_roi(),
            "impressions": self.kpi_tracker.impressions,
            "clicks": self.kpi_tracker.clicks,
            "conversions": self.kpi_tracker.conversions
        }
    
    def replenish_budget(self, amount: float) -> None:
        """Replenish marketing budget"""
        self.remaining_budget += amount
        self.marketing_budget += amount
        
        self.logger.info(f"Budget replenished", extra={
            "agent_id": self.agent_id,
            "amount": amount,
            "new_total": self.remaining_budget
        })

