"""
Simple Agent Example - Demonstrates basic agent functionality
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Agent import (
    CreatorAgent,
    CreatorConfig,
    AudienceAgent,
    AudienceConfig,
    BrandAgent,
    BrandConfig,
    ModeratorAgent,
    ModeratorConfig,
    EnvironmentState,
    ActionFeedback,
    LearningStage
)


async def main():
    """Main example function"""
    
    print("=" * 80)
    print("Agent System - Simple Example")
    print("=" * 80)
    print()
    
    # 1. Create Creator Agent
    print("1. Creating Creator Agent...")
    creator_config = CreatorConfig(
        agent_id="creator_001",
        role="creator",
        learning_stage=LearningStage.COLD_START,
        niche_specialty="technology"
    )
    creator = CreatorAgent("creator_001", creator_config)
    await creator.initialize()
    print(f"✓ Creator agent created: {creator.agent_id}")
    print()
    
    # 2. Create Audience Agent
    print("2. Creating Audience Agent...")
    audience_config = AudienceConfig(
        agent_id="audience_001",
        role="audience",
        interests=["technology", "AI", "programming"],
        daily_time_budget=60
    )
    audience = AudienceAgent("audience_001", audience_config)
    await audience.initialize()
    print(f"✓ Audience agent created: {audience.agent_id}")
    print()
    
    # 3. Create Brand Agent
    print("3. Creating Brand Agent...")
    brand_config = BrandConfig(
        agent_id="brand_001",
        role="brand",
        brand_name="TechCorp",
        marketing_budget=5000.0
    )
    brand = BrandAgent("brand_001", brand_config)
    await brand.initialize()
    print(f"✓ Brand agent created: {brand.agent_id}")
    print()
    
    # 4. Create Moderator Agent
    print("4. Creating Moderator Agent...")
    moderator_config = ModeratorConfig(
        agent_id="moderator_001",
        role="moderator",
        toxicity_threshold=0.7
    )
    moderator = ModeratorAgent("moderator_001", moderator_config)
    await moderator.initialize()
    print(f"✓ Moderator agent created: {moderator.agent_id}")
    print()
    
    # 5. Simulate environment state
    print("5. Simulating Environment State...")
    environment_state = EnvironmentState(
        timestamp=datetime.now(),
        trending_topics=["AI", "MachineLearning", "TechNews"],
        audience_activity={"engagement_level": 0.7, "active_users": 1000},
        platform_metrics={"health_score": 0.8, "negative_sentiment": 0.2},
        recommended_content=[
            {"id": "content_001", "text": "Amazing AI breakthrough!", "topics": ["AI"], "quality_score": 0.8, "length": 100},
            {"id": "content_002", "text": "New tech trends for 2024", "topics": ["Tech"], "quality_score": 0.7, "length": 120},
        ]
    )
    print("✓ Environment state created")
    print()
    
    # 6. Run agent actions
    print("6. Running Agent Actions...")
    print()
    
    # Creator creates content
    print("   6.1. Creator Agent Action:")
    creator_action = await creator.act(environment_state)
    print(f"   ✓ Action type: {creator_action.action_type}")
    print(f"   ✓ Timestamp: {creator_action.timestamp}")
    if hasattr(creator_action, 'content'):
        print(f"   ✓ Content preview: {creator_action.content[:50]}...")
    print()
    
    # Audience consumes content
    print("   6.2. Audience Agent Action:")
    audience_action = await audience.act(environment_state)
    print(f"   ✓ Action type: {audience_action.action_type}")
    if hasattr(audience_action, 'engagement_type'):
        print(f"   ✓ Engagement type: {audience_action.engagement_type}")
        print(f"   ✓ Dwell time: {audience_action.dwell_time:.1f}s")
    print()
    
    # Brand launches campaign
    print("   6.3. Brand Agent Action:")
    brand_action = await brand.act(environment_state)
    print(f"   ✓ Action type: {brand_action.action_type}")
    print(f"   ✓ Remaining budget: ${brand.remaining_budget:.2f}")
    print()
    
    # Moderator checks platform health
    print("   6.4. Moderator Agent Action:")
    moderator_action = await moderator.act(environment_state)
    print(f"   ✓ Action type: {moderator_action.action_type}")
    print()
    
    # 7. Provide feedback to agents
    print("7. Providing Feedback to Agents...")
    
    # Creator feedback
    creator_feedback = ActionFeedback(
        action_id="action_001",
        action_type="create_content",
        success=True,
        metrics={"engagement_rate": 0.05, "virality_score": 0.7},
        timestamp=datetime.now(),
        content_id="content_001",
        engagement_metrics={"likes": 150, "shares": 45, "comments": 23, "views": 3000},
        audience_reaction={"sentiment": 0.8, "topics": ["AI", "innovation"]}
    )
    await creator.update_from_feedback(creator_feedback)
    print("✓ Creator feedback processed")
    
    # Audience feedback
    audience_feedback = ActionFeedback(
        action_id="action_002",
        action_type="engage_content",
        success=True,
        metrics={"interest_match": 0.9, "satisfaction": 0.8},
        timestamp=datetime.now(),
        content_id="content_001"
    )
    await audience.update_from_feedback(audience_feedback)
    print("✓ Audience feedback processed")
    
    # Brand feedback
    brand_feedback = ActionFeedback(
        action_id="action_003",
        action_type="launch_campaign",
        success=True,
        metrics={
            "marketing_metrics": {
                "impressions": 10000,
                "clicks": 300,
                "conversions": 15,
                "spend": 500.0,
                "revenue": 1500.0
            }
        },
        timestamp=datetime.now()
    )
    await brand.update_from_feedback(brand_feedback)
    print("✓ Brand feedback processed")
    print()
    
    # 8. Get agent states
    print("8. Agent States:")
    print()
    
    creator_state = creator.get_current_state()
    print(f"   Creator Agent:")
    print(f"   - Total actions: {creator_state.performance_metrics.total_actions}")
    print(f"   - Average reward: {creator_state.performance_metrics.average_reward:.3f}")
    print(f"   - Learning stage: {creator_state.learning_stage.value}")
    print()
    
    audience_state = audience.get_current_state()
    print(f"   Audience Agent:")
    print(f"   - Total actions: {audience_state.performance_metrics.total_actions}")
    print(f"   - Time spent today: {audience.time_spent_today:.1f}s")
    print()
    
    brand_state = brand.get_current_state()
    print(f"   Brand Agent:")
    print(f"   - Total actions: {brand_state.performance_metrics.total_actions}")
    print(f"   - ROI: {brand.kpi_tracker.get_roi():.2f}")
    print(f"   - CTR: {brand.kpi_tracker.get_ctr():.4f}")
    print()
    
    moderator_state = moderator.get_current_state()
    print(f"   Moderator Agent:")
    print(f"   - Total actions: {moderator_state.performance_metrics.total_actions}")
    print(f"   - Content reviewed: {moderator.moderation_stats['content_reviewed']}")
    print()
    
    print("=" * 80)
    print("Example Completed Successfully!")
    print("=" * 80)
    print()
    print("Check the 'trace/' directory for detailed logs.")
    print()


if __name__ == "__main__":
    asyncio.run(main())

