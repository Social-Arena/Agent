"""
Simple Social Arena Simulation
==============================

Demonstrates:
- 3 agents creating initial posts
- Recommendation system mediating content
- Agents reacting to recommended feeds using LLM
- Saving all simulation data to cache

Usage:
    python examples/simple_simulation.py
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

from agent import Agent
from openai import AsyncOpenAI

# Load environment variables from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# ================================================================
# SIMPLE RECOMMENDATION SYSTEM
# ================================================================

class SimpleRecommendationSystem:
    """
    Simple recommendation system for demonstration
    
    Shows all agents all feeds (no filtering).
    In production, implement personalization algorithms.
    """
    
    def __init__(self):
        self.feed_pool: List[Dict[str, Any]] = []
        self.agent_pool: Dict[str, Agent] = {}
        self.social_graph: Dict[str, List[str]] = {}
        self.action_history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent: Agent):
        """Register an agent in the system"""
        self.agent_pool[agent.agent_id] = agent
        self.social_graph[agent.agent_id] = []
    
    def ingest_feed(self, feed: Dict[str, Any]) -> None:
        """Add new feed to the pool"""
        self.feed_pool.append(feed)
        print(f"  📥 Ingested feed {feed['id'][:8]}... by {feed['author_id']}")
    
    def fetch(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized content for agent"""
        # Simple: show all feeds except agent's own
        other_feeds = [
            f for f in self.feed_pool 
            if f['author_id'] != agent_id
        ]
        
        # Sort by recency
        other_feeds.sort(key=lambda f: f.get('created_at', ''), reverse=True)
        
        # Get other users
        other_users = [
            uid for uid in self.agent_pool.keys() 
            if uid != agent_id
        ]
        
        return {
            "feeds": other_feeds[:5],  # Top 5 recent feeds
            "users": other_users,
            "trends": []
        }
    
    def record_action(
        self, 
        agent_id: str, 
        action: str, 
        target_id: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record agent action"""
        action_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "action": action,
            "target_id": target_id,
            "metadata": metadata or {}
        }
        self.action_history.append(action_record)
        
        # Update social graph for follow actions
        if action == "follow" and target_id not in self.social_graph[agent_id]:
            self.social_graph[agent_id].append(target_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_feeds": len(self.feed_pool),
            "total_agents": len(self.agent_pool),
            "total_actions": len(self.action_history),
            "network_edges": sum(len(follows) for follows in self.social_graph.values())
        }


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def format_feeds(feeds: List[Dict[str, Any]]) -> str:
    """Format feeds for LLM prompt"""
    lines = []
    for i, feed in enumerate(feeds, 1):
        lines.append(f"{i}. [{feed['id'][:8]}] @{feed['author_id']}: {feed['text']}")
    return "\n".join(lines)


def parse_decision(llm_response: str) -> Dict[str, Any]:
    """Parse LLM decision into action dict"""
    response_lower = llm_response.lower().strip()
    
    # Extract action and target
    if "like" in response_lower:
        # Extract feed ID if present
        parts = response_lower.split()
        target_id = parts[1] if len(parts) > 1 else None
        return {"action": "like", "target_id": target_id}
    
    elif "reply" in response_lower:
        parts = response_lower.split()
        target_id = parts[1] if len(parts) > 1 else None
        return {"action": "reply", "target_id": target_id}
    
    elif "follow" in response_lower:
        parts = response_lower.split()
        target_id = parts[1] if len(parts) > 1 else None
        return {"action": "follow", "target_id": target_id}
    
    elif "post" in response_lower:
        return {"action": "post"}
    
    else:
        return {"action": "idle"}


def find_feed_by_id(feeds: List[Dict[str, Any]], feed_id: str) -> Dict[str, Any]:
    """Find feed by ID (supports partial match)"""
    for feed in feeds:
        if feed['id'].startswith(feed_id) or feed['id'] == feed_id:
            return feed
    return feeds[0] if feeds else None  # Fallback to first


# ================================================================
# MAIN SIMULATION
# ================================================================

async def main():
    print("=" * 60)
    print("SOCIAL ARENA SIMULATION")
    print("=" * 60)
    
    # ============================================================
    # PHASE 1: SETUP
    # ============================================================
    print("\n📋 PHASE 1: Setup")
    print("-" * 60)
    
    # Create LLM client (assumes host is already running)
    print("Connecting to LLM host at http://localhost:8000...")
    llm_client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )
    
    # Create agents
    print("\nCreating agents...")
    agent_a = Agent(
        agent_id="agent_a",
        username="alice",
        bio="Tech enthusiast who loves Python"
    )
    
    agent_b = Agent(
        agent_id="agent_b",
        username="bob",
        bio="Crypto investor tracking Bitcoin"
    )
    
    agent_c = Agent(
        agent_id="agent_c",
        username="carol",
        bio="AI researcher working on LLMs"
    )
    
    agents = [agent_a, agent_b, agent_c]
    print(f"✓ Created {len(agents)} agents")
    
    # Initialize recommendation system
    print("\nInitializing recommendation system...")
    rec_system = SimpleRecommendationSystem()
    for agent in agents:
        rec_system.register_agent(agent)
    print(f"✓ Registered {len(agents)} agents")
    
    # ============================================================
    # PHASE 2 & 3: 30-DAY SIMULATION
    # ============================================================
    print("\n\n🗓️  PHASE 2-3: 30-Day Simulation")
    print("-" * 60)
    
    num_days = 30
    posts_per_agent_per_day = 3
    
    for day in range(1, num_days + 1):
        print(f"\n{'='*60}")
        print(f"DAY {day}/{num_days}")
        print(f"{'='*60}")
        
        # STEP 1: Each agent creates 3 posts
        print(f"\n📝 Morning: Agents create posts...")
        for agent in agents:
            for post_num in range(posts_per_agent_per_day):
                post_text = f"Day {day} Post {post_num+1}: {agent.bio.split()[0]} thoughts!"
                post = agent.create_post(post_text)
                rec_system.ingest_feed(post)
                print(f"  📥 @{agent.username} posted")
        
        print(f"  ✓ {len(agents) * posts_per_agent_per_day} posts created today")
        print(f"  📊 Total feeds in system: {len(rec_system.feed_pool)}")
        
        # STEP 2: Each agent reacts to all posts
        print(f"\n🤖 Afternoon: Agents react to content...")
        for agent in agents:
            # Get all feeds (not just top 5)
            recommended = rec_system.fetch(agent.agent_id, {"interests": agent.bio})
            
            if not recommended['feeds']:
                print(f"  ⏭️  @{agent.username}: No feeds to react to")
                continue
            
            # DECIDE: Ask LLM what to do
            decision_prompt = f"""You are {agent.username} ({agent.bio}).

You see these recent posts:
{format_feeds(recommended['feeds'][:5])}

What do you want to do? Choose ONE simple action:
- like [feed_id] - Like a post
- reply [feed_id] - Reply to a post
- follow [username] - Follow someone
- idle - Do nothing

Respond with just the action and ID, like: "like {recommended['feeds'][0]['id'][:8]}"
"""
            
            response = await llm_client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": "You are a social media user. Be concise."},
                    {"role": "user", "content": decision_prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            decision = parse_decision(response.choices[0].message.content)
            
            # ACT: Execute the action
            if decision['action'] == 'like':
                target = find_feed_by_id(recommended['feeds'], decision.get('target_id', ''))
                if target:
                    agent.like(target['id'])
                    rec_system.record_action(agent.agent_id, "like", target['id'])
                    print(f"  ❤️  @{agent.username} liked @{target['author_id']}'s post")
            
            elif decision['action'] == 'reply':
                target = find_feed_by_id(recommended['feeds'], decision.get('target_id', ''))
                if target:
                    reply_text = f"Interesting perspective on {target['text'][:20]}..."
                    reply = agent.reply(target['id'], reply_text, target['author_id'])
                    rec_system.ingest_feed(reply)
                    rec_system.record_action(agent.agent_id, "reply", target['id'])
                    print(f"  💬 @{agent.username} replied to @{target['author_id']}")
            
            elif decision['action'] == 'follow':
                if recommended['users']:
                    target_user = recommended['users'][0]
                    if agent.follow(target_user):
                        rec_system.record_action(agent.agent_id, "follow", target_user)
                        print(f"  ➕ @{agent.username} followed @{target_user}")
            
            else:
                print(f"  😴 @{agent.username} idle")
        
        # Daily summary
        print(f"\n📈 End of Day {day}:")
        print(f"  Total feeds: {len(rec_system.feed_pool)}")
        print(f"  Total actions: {len(rec_system.action_history)}")
    
    # ============================================================
    # PHASE 4: SAVE TO CACHE
    # ============================================================
    print("\n\n💾 PHASE 4: Saving to Cache")
    print("-" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = Path("examples/cache") / f"sim_{timestamp}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feeds (convert Pydantic models to dicts)
    feeds_data = []
    for feed in rec_system.feed_pool:
        if hasattr(feed, 'model_dump'):
            feeds_data.append(feed.model_dump())
        elif isinstance(feed, dict):
            # Handle dict feeds - convert nested Pydantic objects
            feed_dict = feed.copy()
            if 'entities' in feed_dict and hasattr(feed_dict['entities'], 'model_dump'):
                feed_dict['entities'] = feed_dict['entities'].model_dump()
            feeds_data.append(feed_dict)
        else:
            feeds_data.append(feed)
    
    with open(cache_dir / "feeds.json", "w") as f:
        json.dump(feeds_data, f, indent=2)
    print(f"✓ Saved {len(rec_system.feed_pool)} feeds")
    
    # Save agents
    agents_data = {}
    for agent in agents:
        agents_data[agent.agent_id] = {
            "username": agent.username,
            "bio": agent.bio,
            "following": agent.following,
            "followers": agent.followers,
            "liked_tweets": agent.liked_tweets,
            "stats": agent.get_stats()
        }
    
    with open(cache_dir / "agents.json", "w") as f:
        json.dump(agents_data, f, indent=2)
    print(f"✓ Saved {len(agents)} agent states")
    
    # Save social graph
    with open(cache_dir / "social_graph.json", "w") as f:
        json.dump(rec_system.social_graph, f, indent=2)
    print(f"✓ Saved social graph")
    
    # Save actions
    with open(cache_dir / "actions.json", "w") as f:
        json.dump(rec_system.action_history, f, indent=2)
    print(f"✓ Saved {len(rec_system.action_history)} actions")
    
    # Save stats
    with open(cache_dir / "stats.json", "w") as f:
        json.dump(rec_system.get_stats(), f, indent=2)
    print(f"✓ Saved system statistics")
    
    print(f"\n📁 All data saved to: {cache_dir}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    stats = rec_system.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total feeds: {stats['total_feeds']}")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Network connections: {stats['network_edges']}")
    
    print(f"\nAgent Summaries:")
    for agent in agents:
        summary = agent.get_stats()
        print(f"  @{agent.username}:")
        print(f"    Following: {summary['following_count']}")
        print(f"    Likes: {summary['liked_tweets_count']}")
    
    print(f"\n💡 Next steps:")
    print(f"  - Analyze data in {cache_dir}")
    print(f"  - Run visualization tools on the cache")
    print(f"  - Compare with other simulation runs")


if __name__ == "__main__":
    asyncio.run(main())

