"""
Example: Agent using hosted LLM API

This example shows how an agent calls the hosted API to get LLM responses.
Make sure to start one of the host servers first!

Usage:
    # Terminal 1: Start host server
    python examples/host_qwen_local.py
    
    # Terminal 2: Run this script
    python examples/agent_with_host.py
"""

import asyncio
from openai import AsyncOpenAI
from Agent import Agent


class AIAgent(Agent):
    """Agent with LLM-powered decision making"""
    
    def __init__(self, agent_id: str, username: str, api_base: str = "http://localhost:8000/v1"):
        super().__init__(agent_id=agent_id, username=username)
        self.client = AsyncOpenAI(
            base_url=api_base,
            api_key="not-needed"  # Local server doesn't require auth
        )
    
    async def decide_next_action_with_llm(self, context: dict) -> str:
        """Use LLM to decide next action"""
        
        # Build prompt from context
        prompt = f"""You are a social media AI agent named {self.username}.
        
Your current state:
- Following: {len(self.following)} users
- Followers: {len(self.followers)} users
- Liked tweets: {len(self.liked_tweets)}

Available actions:
1. "post" - Create original content
2. "engage" - Like/reply to tweets
3. "network" - Follow new users
4. "browse" - Browse feed
5. "idle" - Do nothing

Context: {context}

Based on the context, what should you do next? Respond with just one word: post, engage, network, browse, or idle."""

        # Call LLM
        response = await self.client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        
        action = response.choices[0].message.content.strip().lower()
        print(f"🤖 LLM decided: {action}")
        return action
    
    async def generate_post_content(self) -> str:
        """Use LLM to generate post content"""
        
        prompt = f"""You are {self.username}, a social media AI agent.
Generate an interesting, engaging tweet (max 280 characters).
Topic: artificial intelligence, technology, or innovation."""

        response = await self.client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.8
        )
        
        content = response.choices[0].message.content.strip()
        return content


async def main():
    """Demo agent using hosted LLM"""
    
    # Create AI agent
    agent = AIAgent(
        agent_id="agent_001",
        username="ai_bot_1"
    )
    
    print("=" * 60)
    print("🤖 AI Agent Demo with Hosted LLM")
    print("=" * 60)
    
    # Decide next action
    context = {
        "time_of_day": "morning",
        "trending_topics": ["#AI", "#MachineLearning"],
        "recent_activity": "low"
    }
    
    action = await agent.decide_next_action_with_llm(context)
    
    # Execute action
    if action == "post":
        print("\n📝 Generating post content...")
        content = await agent.generate_post_content()
        tweet = agent.create_post(content)
        print(f"✅ Posted: {tweet['text']}")
    
    elif action == "engage":
        print("\n💬 Engaging with tweets...")
        print("(Would browse and like/reply to tweets)")
    
    elif action == "network":
        print("\n🤝 Building network...")
        print("(Would discover and follow new users)")
    
    elif action == "browse":
        print("\n👀 Browsing feed...")
        feed = agent.browse_feed()
        print(f"(Retrieved {len(feed)} tweets)")
    
    else:
        print("\n😴 Agent is idle")
    
    # Show stats
    print("\n" + "=" * 60)
    print("Agent Stats:", agent.get_stats())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

