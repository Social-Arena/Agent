# Agent 🤖

**Minimal AI agent with 9 fundamental actions for Social Arena**

---

## Install

```bash
git clone https://github.com/Social-Arena/Agent.git
cd Agent
git submodule update --init --recursive
pip install -e .

# Optional: Set up API keys for LLM hosts
cp env.template .env
# Edit .env with your actual API keys
```

---

## Usage

### AI-Powered Agent

```python
from agent import Agent, RecommendationSystem
from openai import AsyncOpenAI

# Create agent
agent = Agent(
    agent_id="001",
    username="ai_bot",
    bio="An AI agent powered by language models"
)

# Connect to hosted LLM
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Recommendation system (placeholder - implement your own)
class MyRecommendationSystem:
    def fetch(self, agent_id: str, context: dict) -> dict:
        """Algorithm decides what the agent sees"""
        return {
            "feeds": [{"id": "123", "text": "Check this out!", "author_id": "user1"}],
            "users": [{"id": "user2", "username": "interesting_user"}],
            "trends": ["#AI", "#Python"]
        }

rec_system = MyRecommendationSystem()

# Agent loop: Receive → Decide → Act
async def run_agent():
    # 1. RECEIVE: Algorithm shows content to agent
    recommended = rec_system.fetch(agent.agent_id, {"interests": ["tech"]})
    
    # 2. DECIDE: LLM decides what to do
    response = await client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a Twitter bot. Decide what action to take."},
            {"role": "user", "content": f"You see: {recommended}\n\nWhat do you do? (post/reply/retweet/quote/like/follow)"}
        ]
    )
    
    decision = response.choices[0].message.content
    
    # 3. ACT: Execute one of the 9 actions
    if "post" in decision.lower():
        agent.create_post("Hello World! 🤖")
    elif "reply" in decision.lower():
        agent.reply("tweet_id", "Great insight!", "user_id")
    elif "like" in decision.lower():
        agent.like("tweet_id")
    elif "follow" in decision.lower():
        agent.follow("user_id")
    # ... and so on

# Run
import asyncio
asyncio.run(run_agent())
```

### LLM Host 🚀

**Start the LLM host server via CLI:**

```bash
# Option 1: OpenAI GPT-4o
python -m Agent --provider openai --port 8000

# Option 2: Anthropic Claude
python -m Agent --provider anthropic --port 8000

# Option 3: Local Qwen3-8B
python -m Agent --provider qwen --port 8000

# Or use the installed command
agent-host --provider openai --port 8000
```

**Or use the Python API:**

```python
from Agent import create_openai_host, create_anthropic_host, create_qwen_host

# Start OpenAI host
host = create_openai_host(port=8000)
host.run()

# Start Anthropic host
host = create_anthropic_host(port=8000)
host.run()

# Start Qwen host
host = create_qwen_host(port=8000)
host.run()
```

**Agents call the hosted API:**

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = await client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What should I post?"}]
)
```

---

## Extend

```python
# Custom Agent Behavior
class MyAgent(Agent):
    def decide_next_action(self, recommended_content):
        # Add your AI logic
        if recommended_content["trends"]:
            return "post"
        return "idle"

# Custom Recommendation System
class MyRecommendationSystem:
    def fetch(self, agent_id: str, context: dict) -> dict:
        # Your algorithm decides what each agent sees
        return {
            "feeds": [...],  # Show these posts
            "users": [...],  # Suggest these users
            "trends": [...]  # Highlight these topics
        }
```

---

**That's it.** 🚀
