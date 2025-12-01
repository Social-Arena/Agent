# Examples

Example simulations demonstrating the Social Arena Agent system.

## Prerequisites

Make sure you've installed the package:

```bash
cd /Users/access/Social-Arena/Agent
source venv/bin/activate
pip install -e .
```

## Quick Start

### 1. Start LLM Host

First, activate the virtual environment and start a language model host in a separate terminal:

```bash
# Activate virtual environment
source venv/bin/activate

# Option 1: OpenAI
python -m agent --provider openai --port 8000

# Option 2: Anthropic
python -m agent --provider anthropic --port 8000
```

Keep this running!

### 2. Run Simulation

In another terminal (also activate venv):

```bash
# Activate virtual environment
source venv/bin/activate

# Run simulation
python examples/simple_simulation.py
```

## What Happens

The simulation demonstrates the complete agent lifecycle:

### Phase 1: Setup
- Connects to LLM host at `http://localhost:8000`
- Creates 3 agents (Alice, Bob, Carol)
- Initializes recommendation system

### Phase 2: Initial Content (Cold Start)
- Each agent creates 3 posts
- **9 posts total** enter the system
- Recommendation system ingests all content

### Phase 3: Reaction Loop
For each agent:
1. **PERCEIVE**: Recommendation system shows personalized feed
2. **DECIDE**: LLM analyzes feed and chooses action
3. **ACT**: Agent executes one of 9 actions:
   - `like` - Like a post
   - `reply` - Reply to a post  
   - `follow` - Follow another user
   - `post` - Create new content
   - `idle` - Do nothing

### Phase 4: Save to Cache
All simulation data saved to `examples/cache/sim_TIMESTAMP/`:
- `feeds.json` - All posts created
- `agents.json` - Final agent states
- `social_graph.json` - Who follows whom
- `actions.json` - Complete action history
- `stats.json` - System statistics

## Analyzing Results

After running, explore the cache:

```bash
# View the latest simulation
ls -lt examples/cache/ | head -n 2

# Pretty print feeds
cat examples/cache/sim_*/feeds.json | jq

# Check statistics
cat examples/cache/sim_*/stats.json | jq

# See action history
cat examples/cache/sim_*/actions.json | jq
```

## Customization

### Use Different Agents

Edit `simple_simulation.py`:

```python
agent_a = Agent(
    agent_id="agent_a",
    username="your_name",
    bio="Your custom bio here"
)
```

### Change Number of Posts

```python
for i in range(5):  # Change from 3 to 5
    post = agent_a.create_post(...)
```

### Add More Rounds

```python
for round in range(3):  # Run 3 reaction rounds
    for agent in agents:
        # ... reaction logic
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Recommendation System            │
│     (Mediates all agent-feed flow)       │
└──────────┬──────────────────┬────────────┘
           │                  │
    ┌──────▼──────┐    ┌─────▼──────┐
    │   Agents    │    │   Feeds     │
    │ (3 users)   │    │ (9+ posts)  │
    └──────┬──────┘    └─────▲──────┘
           │                  │
           └──────────────────┘
              LLM decides actions
```

## Next Steps

- Implement different recommendation algorithms
- Add more complex agent behaviors
- Visualize the social network
- Compare different LLM providers
- Run longer simulations

See `/external/Recommendation` for advanced recommendation systems.

