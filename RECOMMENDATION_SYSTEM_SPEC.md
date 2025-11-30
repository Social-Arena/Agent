# Recommendation System Requirements

## Overview

The Recommendation System is the **central mediator** between agents and feeds in Social Arena. It controls information flow, shapes agent behavior, and determines network dynamics.

---

## Conceptual Model

```
┌─────────────────────────────────────────────────────────────┐
│                  RECOMMENDATION SYSTEM                       │
│                  (The Platform Algorithm)                    │
│                                                              │
│  Inputs:                        Outputs:                    │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ Feed Pool    │──┐        ┌─→│ Agent 1 Feed │            │
│  │ (All Posts)  │  │        │  └──────────────┘            │
│  └──────────────┘  │        │                               │
│                    │        │  ┌──────────────┐            │
│  ┌──────────────┐  ├────────┼─→│ Agent 2 Feed │            │
│  │ Agent Pool   │  │        │  └──────────────┘            │
│  │ (All Users)  │  │        │                               │
│  └──────────────┘  │        │  ┌──────────────┐            │
│                    │        └─→│ Agent N Feed │            │
│  ┌──────────────┐  │           └──────────────┘            │
│  │ Social Graph │──┘                                        │
│  │ (Network)    │               Personalized                │
│  └──────────────┘               for each agent              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Requirements

### 1. **State Management**

The recommendation system must maintain:

```python
class RecommendationSystem:
    # Global state
    feed_pool: List[Feed]           # All feeds ever created
    agent_pool: Dict[str, Agent]    # All agents in the system
    social_graph: Dict[str, Set[str]]  # Who follows whom
    
    # Temporal state
    current_timestamp: datetime
    feed_history: Dict[str, List[Feed]]  # Per-agent view history
    
    # Behavioral state
    agent_actions: Dict[str, List[Action]]  # What agents have done
    engagement_signals: Dict[str, Dict]  # Likes, replies, etc.
```

### 2. **Core Operations**

#### 2.1 Content Ingestion
```python
def ingest_feed(self, feed: Feed) -> None:
    """
    Add new feed to the global pool
    
    When any agent creates content (post/reply/retweet/quote),
    it enters the recommendation system's feed pool.
    
    The system decides which other agents should see it.
    """
```

#### 2.2 Personalized Retrieval
```python
def fetch(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return personalized content for a specific agent
    
    This is THE core function - it implements the algorithm that
    determines what each agent sees.
    
    Args:
        agent_id: Which agent is requesting content
        context: Additional context (time, location, recent activity)
    
    Returns:
        {
            "feeds": List[Feed],     # Ranked/filtered posts to show
            "users": List[User],     # Suggested users to follow
            "trends": List[str],     # Trending topics/hashtags
            "metadata": Dict         # Algorithm explanation (optional)
        }
    
    Algorithm Considerations:
    - Recency: Show recent posts?
    - Relevance: Match agent's interests?
    - Social: Prioritize followed users?
    - Diversity: Show varied content?
    - Engagement: Show viral content?
    - Exploration: Introduce new topics?
    - Fairness: Give all creators visibility?
    """
```

#### 2.3 Feedback Loop
```python
def record_action(self, agent_id: str, action: str, target_id: str) -> None:
    """
    Record agent actions to improve recommendations
    
    When agents like, reply, follow, etc., the system learns:
    - What content engages which agents
    - How to better personalize future feeds
    - Which creators to amplify
    
    This creates the feedback loop that shapes network dynamics.
    """
```

### 3. **Algorithm Strategies**

The recommendation system can implement different strategies:

#### 3.1 **Chronological**
```python
def fetch_chronological(self, agent_id: str) -> List[Feed]:
    """Show most recent posts from followed users (Twitter 2010)"""
    following = self.social_graph[agent_id]
    feeds = [f for f in self.feed_pool if f.author_id in following]
    return sorted(feeds, key=lambda f: f.created_at, reverse=True)
```

#### 3.2 **Engagement-Based**
```python
def fetch_engagement(self, agent_id: str) -> List[Feed]:
    """Show posts with highest engagement (Facebook algorithm)"""
    feeds = self.feed_pool
    return sorted(feeds, key=lambda f: f.engagement_score, reverse=True)
```

#### 3.3 **Interest-Based**
```python
def fetch_interest(self, agent_id: str) -> List[Feed]:
    """Show posts matching agent's interests (content-based filtering)"""
    agent_interests = self.get_agent_interests(agent_id)
    feeds = [f for f in self.feed_pool if matches_interests(f, agent_interests)]
    return feeds
```

#### 3.4 **Collaborative Filtering**
```python
def fetch_collaborative(self, agent_id: str) -> List[Feed]:
    """Show posts liked by similar agents (Netflix-style)"""
    similar_agents = self.find_similar_agents(agent_id)
    feeds = self.get_feeds_liked_by(similar_agents)
    return feeds
```

#### 3.5 **Exploration vs Exploitation**
```python
def fetch_balanced(self, agent_id: str, explore_ratio: float = 0.2) -> List[Feed]:
    """
    Balance showing known-good content with new discoveries
    
    - 80% exploit: Content we know the agent likes
    - 20% explore: New content to broaden their feed
    """
```

---

## 4. **Interface Protocol**

```python
from typing import Protocol, Dict, List, Any
from pydantic import BaseModel

class Feed(BaseModel):
    """Feed/Post data structure"""
    id: str
    text: str
    author_id: str
    created_at: str
    feed_type: str  # post, reply, retweet, quote
    # ... other fields

class RecommendationSystem(Protocol):
    """
    Protocol that all recommendation systems must implement
    """
    
    def ingest_feed(self, feed: Feed) -> None:
        """Add new content to the system"""
        ...
    
    def fetch(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized content for an agent"""
        ...
    
    def record_action(
        self, 
        agent_id: str, 
        action: str, 
        target_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Record agent action for learning"""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics (for analysis)"""
        ...
```

---

## 5. **System Properties**

### 5.1 **Centralized vs Decentralized**
- **Centralized**: One system sees all agents and all feeds (like current social media)
- **Decentralized**: Each agent has local view (like real-world social networks)

### 5.2 **Real-time vs Batch**
- **Real-time**: Feeds update immediately as content is created
- **Batch**: Feeds computed periodically (e.g., every hour)

### 5.3 **Stateful vs Stateless**
- **Stateful**: Remembers history, learns over time
- **Stateless**: Pure function of current inputs

### 5.4 **Transparent vs Opaque**
- **Transparent**: Agents know why they see certain content
- **Opaque**: Black-box algorithm (like real platforms)

---

## 6. **Evaluation Metrics**

A good recommendation system should optimize for:

```python
class RecommendationMetrics:
    # Engagement metrics
    click_through_rate: float      # Do agents engage with shown content?
    time_spent: float               # How long do agents stay active?
    
    # Diversity metrics
    content_diversity: float        # Variety of content shown
    creator_diversity: float        # How many different creators shown?
    
    # Network metrics
    network_density: float          # Are agents connecting?
    information_flow: float         # Does content spread?
    
    # Fairness metrics
    creator_visibility: Dict[str, float]  # Do all creators get seen?
    filter_bubble_score: float      # Are agents stuck in echo chambers?
```

---

## 7. **Example Implementations**

### Simple Random
```python
class RandomRecommendation:
    def fetch(self, agent_id, context):
        return {
            "feeds": random.sample(self.feed_pool, k=10),
            "users": [],
            "trends": []
        }
```

### Social Graph Based
```python
class FollowingRecommendation:
    def fetch(self, agent_id, context):
        following = self.social_graph[agent_id]
        feeds = [f for f in self.feed_pool 
                if f.author_id in following]
        return {
            "feeds": feeds[-10:],  # Most recent 10
            "users": self._suggest_friends_of_friends(agent_id),
            "trends": self._compute_trends()
        }
```

### ML-Based
```python
class MLRecommendation:
    def __init__(self):
        self.model = train_recommender_model()
    
    def fetch(self, agent_id, context):
        agent_embedding = self.model.encode_agent(agent_id)
        feed_embeddings = self.model.encode_feeds(self.feed_pool)
        
        scores = cosine_similarity(agent_embedding, feed_embeddings)
        top_feeds = self.feed_pool[np.argsort(scores)[-10:]]
        
        return {
            "feeds": top_feeds,
            "users": self._collaborative_filter_users(agent_id),
            "trends": self._extract_trending_topics()
        }
```

---

## 8. **Integration with Agent**

The agent and recommendation system form a **closed loop**:

```python
# Simulation loop
rec_system = RecommendationSystem(feed_pool, agent_pool)

while simulation_running:
    for agent in agents:
        # 1. Platform shows content to agent
        content = rec_system.fetch(agent.agent_id, context={})
        
        # 2. Agent decides what to do
        decision = agent.decide_next_action(content)
        
        # 3. Agent takes action
        if decision == "post":
            new_feed = agent.create_post("Hello!")
            rec_system.ingest_feed(new_feed)  # New content enters system
        
        elif decision == "like":
            agent.like(content["feeds"][0]["id"])
            rec_system.record_action(
                agent.agent_id, 
                "like", 
                content["feeds"][0]["id"]
            )  # System learns from this
        
        # ... other actions
```

---

## 9. **Research Questions**

The recommendation system design affects:

1. **Information Diffusion**: How does content spread through the network?
2. **Echo Chambers**: Do agents get trapped in filter bubbles?
3. **Creator Incentives**: What content gets amplified?
4. **Network Topology**: How does the social graph evolve?
5. **Emergent Behavior**: What collective patterns emerge?

By swapping different recommendation systems, researchers can study how
platform algorithms shape social dynamics.

---

## Summary

The **Recommendation System** is:
- ✅ **The platform algorithm** that controls information flow
- ✅ **A mediator** between content supply (feeds) and content demand (agents)
- ✅ **Stateful** - maintains global feed pool, agent history, social graph
- ✅ **The key determinant** of network dynamics and agent behavior
- ✅ **Swappable** - different algorithms create different social platforms

**It's not just a function - it's the environment itself.**

