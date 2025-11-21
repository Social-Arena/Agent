# Agent - Intelligent Agent System 🤖

**Self-Evolving Agents for Social Media Simulation**

The Agent module implements 100-1000 self-evolving intelligent agents with four specialized roles, supporting progressive learning from multi-armed bandits to reinforcement learning to evolutionary strategies.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Agent Roles](#agent-roles)
- [Learning System](#learning-system)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Agent Architecture](#agent-architecture)
- [Memory System](#memory-system)
- [Content Generation](#content-generation)
- [Social Intelligence](#social-intelligence)
- [Integration](#integration)
- [Development Guide](#development-guide)

---

## 🎯 Overview

The Agent system is the **intelligent behavioral unit** of Social-Arena. It implements autonomous agents capable of:

- **Multi-role Behavior**: 4 specialized agent types with distinct goals
- **Adaptive Learning**: Progressive learning from exploration to exploitation
- **Viral Optimization**: Strategy optimization based on propagation metrics
- **Social Intelligence**: Understanding and leveraging social network dynamics

### Six Core Capabilities

| Capability | Description |
|------------|-------------|
| **Multi-Role Behavior** | Creator, Audience, Brand, Moderator with distinct behaviors |
| **Multi-Stage Learning** | Cold Start → Bandit → RL → Evolution |
| **Viral Optimization** | Content generation + audience feedback → iterative improvement |
| **Social Intelligence** | Cross-agent interaction, collaboration, competition, information diffusion |
| **Memory System** | Beliefs, strategy memory, fan segments, content performance |
| **Evolvable Strategy Head** | Decision-making, timing selection, collaboration, sponsorship, style adaptation |

---

## 👥 Agent Roles

### Agent Hierarchy

```
                      ┌────────────────────────────┐
                      │         BaseAgent           │
                      │  (Abstract Behavior Interface)
                      └───────────┬────────────────┘
                                  │
 ┌─────────────────────┬─────────┴──────────┬──────────────────────┐
 │ CreatorAgent        │ AudienceAgent       │ BrandAgent            │ ModeratorAgent
 │ (Content Creator)   │ (User Simulation)   │ (Commercial Goals)    │ (Rule Enforcement)
 └─────────────────────┴─────────────────────┴──────────────────────┘
```

### 1. Creator Agent 👨‍💼

**Goal**: Maximize reach and engagement through content creation

**Capabilities**:
- Content creation and optimization
- Fan segment targeting
- Collaboration with other creators
- Brand partnership management
- Viral optimization strategies

**Behavior Pattern**:
```python
def creator_behavior():
    # Analyze trending topics
    trends = analyze_current_trends()
    
    # Generate optimized content
    content = generate_viral_content(trends, fan_segments)
    
    # Select optimal posting time
    timing = optimize_posting_time(audience_activity)
    
    # Publish and track performance
    publish(content, timing)
    track_engagement_metrics()
```

**Key Metrics**:
- Follower growth rate
- Content virality score
- Engagement rate
- Brand partnership ROI

---

### 2. Audience Agent 👥

**Goal**: Consume relevant content and provide realistic user feedback

**Capabilities**:
- Content consumption with time budgets
- Probabilistic engagement decisions
- Preference drift over time
- Social proof sensitivity
- Network influence

**Behavior Pattern**:
```python
def audience_behavior():
    # Browse recommended content
    recommended = get_recommendations()
    
    # Make viewing decisions based on preferences
    for content in recommended:
        if should_view(content, preferences, social_signals):
            engage_with_content(content)
            update_preferences(content)
    
    # Learn from social network
    observe_friends_behavior()
```

**Key Metrics**:
- Session duration
- Content diversity consumed
- Engagement quality
- Preference stability

---

### 3. Brand Agent 🏢

**Goal**: Achieve marketing objectives through strategic content placement

**Capabilities**:
- Creator partnership identification
- Budget allocation optimization
- Campaign performance tracking
- Target audience refinement
- ROI maximization

**Behavior Pattern**:
```python
def brand_behavior():
    # Identify partnership opportunities
    creators = find_relevant_creators(target_demographics)
    
    # Allocate marketing budget
    allocation = optimize_budget(creators, past_performance)
    
    # Launch sponsored content
    campaigns = launch_campaigns(allocation)
    
    # Track and optimize
    track_kpis(campaigns)
    adjust_strategy(performance_data)
```

**Key Metrics**:
- ROI (Return on Investment)
- CTR (Click-Through Rate)
- Conversion rate
- Brand sentiment

---

### 4. Moderator Agent 👮‍♂️

**Goal**: Maintain platform health and enforce community guidelines

**Capabilities**:
- Content safety monitoring
- Crisis detection and response
- Exploration parameter tuning
- Rule enforcement
- Fairness monitoring

**Behavior Pattern**:
```python
def moderator_behavior():
    # Scan for policy violations
    violations = scan_content(safety_rules)
    
    # Detect potential crises
    crises = detect_crises(platform_activity)
    
    # Take action
    if violations:
        moderate_content(violations)
    
    if crises:
        activate_crisis_response(crises)
    
    # Tune system parameters for platform health
    adjust_exploration_params(platform_metrics)
```

**Key Metrics**:
- Content safety score
- Crisis response time
- Platform health indicators
- Fairness metrics

---

## 🎓 Learning System

### Progressive Learning Strategy

```
🎯 Stage 1: Cold Start (Offline Imitation)
   │
   ├─ Learn from existing data
   ├─ Bootstrap initial strategy
   └─ Build basic behavior patterns
   │
   ▼
🎰 Stage 2: Multi-Armed Bandit (Fast A/B Testing)
   │
   ├─ Rapid strategy exploration
   ├─ ε-greedy action selection
   └─ Quick win identification
   │
   ▼
🧠 Stage 3: Reinforcement Learning (Long-term Optimization)
   │
   ├─ PPO/DDPG algorithms
   ├─ Reward maximization
   └─ Policy gradient optimization
   │
   ▼
🧬 Stage 4: Evolution Strategy (Population Optimization)
   │
   ├─ Strategy mutation
   ├─ Crossover operations
   └─ Fitness-based selection
```

### Learning Components

#### 1. Bandit Learner

```python
class BanditLearner:
    """Multi-armed bandit for fast A/B testing"""
    
    def select_action(self, available_actions: List[Action]) -> Action:
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(available_actions)  # Explore
        else:
            return self.select_best_action(available_actions)  # Exploit
    
    def update_reward(self, action: Action, reward: float) -> None:
        """Update action value estimates"""
```

#### 2. RL Trainer

```python
class RLTrainer:
    """Reinforcement learning trainer - PPO/DDPG"""
    
    async def train_step(self, batch: TrainingBatch) -> TrainingMetrics:
        """Execute one training step"""
        
    def get_action(self, state: State) -> Action:
        """Get action from current policy"""
        
    def update_policy(self, trajectories: List[Trajectory]) -> None:
        """Update policy from collected experience"""
```

#### 3. Evolution Strategy

```python
class EvolutionStrategy:
    """Evolution strategy - population-based optimization"""
    
    def evolve_population(self, fitness_scores: Dict[str, float]) -> None:
        """Evolve population through selection, crossover, mutation"""
        
    def _selection(self, fitness_scores: Dict[str, float]) -> List[Individual]:
        """Select individuals based on fitness"""
        
    def _crossover(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring through crossover"""
        
    def _mutation(self, individuals: List[Individual]) -> List[Individual]:
        """Apply mutation to individuals"""
```

---

## 📁 Directory Structure

```
agent/
│
├── base/                          # Base agent architecture
│   ├── __init__.py
│   ├── base_agent.py             # Abstract base class
│   ├── agent_memory.py           # Memory system
│   ├── strategy_head.py          # Decision engine
│   └── content_head.py           # Content generation
│
├── roles/                         # Specialized agent roles
│   ├── __init__.py
│   ├── creator_agent.py          # Creator agent implementation
│   ├── audience_agent.py         # Audience agent implementation
│   ├── brand_agent.py            # Brand agent implementation
│   └── moderator_agent.py        # Moderator agent implementation
│
├── learning/                      # Learning systems
│   ├── __init__.py
│   ├── bandit_learner.py         # Multi-armed bandit
│   ├── rl_trainer.py             # Reinforcement learning
│   └── evolution_strategy.py     # Evolution algorithms
│
├── components/                    # Agent components
│   ├── __init__.py
│   ├── norm_learner.py           # Platform norm learning
│   ├── virality_optimizer.py    # Viral content optimization
│   ├── fan_segment_memory.py    # Audience segmentation
│   ├── belief_system.py          # Belief management
│   └── strategy_memory.py        # Strategy storage
│
├── config/                        # Configuration
│   ├── __init__.py
│   └── agent_config.py           # Agent configuration
│
├── tests/                         # Test suite
│   ├── test_base_agent.py
│   ├── test_roles.py
│   ├── test_learning.py
│   └── test_memory.py
│
├── examples/                      # Usage examples
│   ├── creator_example.py
│   ├── agent_training_example.py
│   └── evolution_example.py
│
└── README.md                      # This file
```

---

## 🏗️ Core Components

### 1. Base Agent (`base/base_agent.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum

class AgentRole(Enum):
    CREATOR = "creator"
    AUDIENCE = "audience"
    BRAND = "brand"
    MODERATOR = "moderator"

class LearningStage(Enum):
    COLD_START = "cold_start"
    BANDIT = "bandit"
    REINFORCEMENT = "reinforcement"
    EVOLUTION = "evolution"

class BaseAgent(ABC):
    """Base agent abstract class - common interface for all agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, config: AgentConfig):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        
        # Core components
        self.memory = AgentMemory(agent_id)
        self.strategy_head = StrategyHead(role, config.strategy_config)
        self.content_head = ContentHead(role, config.content_config)
        self.norm_learner = NormLearner(config.norm_config)
        
        # Learning related
        self.learning_stage = LearningStage.COLD_START
        self.bandit_learner = BanditLearner()
        self.rl_trainer = RLTrainer()
        self.evolution_strategy = EvolutionStrategy()
        
        # Performance tracking
        self.performance_history = PerformanceHistory()
        self.current_strategy = None
    
    @abstractmethod
    async def act(self, environment_state: EnvironmentState) -> AgentAction:
        """Agent action - implemented by specific roles"""
        pass
    
    @abstractmethod
    async def update_from_feedback(self, feedback: ActionFeedback) -> None:
        """Learn from feedback - implemented by specific roles"""
        pass
    
    async def evolve_strategy(self) -> None:
        """Strategy evolution based on current learning stage"""
        if self.learning_stage == LearningStage.BANDIT:
            await self._bandit_update()
        elif self.learning_stage == LearningStage.REINFORCEMENT:
            await self._rl_update()
        elif self.learning_stage == LearningStage.EVOLUTION:
            await self._evolution_update()
    
    def transition_learning_stage(self, new_stage: LearningStage) -> None:
        """Transition to new learning stage"""
        self.learning_stage = new_stage
        self.memory.log_stage_transition(new_stage)
```

---

### 2. Agent Memory (`base/agent_memory.py`)

```python
class AgentMemory:
    """Agent memory system - stores and retrieves agent experience"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.fan_segment_memory = FanSegmentMemory()
        self.belief_system = BeliefSystem()
        self.strategy_memory = StrategyMemory()
        self.interaction_history = []
        self.content_performance = {}
    
    def remember_interaction(self, interaction: Interaction) -> None:
        """Remember interaction experience"""
        self.interaction_history.append(interaction)
        self._update_related_memories(interaction)
    
    def remember_content_performance(self, content_id: str, metrics: ContentMetrics) -> None:
        """Remember content performance"""
        self.content_performance[content_id] = metrics
    
    def recall_similar_situations(self, current_context: Context) -> List[MemoryItem]:
        """Recall similar past situations"""
        similar_memories = []
        for memory in self.interaction_history:
            if self._is_similar_context(memory.context, current_context):
                similar_memories.append(memory)
        return similar_memories
    
    def update_beliefs(self, new_evidence: Evidence) -> None:
        """Update belief system with new evidence"""
        self.belief_system.update(new_evidence)
```

---

### 3. Strategy Head (`base/strategy_head.py`)

```python
class StrategyHead:
    """Strategy head - decision engine"""
    
    def __init__(self, role: AgentRole, config: StrategyConfig):
        self.role = role
        self.config = config
        self.current_strategy = self._initialize_strategy()
    
    async def decide_action(self, context: ActionContext) -> StrategyDecision:
        """Decide on action considering environment, history, goals"""
        # Analyze context
        situation_assessment = self._assess_situation(context)
        
        # Consider available actions
        available_actions = self._get_available_actions(context)
        
        # Evaluate expected outcomes
        action_values = self._evaluate_actions(available_actions, situation_assessment)
        
        # Select best action
        selected_action = self._select_action(action_values)
        
        return StrategyDecision(action=selected_action, rationale=situation_assessment)
    
    async def decide_content_timing(self, content: Content) -> TimingDecision:
        """Decide optimal posting time"""
        # Analyze audience activity patterns
        audience_activity = self._analyze_audience_activity()
        
        # Consider competition
        competition_level = self._assess_competition_by_time()
        
        # Optimize timing
        optimal_time = self._optimize_timing(audience_activity, competition_level)
        
        return TimingDecision(scheduled_time=optimal_time)
    
    async def decide_collaboration(self, opportunity: CollaborationOpp) -> CollaborationDecision:
        """Decide on collaboration opportunity"""
        # Evaluate partner compatibility
        compatibility = self._evaluate_partner_compatibility(opportunity.partner)
        
        # Estimate mutual benefit
        benefit_estimate = self._estimate_collaboration_benefit(opportunity)
        
        # Make decision
        should_collaborate = compatibility > 0.7 and benefit_estimate > 0.5
        
        return CollaborationDecision(accept=should_collaborate, terms=opportunity.terms)
```

---

### 4. Content Head (`base/content_head.py`)

```python
class ContentHead:
    """Content generation head - LLM-based content creation"""
    
    def __init__(self, role: AgentRole, config: ContentConfig):
        self.role = role
        self.llm_model = self._load_llm_model(config.model_name)
        self.content_templates = self._load_templates(role)
        self.virality_optimizer = ViralityOptimizer()
    
    async def generate_content(self, context: ContentContext) -> GeneratedContent:
        """Generate content based on role, target audience, current trends"""
        # Prepare prompt
        prompt = self._create_generation_prompt(context)
        
        # Generate using LLM
        raw_content = await self.llm_model.generate(prompt)
        
        # Post-process
        processed_content = self._post_process_content(raw_content, context)
        
        # Extract entities
        entities = extract_entities(processed_content.text)
        
        return GeneratedContent(
            text=processed_content.text,
            entities=entities,
            metadata=processed_content.metadata
        )
    
    async def optimize_for_virality(self, content: Content, target_audience: Audience) -> Content:
        """Optimize content for viral propagation"""
        # Add emotional triggers
        content = self.virality_optimizer.enhance_emotional_appeal(content)
        
        # Align with trends
        content = self.virality_optimizer.align_with_trends(content, target_audience)
        
        # Add social proof elements
        content = self.virality_optimizer.add_social_proof(content)
        
        return content
    
    async def adapt_style(self, audience_feedback: AudienceFeedback) -> None:
        """Adapt content style based on audience feedback"""
        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback(audience_feedback)
        
        # Update style parameters
        self._update_style_parameters(feedback_analysis)
    
    def generate_hashtags(self, content: Content, trending_tags: List[str]) -> List[str]:
        """Generate relevant hashtags"""
        # Extract topics from content
        topics = self._extract_topics(content.text)
        
        # Match with trending tags
        relevant_tags = self._match_trending_tags(topics, trending_tags)
        
        # Generate custom tags
        custom_tags = self._generate_custom_tags(topics)
        
        return relevant_tags + custom_tags
```

---

## 🧠 Social Intelligence

### Cross-Agent Interaction

```python
class SocialIntelligence:
    """Social intelligence for cross-agent interaction"""
    
    async def analyze_network_position(self, agent_id: str) -> NetworkPosition:
        """Analyze agent's position in social network"""
        connections = self._get_agent_connections(agent_id)
        influence_score = self._calculate_influence_score(agent_id, connections)
        centrality = self._calculate_network_centrality(agent_id)
        
        return NetworkPosition(
            connections=connections,
            influence_score=influence_score,
            centrality=centrality
        )
    
    async def identify_collaboration_opportunities(self, agent_id: str) -> List[CollaborationOpp]:
        """Identify potential collaboration partners"""
        network_position = await self.analyze_network_position(agent_id)
        compatible_agents = self._find_compatible_agents(agent_id, network_position)
        
        opportunities = []
        for partner in compatible_agents:
            benefit = self._estimate_collaboration_benefit(agent_id, partner)
            if benefit > self.collaboration_threshold:
                opportunities.append(CollaborationOpp(partner=partner, benefit=benefit))
        
        return opportunities
    
    async def observe_and_learn(self, agent_id: str) -> LearningSignals:
        """Learn from observing other agents"""
        # Observe successful agents
        successful_agents = self._identify_successful_agents()
        
        # Analyze their strategies
        strategy_patterns = self._analyze_agent_strategies(successful_agents)
        
        # Extract learning signals
        signals = self._extract_learning_signals(strategy_patterns)
        
        return signals
```

---

## 🔌 Integration

### With Arena System

```python
async def register_with_arena(self, arena_manager: ArenaManager) -> None:
    """Register agent with Arena"""
    await arena_manager.register_agent(self)
    self.arena_manager = arena_manager

async def receive_environment_update(self, environment_state: EnvironmentState) -> None:
    """Receive environment updates from Arena"""
    self.last_environment_state = environment_state
    await self._adapt_to_environment(environment_state)
```

### With Feed System

```python
async def publish_content_to_feed(self, content: Content) -> str:
    """Publish content to Feed system"""
    feed = self._convert_to_feed_format(content)
    feed_id = await self.feed_manager.save_feed(feed)
    self.memory.remember_content_performance(feed_id, ContentMetrics())
    return feed_id

async def consume_feed_content(self, feed: Feed) -> EngagementAction:
    """Consume content from Feed system"""
    engagement = self._decide_engagement(feed)
    await self._execute_engagement(feed, engagement)
    return engagement
```

### With Recommendation System

```python
async def receive_recommendations(self, recommendations: List[Feed]) -> None:
    """Receive personalized recommendations"""
    self.current_recommendations = recommendations
    await self._process_recommendations(recommendations)

async def provide_interaction_feedback(self, interaction: Interaction) -> None:
    """Provide interaction feedback to recommendation system"""
    await self.recommendation_system.update_from_interaction(interaction)
```

---

## 🛠️ Development Guide

### Quick Start

```python
from agent import CreatorAgent, AgentConfig

# Create a creator agent
config = AgentConfig(
    agent_id="creator_001",
    role="creator",
    niche_specialty="technology",
    learning_stage="bandit",
    content_generation_model="gpt-4"
)

agent = CreatorAgent("creator_001", config)

# Initialize agent
await agent.initialize()

# Agent acts in environment
environment_state = arena.get_environment_state()
action = await agent.act(environment_state)

# Receive feedback and learn
feedback = arena.get_action_feedback(action)
await agent.update_from_feedback(feedback)
```

### Development Priority

#### Phase 1: Base Infrastructure ✅
- [x] Trace logging system
- [ ] BaseAgent implementation
- [ ] AgentMemory system
- [ ] Strategy and Content heads

#### Phase 2: Role Implementation 🚧
- [ ] CreatorAgent
- [ ] AudienceAgent  
- [ ] BrandAgent
- [ ] ModeratorAgent

#### Phase 3: Learning Systems
- [ ] Bandit learner
- [ ] RL trainer (PPO/DDPG)
- [ ] Evolution strategy
- [ ] Learning stage transitions

#### Phase 4: Social Intelligence
- [ ] Network analysis
- [ ] Collaboration mechanisms
- [ ] Social learning
- [ ] Influence tracking

---

## 📝 Trace Logging

**CRITICAL**: Use file-based trace logging. **NO console logs**.

```python
from agent.utils.logging_utils import get_logger

logger = get_logger(__name__, component="creator_agent")

# Log agent action
logger.info("Content created", extra={
    "agent_id": self.agent_id,
    "content_id": content.id,
    "virality_score": content.virality_score
})

# Log learning update
logger.debug("Strategy updated", extra={
    "agent_id": self.agent_id,
    "learning_stage": self.learning_stage,
    "reward": reward
})
```

---

## 📚 API Reference

### BaseAgent API

```python
# Core methods
async def act(environment_state: EnvironmentState) -> AgentAction
async def update_from_feedback(feedback: ActionFeedback) -> None
async def evolve_strategy() -> None

# Learning control
def transition_learning_stage(new_stage: LearningStage) -> None
async def get_current_strategy() -> Strategy

# State management
def get_current_state() -> AgentState
def get_performance_metrics() -> PerformanceMetrics
```

---

## 🤝 Contributing

When contributing to Agent module:

1. **Follow Role Patterns** - Each role has distinct behavior
2. **Use Trace Logging** - Never use console logs
3. **Test Learning** - Verify convergence in learning algorithms
4. **Document Strategies** - Explain decision-making logic
5. **Performance Aware** - Profile agent computational costs

---

## 📖 Related Documentation

- [Arena Module](../Arena/README.md)
- [Feed Module](../Feed/README.md)
- [Recommendation Module](../Recommendation/README.md)
- [Project Overview](../README.md)

---

**Agent - The Intelligence of Social-Arena** 🤖🧠

*Self-evolving agents learning in competitive social media environments*
