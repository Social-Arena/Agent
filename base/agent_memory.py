"""
Agent Memory System - Stores and retrieves agent experience
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from Agent.utils.logging_utils import get_logger


@dataclass
class MemoryItem:
    """Individual memory item"""
    timestamp: datetime
    item_type: str  # "interaction", "content", "strategy", "observation"
    data: Dict[str, Any]
    importance: float = 0.5  # 0-1, higher = more important
    recall_count: int = 0


@dataclass
class Evidence:
    """Evidence for belief updates"""
    evidence_type: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime


@dataclass
class Interaction:
    """Interaction record"""
    interaction_id: str
    interaction_type: str
    timestamp: datetime
    participants: List[str]
    outcome: Dict[str, Any]
    context: Dict[str, Any]


@dataclass
class ContentMetrics:
    """Content performance metrics"""
    content_id: str
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    engagement_rate: float = 0.0
    virality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BeliefSystem:
    """Agent belief system"""
    
    def __init__(self):
        self.beliefs: Dict[str, float] = {}  # belief_name -> confidence (0-1)
        self.evidence_history: List[Evidence] = []
        self.logger = get_logger("BeliefSystem", component="agent_memory")
    
    def get_belief(self, belief_name: str) -> float:
        """Get belief confidence"""
        return self.beliefs.get(belief_name, 0.5)
    
    def update(self, evidence: Evidence) -> None:
        """Update beliefs based on new evidence"""
        self.evidence_history.append(evidence)
        
        # Simple Bayesian-like update
        belief_name = evidence.evidence_type
        current_belief = self.beliefs.get(belief_name, 0.5)
        
        # Update with evidence, weighted by confidence
        updated_belief = (current_belief + evidence.confidence * 0.3) / 1.3
        updated_belief = max(0.0, min(1.0, updated_belief))
        
        self.beliefs[belief_name] = updated_belief
        
        self.logger.debug(f"Belief updated", extra={
            "belief": belief_name,
            "old_value": current_belief,
            "new_value": updated_belief,
            "evidence_confidence": evidence.confidence
        })
    
    def get_all_beliefs(self) -> Dict[str, float]:
        """Get all beliefs"""
        return self.beliefs.copy()


class StrategyMemory:
    """Memory of strategies and their performance"""
    
    def __init__(self):
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.logger = get_logger("StrategyMemory", component="agent_memory")
    
    def record_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> None:
        """Record a strategy"""
        self.strategies[strategy_id] = {
            "data": strategy_data,
            "created_at": datetime.now(),
            "usage_count": 0
        }
    
    def record_performance(self, strategy_id: str, performance: float) -> None:
        """Record strategy performance"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]["usage_count"] += 1
        
        self.performance_history[strategy_id].append(performance)
        
        self.logger.debug(f"Strategy performance recorded", extra={
            "strategy_id": strategy_id,
            "performance": performance,
            "avg_performance": self.get_average_performance(strategy_id)
        })
    
    def get_average_performance(self, strategy_id: str) -> float:
        """Get average performance of strategy"""
        history = self.performance_history.get(strategy_id, [])
        return sum(history) / max(len(history), 1)
    
    def get_best_strategy(self) -> Optional[str]:
        """Get best performing strategy"""
        if not self.performance_history:
            return None
        
        best_strategy = max(
            self.performance_history.keys(),
            key=lambda k: self.get_average_performance(k)
        )
        return best_strategy


class FanSegmentMemory:
    """Memory of audience segments and preferences"""
    
    def __init__(self):
        self.segments: Dict[str, Dict[str, Any]] = {}
        self.preferences: Dict[str, List[str]] = defaultdict(list)
        self.engagement_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.logger = get_logger("FanSegmentMemory", component="agent_memory")
    
    def add_segment(self, segment_id: str, segment_data: Dict[str, Any]) -> None:
        """Add audience segment"""
        self.segments[segment_id] = segment_data
    
    def update_audience_preferences(self, audience_reaction: Dict[str, Any]) -> None:
        """Update audience preferences based on reactions"""
        # Extract preferences from reaction data
        for key, value in audience_reaction.items():
            if isinstance(value, (list, tuple)):
                self.preferences[key].extend(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.engagement_patterns[key][sub_key] = sub_value
    
    def get_primary_audience(self) -> Optional[Dict[str, Any]]:
        """Get primary audience segment"""
        if not self.segments:
            return None
        
        # Return first segment for now
        return list(self.segments.values())[0]
    
    def get_preferences_summary(self) -> Dict[str, List[str]]:
        """Get summary of audience preferences"""
        return dict(self.preferences)


class AgentMemory:
    """Agent memory system - stores and retrieves agent experience"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = get_logger(f"AgentMemory_{agent_id}", component="agent_memory")
        
        # Memory components
        self.fan_segment_memory = FanSegmentMemory()
        self.belief_system = BeliefSystem()
        self.strategy_memory = StrategyMemory()
        
        # Interaction history
        self.interaction_history: List[Interaction] = []
        
        # Content performance tracking
        self.content_performance: Dict[str, ContentMetrics] = {}
        
        # General memory storage
        self.memory_items: List[MemoryItem] = []
        
        # Learning stage transitions
        self.stage_transitions: List[Dict[str, Any]] = []
        
        self.logger.info(f"AgentMemory initialized for agent {agent_id}")
    
    def remember_interaction(self, interaction: Interaction) -> None:
        """Remember interaction experience"""
        self.interaction_history.append(interaction)
        
        # Create memory item
        memory_item = MemoryItem(
            timestamp=interaction.timestamp,
            item_type="interaction",
            data={
                "interaction_id": interaction.interaction_id,
                "type": interaction.interaction_type,
                "outcome": interaction.outcome
            },
            importance=self._calculate_importance(interaction.outcome)
        )
        self.memory_items.append(memory_item)
        
        self.logger.debug(f"Interaction remembered", extra={
            "agent_id": self.agent_id,
            "interaction_id": interaction.interaction_id,
            "interaction_type": interaction.interaction_type
        })
    
    def remember_content_performance(self, content_id: str, metrics: ContentMetrics) -> None:
        """Remember content performance"""
        self.content_performance[content_id] = metrics
        
        # Create memory item
        memory_item = MemoryItem(
            timestamp=metrics.timestamp,
            item_type="content",
            data={
                "content_id": content_id,
                "engagement_rate": metrics.engagement_rate,
                "virality_score": metrics.virality_score
            },
            importance=metrics.virality_score
        )
        self.memory_items.append(memory_item)
        
        self.logger.debug(f"Content performance remembered", extra={
            "agent_id": self.agent_id,
            "content_id": content_id,
            "engagement_rate": metrics.engagement_rate,
            "virality_score": metrics.virality_score
        })
    
    def recall_similar_situations(self, current_context: Dict[str, Any]) -> List[MemoryItem]:
        """
        Recall similar past situations
        
        Args:
            current_context: Current situation context
            
        Returns:
            List of similar memory items
        """
        similar_memories = []
        
        for memory in self.memory_items:
            # Simple similarity based on context keys
            similarity = self._calculate_similarity(memory.data, current_context)
            
            if similarity > 0.5:  # Threshold for similarity
                memory.recall_count += 1
                similar_memories.append(memory)
        
        # Sort by importance and recency
        similar_memories.sort(
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )
        
        self.logger.debug(f"Recalled similar situations", extra={
            "agent_id": self.agent_id,
            "context_keys": list(current_context.keys()),
            "num_similar": len(similar_memories)
        })
        
        return similar_memories[:10]  # Return top 10
    
    def update_beliefs(self, new_evidence: Evidence) -> None:
        """Update belief system with new evidence"""
        self.belief_system.update(new_evidence)
        
        self.logger.debug(f"Beliefs updated", extra={
            "agent_id": self.agent_id,
            "evidence_type": new_evidence.evidence_type,
            "confidence": new_evidence.confidence
        })
    
    def log_stage_transition(self, new_stage) -> None:
        """Log learning stage transition"""
        transition = {
            "timestamp": datetime.now(),
            "new_stage": new_stage.value if hasattr(new_stage, 'value') else str(new_stage),
            "memory_size": len(self.memory_items),
            "interaction_count": len(self.interaction_history)
        }
        self.stage_transitions.append(transition)
        
        self.logger.info(f"Stage transition logged", extra={
            "agent_id": self.agent_id,
            **transition
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        return {
            "total_memories": len(self.memory_items),
            "interactions": len(self.interaction_history),
            "content_tracked": len(self.content_performance),
            "beliefs": len(self.belief_system.beliefs),
            "strategies": len(self.strategy_memory.strategies),
            "audience_segments": len(self.fan_segment_memory.segments),
            "stage_transitions": len(self.stage_transitions)
        }
    
    def _calculate_importance(self, outcome: Dict[str, Any]) -> float:
        """Calculate importance of an outcome"""
        # Simple heuristic based on outcome metrics
        importance = 0.5
        
        if "success" in outcome:
            importance += 0.2 if outcome["success"] else -0.2
        
        if "reward" in outcome:
            importance += min(outcome["reward"] / 100, 0.3)
        
        return max(0.0, min(1.0, importance))
    
    def _calculate_similarity(self, memory_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate similarity between memory and current context"""
        if not memory_data or not context:
            return 0.0
        
        # Simple Jaccard similarity on keys
        memory_keys = set(memory_data.keys())
        context_keys = set(context.keys())
        
        intersection = len(memory_keys & context_keys)
        union = len(memory_keys | context_keys)
        
        return intersection / max(union, 1)
    
    def clear_old_memories(self, days: int = 30) -> int:
        """Clear memories older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        initial_count = len(self.memory_items)
        self.memory_items = [
            m for m in self.memory_items
            if m.timestamp.timestamp() > cutoff
        ]
        
        removed_count = initial_count - len(self.memory_items)
        
        self.logger.info(f"Cleared old memories", extra={
            "agent_id": self.agent_id,
            "removed_count": removed_count,
            "remaining_count": len(self.memory_items)
        })
        
        return removed_count

