"""
AI Agent - Autonomous agent with decision-making capabilities
"""

from typing import Dict, Any, Optional
from pydantic import Field

from Agent.base.actor import Actor, ActorType


class AIAgent(Actor):
    """
    AI-driven autonomous agent that can make decisions and act independently
    
    This is the base class for all AI agents in Social Arena.
    Extend this class to create agents with specific personalities and behaviors.
    """
    
    actor_type: str = Field(default=ActorType.AI_AGENT)
    
    # AI-specific configuration
    autonomous: bool = Field(default=True, description="Whether agent acts autonomously")
    action_frequency: float = Field(default=1.0, description="Actions per time unit (0.0 to 1.0)")
    
    # Decision-making parameters
    engagement_threshold: float = Field(default=0.3, description="Minimum score to engage")
    posting_threshold: float = Field(default=0.5, description="Minimum score to post")
    
    # ================================================================
    # AI-SPECIFIC METHODS (Override these for custom behavior)
    # ================================================================
    
    def decide_next_action(self, context: Dict[str, Any]) -> str:
        """
        Decide what action to take next
        
        Override this method to implement custom decision logic.
        
        Args:
            context: Current context with:
                - tweets: List of available tweets
                - trending_topics: List of trending topics
                - time_of_day: Current time
                - agent_state: Agent's current state
                
        Returns:
            Action name: "post", "engage", "network", "browse", "idle"
        """
        # Simple decision logic - can be overridden
        tweets = context.get("tweets", [])
        
        if not tweets:
            return "browse"
        
        # Check if we should engage with tweets
        if len(tweets) > 0:
            return "engage"
        
        # Default to idle
        return "idle"
    
    def generate_content(self, topic: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate content for a post
        
        Override this method to implement custom content generation (e.g., LLM-based).
        
        Args:
            topic: Topic to post about
            context: Optional context for content generation
            
        Returns:
            Generated text content
        """
        # Simple template-based generation - should be overridden
        return f"Posting about {topic}! #SocialArena"
    
    def evaluate_tweet(self, tweet: Dict[str, Any]) -> float:
        """
        Evaluate relevance/interest in a tweet
        
        Override this method to implement custom evaluation logic.
        
        Args:
            tweet: Feed data dict to evaluate
            
        Returns:
            Score from 0.0 to 1.0 (higher = more relevant)
        """
        # Simple evaluation - should be overridden
        text = tweet.get("text", "")
        
        # Basic scoring based on text length
        if len(text) > 100:
            return 0.7
        elif len(text) > 50:
            return 0.5
        else:
            return 0.3
    
    def select_engagement_type(self, tweet: Dict[str, Any], relevance_score: float) -> str:
        """
        Decide how to engage with a tweet
        
        Args:
            tweet: Feed data dict
            relevance_score: Relevance score from evaluate_tweet()
            
        Returns:
            Engagement type: "like", "reply", "retweet", "quote", "ignore"
        """
        if relevance_score > 0.8:
            return "reply"
        elif relevance_score > 0.6:
            return "quote"
        elif relevance_score > 0.4:
            return "retweet"
        elif relevance_score > 0.2:
            return "like"
        else:
            return "ignore"
    
    # ================================================================
    # AUTONOMOUS EXECUTION LOOP
    # ================================================================
    
    async def act(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute one action based on current context
        
        This is the main method called by the simulation loop.
        
        Args:
            context: Current simulation context
            
        Returns:
            Action result dict or None
        """
        # Decide what to do
        action = self.decide_next_action(context)
        
        self.logger.log_decision(
            decision_type="action_selection",
            decision_data={"selected_action": action},
            reasoning=f"Based on context with {len(context.get('tweets', []))} tweets"
        )
        
        # Execute action
        if action == "post":
            return await self._action_post(context)
        
        elif action == "engage":
            return await self._action_engage(context)
        
        elif action == "network":
            return await self._action_network(context)
        
        elif action == "browse":
            return await self._action_browse(context)
        
        else:  # idle
            return None
    
    # ================================================================
    # INTERNAL ACTION EXECUTORS
    # ================================================================
    
    async def _action_post(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute posting action"""
        # Generate content
        topic = context.get("trending_topics", ["general"])[0]
        text = self.generate_content(topic, context)
        
        # Create post
        feed_data = self.create_post(text, context)
        
        return {"action": "post", "result": feed_data}
    
    async def _action_engage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute engagement action"""
        tweets = context.get("tweets", [])
        
        if not tweets:
            return {"action": "engage", "result": None}
        
        # Evaluate first tweet
        tweet = tweets[0]
        score = self.evaluate_tweet(tweet)
        
        # Decide engagement type
        engagement_type = self.select_engagement_type(tweet, score)
        
        result = None
        
        if engagement_type == "like":
            result = self.like(tweet["id"])
        
        elif engagement_type == "reply":
            reply_text = self.generate_content(f"reply to: {tweet['text'][:50]}", context)
            result = self.reply(tweet["id"], reply_text, tweet["author_id"])
        
        elif engagement_type == "retweet":
            result = self.retweet(tweet["id"])
        
        elif engagement_type == "quote":
            quote_text = self.generate_content(f"quote: {tweet['text'][:50]}", context)
            result = self.quote(tweet["id"], quote_text)
        
        return {"action": "engage", "engagement_type": engagement_type, "result": result}
    
    async def _action_network(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute network building action"""
        # Find users to follow (simplified)
        potential_users = context.get("potential_follows", [])
        
        if potential_users:
            user_id = potential_users[0]
            result = self.follow(user_id)
            return {"action": "network", "result": result}
        
        return {"action": "network", "result": None}
    
    async def _action_browse(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute browsing action"""
        tweets = self.browse_feed("home", limit=50)
        
        return {"action": "browse", "result": {"tweets_count": len(tweets)}}

