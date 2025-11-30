"""
Base Actor class - shared interface for AI Agents and Human Users
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from Agent.utils import AgentLogger, trace_agent_action


class ActorType:
    """Actor type constants"""
    AI_AGENT = "ai_agent"
    HUMAN_USER = "human_user"


class Actor(BaseModel):
    """
    Base class for any entity that can act in Social Arena
    Provides 12 fundamental actions for both AI agents and human users
    """
    
    actor_id: str
    actor_type: str = Field(default=ActorType.AI_AGENT)
    username: str
    display_name: str
    bio: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    # Social graph state
    following: List[str] = Field(default_factory=list)
    followers: List[str] = Field(default_factory=list)
    
    # Engagement state
    liked_tweets: List[str] = Field(default_factory=list)
    bookmarked_tweets: List[str] = Field(default_factory=list)
    
    # Logging
    logger: Optional[AgentLogger] = Field(default=None, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = AgentLogger(
                agent_id=self.actor_id,
                agent_role=self.actor_type
            )
    
    # ================================================================
    # FUNDAMENTAL ACTIONS - 12 Core Actions for Social Arena
    # ================================================================
    
    # -------------------- CONTENT CREATION (4) --------------------
    
    @trace_agent_action("create_post")
    def create_post(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create original post (tweet)
        
        Args:
            text: Tweet text content
            context: Optional context for logging
            
        Returns:
            Feed data dict ready for Feed.Feed(**data)
        """
        from external.Feed import generate_feed_id, extract_entities, FeedType
        
        feed_data = {
            "id": generate_feed_id(),
            "text": text,
            "author_id": self.actor_id,
            "feed_type": FeedType.POST,
        }
        
        entities = extract_entities(text)
        if entities:
            feed_data["entities"] = entities
        
        self.logger.log_content_generation(
            content_type="post",
            content_id=feed_data["id"],
            metadata={"text_length": len(text), "context": context or {}}
        )
        
        return feed_data
    
    @trace_agent_action("reply")
    def reply(self, tweet_id: str, text: str, author_id: str) -> Dict[str, Any]:
        """
        Reply to existing tweet
        
        Args:
            tweet_id: ID of tweet being replied to
            text: Reply text content
            author_id: Original tweet author ID
            
        Returns:
            Feed data dict
        """
        from external.Feed import generate_feed_id, extract_entities, FeedType, ReferencedFeed, ReferencedFeedType
        
        feed_data = {
            "id": generate_feed_id(),
            "text": text,
            "author_id": self.actor_id,
            "feed_type": FeedType.REPLY,
            "in_reply_to_user_id": author_id,
            "referenced_feeds": [
                {"type": ReferencedFeedType.REPLIED_TO, "id": tweet_id}
            ]
        }
        
        entities = extract_entities(text)
        if entities:
            feed_data["entities"] = entities
        
        self.logger.log_interaction(
            interaction_type="reply",
            target_id=tweet_id,
            interaction_data={"text": text, "author_id": author_id}
        )
        
        return feed_data
    
    @trace_agent_action("retweet")
    def retweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Retweet (simple share without commentary)
        
        Args:
            tweet_id: ID of tweet being retweeted
            
        Returns:
            Feed data dict
        """
        from external.Feed import generate_feed_id, FeedType, ReferencedFeed, ReferencedFeedType
        
        feed_data = {
            "id": generate_feed_id(),
            "text": "",
            "author_id": self.actor_id,
            "feed_type": FeedType.RETWEET,
            "referenced_feeds": [
                {"type": ReferencedFeedType.RETWEETED, "id": tweet_id}
            ]
        }
        
        self.logger.log_interaction(
            interaction_type="retweet",
            target_id=tweet_id,
            interaction_data={"action": "retweet"}
        )
        
        return feed_data
    
    @trace_agent_action("quote")
    def quote(self, tweet_id: str, text: str) -> Dict[str, Any]:
        """
        Quote tweet with commentary
        
        Args:
            tweet_id: ID of tweet being quoted
            text: Commentary text
            
        Returns:
            Feed data dict
        """
        from external.Feed import generate_feed_id, extract_entities, FeedType, ReferencedFeed, ReferencedFeedType
        
        feed_data = {
            "id": generate_feed_id(),
            "text": text,
            "author_id": self.actor_id,
            "feed_type": FeedType.QUOTE,
            "referenced_feeds": [
                {"type": ReferencedFeedType.QUOTED, "id": tweet_id}
            ]
        }
        
        entities = extract_entities(text)
        if entities:
            feed_data["entities"] = entities
        
        self.logger.log_interaction(
            interaction_type="quote",
            target_id=tweet_id,
            interaction_data={"text": text}
        )
        
        return feed_data
    
    # -------------------- ENGAGEMENT (2) --------------------
    
    @trace_agent_action("like")
    def like(self, tweet_id: str) -> bool:
        """
        Like a tweet
        
        Args:
            tweet_id: ID of tweet to like
            
        Returns:
            True if successful
        """
        if tweet_id not in self.liked_tweets:
            self.liked_tweets.append(tweet_id)
            
            self.logger.log_interaction(
                interaction_type="like",
                target_id=tweet_id,
                interaction_data={"action": "like"}
            )
            return True
        return False
    
    @trace_agent_action("unlike")
    def unlike(self, tweet_id: str) -> bool:
        """
        Remove like from tweet
        
        Args:
            tweet_id: ID of tweet to unlike
            
        Returns:
            True if successful
        """
        if tweet_id in self.liked_tweets:
            self.liked_tweets.remove(tweet_id)
            
            self.logger.log_interaction(
                interaction_type="unlike",
                target_id=tweet_id,
                interaction_data={"action": "unlike"}
            )
            return True
        return False
    
    # -------------------- SOCIAL GRAPH (2) --------------------
    
    @trace_agent_action("follow")
    def follow(self, user_id: str) -> bool:
        """
        Follow a user
        
        Args:
            user_id: ID of user to follow
            
        Returns:
            True if successful
        """
        if user_id not in self.following and user_id != self.actor_id:
            self.following.append(user_id)
            
            self.logger.log_interaction(
                interaction_type="follow",
                target_id=user_id,
                interaction_data={"action": "follow"}
            )
            return True
        return False
    
    @trace_agent_action("unfollow")
    def unfollow(self, user_id: str) -> bool:
        """
        Unfollow a user
        
        Args:
            user_id: ID of user to unfollow
            
        Returns:
            True if successful
        """
        if user_id in self.following:
            self.following.remove(user_id)
            
            self.logger.log_interaction(
                interaction_type="unfollow",
                target_id=user_id,
                interaction_data={"action": "unfollow"}
            )
            return True
        return False
    
    # -------------------- DISCOVERY (3) --------------------
    
    def browse_feed(self, feed_type: str = "home", limit: int = 50) -> List[Dict[str, Any]]:
        """
        Browse tweets from feed
        
        Args:
            feed_type: Type of feed ("home", "trending", "following")
            limit: Maximum number of tweets to return
            
        Returns:
            List of Feed data dicts
            
        Note:
            This method returns empty list - should be overridden by Arena environment
        """
        self.logger.debug(
            f"Browsing {feed_type} feed",
            extra={"feed_type": feed_type, "limit": limit}
        )
        return []
    
    def read_tweet(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Read single tweet by ID
        
        Args:
            tweet_id: ID of tweet to read
            
        Returns:
            Feed data dict or None if not found
            
        Note:
            This method returns None - should be overridden by Arena environment
        """
        self.logger.debug(
            f"Reading tweet {tweet_id}",
            extra={"tweet_id": tweet_id}
        )
        return None
    
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search tweets by query
        
        Args:
            query: Search query (hashtag, keyword, @mention)
            limit: Maximum number of results
            
        Returns:
            List of Feed data dicts
            
        Note:
            This method returns empty list - should be overridden by Arena environment
        """
        self.logger.debug(
            f"Searching: {query}",
            extra={"query": query, "limit": limit}
        )
        return []
    
    # -------------------- DECISION MAKING (1) --------------------
    
    def decide_next_action(self, context: Dict[str, Any]) -> str:
        """
        Decide what action to take next
        
        Args:
            context: Current context (available tweets, trends, etc.)
            
        Returns:
            Action name: "post", "engage", "network", "browse", "idle"
            
        Note:
            Base implementation returns "idle" - should be overridden by AI agents
        """
        self.logger.log_decision(
            decision_type="next_action",
            decision_data={"action": "idle"},
            reasoning="Base Actor class - no autonomous behavior"
        )
        return "idle"
    
    # -------------------- HELPER METHODS --------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics"""
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "following_count": len(self.following),
            "followers_count": len(self.followers),
            "liked_tweets_count": len(self.liked_tweets),
            "bookmarked_tweets_count": len(self.bookmarked_tweets)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump(exclude={"logger"})

