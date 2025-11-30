"""
Social Arena Agent - Minimal Implementation
A Twitter-like AI agent with 12 fundamental actions
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import sys
from pathlib import Path

# Add external/Feed to path
_external_feed_path = Path(__file__).parent / "external" / "Feed"
if str(_external_feed_path) not in sys.path:
    sys.path.insert(0, str(_external_feed_path))


class Agent(BaseModel):
    """
    Base Agent class with 12 fundamental actions for Social Arena simulation
    
    Every agent (AI or human) has these same 12 actions:
    - Content Creation (4): create_post, reply, retweet, quote
    - Engagement (2): like, unlike
    - Social Graph (2): follow, unfollow
    - Discovery (3): browse_feed, read_tweet, search
    - Decision Making (1): decide_next_action
    """
    
    # Identity
    agent_id: str
    username: str
    display_name: str = ""
    bio: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    # State
    following: List[str] = Field(default_factory=list)
    followers: List[str] = Field(default_factory=list)
    liked_tweets: List[str] = Field(default_factory=list)
    
    # ================================================================
    # 12 FUNDAMENTAL ACTIONS
    # ================================================================
    
    # -------------------- CONTENT CREATION (4) --------------------
    
    def create_post(self, text: str) -> Dict[str, Any]:
        """
        ACTION 1: Create original post
        
        Args:
            text: Tweet text content
            
        Returns:
            Feed data dict
        """
        import feed
        
        return {
            "id": feed.generate_feed_id(),
            "text": text,
            "author_id": self.agent_id,
            "feed_type": feed.FeedType.POST,
            "entities": feed.extract_entities(text)
        }
    
    def reply(self, tweet_id: str, text: str, author_id: str) -> Dict[str, Any]:
        """
        ACTION 2: Reply to tweet
        
        Args:
            tweet_id: ID of tweet being replied to
            text: Reply text
            author_id: Original tweet author
            
        Returns:
            Feed data dict
        """
        import feed
        
        return {
            "id": feed.generate_feed_id(),
            "text": text,
            "author_id": self.agent_id,
            "feed_type": feed.FeedType.REPLY,
            "in_reply_to_user_id": author_id,
            "referenced_feeds": [{"type": feed.ReferencedFeedType.REPLIED_TO, "id": tweet_id}],
            "entities": feed.extract_entities(text)
        }
    
    def retweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        ACTION 3: Retweet (share without commentary)
        
        Args:
            tweet_id: ID of tweet to retweet
            
        Returns:
            Feed data dict
        """
        import feed
        
        return {
            "id": feed.generate_feed_id(),
            "text": "",
            "author_id": self.agent_id,
            "feed_type": feed.FeedType.RETWEET,
            "referenced_feeds": [{"type": feed.ReferencedFeedType.RETWEETED, "id": tweet_id}]
        }
    
    def quote(self, tweet_id: str, text: str) -> Dict[str, Any]:
        """
        ACTION 4: Quote tweet with commentary
        
        Args:
            tweet_id: ID of tweet to quote
            text: Commentary text
            
        Returns:
            Feed data dict
        """
        import feed
        
        return {
            "id": feed.generate_feed_id(),
            "text": text,
            "author_id": self.agent_id,
            "feed_type": feed.FeedType.QUOTE,
            "referenced_feeds": [{"type": feed.ReferencedFeedType.QUOTED, "id": tweet_id}],
            "entities": feed.extract_entities(text)
        }
    
    # -------------------- ENGAGEMENT (2) --------------------
    
    def like(self, tweet_id: str) -> bool:
        """
        ACTION 5: Like a tweet
        
        Args:
            tweet_id: ID of tweet to like
            
        Returns:
            True if successful
        """
        if tweet_id not in self.liked_tweets:
            self.liked_tweets.append(tweet_id)
            return True
        return False
    
    def unlike(self, tweet_id: str) -> bool:
        """
        ACTION 6: Remove like from tweet
        
        Args:
            tweet_id: ID of tweet to unlike
            
        Returns:
            True if successful
        """
        if tweet_id in self.liked_tweets:
            self.liked_tweets.remove(tweet_id)
            return True
        return False
    
    # -------------------- SOCIAL GRAPH (2) --------------------
    
    def follow(self, user_id: str) -> bool:
        """
        ACTION 7: Follow a user
        
        Args:
            user_id: ID of user to follow
            
        Returns:
            True if successful
        """
        if user_id not in self.following and user_id != self.agent_id:
            self.following.append(user_id)
            return True
        return False
    
    def unfollow(self, user_id: str) -> bool:
        """
        ACTION 8: Unfollow a user
        
        Args:
            user_id: ID of user to unfollow
            
        Returns:
            True if successful
        """
        if user_id in self.following:
            self.following.remove(user_id)
            return True
        return False
    
    # -------------------- DISCOVERY (3) --------------------
    
    def browse_feed(self, feed_type: str = "home", limit: int = 50) -> List[Dict[str, Any]]:
        """
        ACTION 9: Browse tweets from feed
        
        Args:
            feed_type: "home", "trending", "following"
            limit: Max tweets to return
            
        Returns:
            List of Feed data dicts (override in environment)
        """
        return []
    
    def read_tweet(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        ACTION 10: Read single tweet
        
        Args:
            tweet_id: ID of tweet to read
            
        Returns:
            Feed data dict or None (override in environment)
        """
        return None
    
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        ACTION 11: Search tweets
        
        Args:
            query: Search query (hashtag, keyword, @mention)
            limit: Max results
            
        Returns:
            List of Feed data dicts (override in environment)
        """
        return []
    
    # -------------------- DECISION MAKING (1) --------------------
    
    def decide_next_action(self, context: Dict[str, Any]) -> str:
        """
        ACTION 12: Decide what to do next
        
        Args:
            context: Current context (tweets, trends, etc.)
            
        Returns:
            Action name: "post", "engage", "network", "browse", "idle"
            
        Note:
            Override this method to implement custom AI behavior
        """
        return "idle"
    
    # -------------------- HELPER METHODS --------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_id": self.agent_id,
            "username": self.username,
            "following_count": len(self.following),
            "followers_count": len(self.followers),
            "liked_tweets_count": len(self.liked_tweets)
        }
