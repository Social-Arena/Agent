"""
Social Arena Agent - Minimal Implementation
A Twitter-like AI agent with 9 fundamental actions
"""

from typing import List, Optional, Dict, Any, Protocol
from pydantic import BaseModel, Field
from datetime import datetime
import sys
from pathlib import Path

# Add external/Feed to path
_external_feed_path = Path(__file__).parent / "external" / "Feed"
if str(_external_feed_path) not in sys.path:
    sys.path.insert(0, str(_external_feed_path))


class RecommendationSystem(Protocol):
    """
    Protocol for recommendation systems - the platform algorithm
    
    The recommendation system is the MEDIATOR between:
    - Supply: All feeds created by all agents (content pool)
    - Demand: What each individual agent sees (personalized)
    
    It's essentially Twitter's algorithm, Facebook's feed, TikTok's For You page.
    This is what makes each social platform unique and shapes all network dynamics.
    
    Key responsibilities:
    1. Maintain global feed pool (all content)
    2. Maintain agent pool (all users)
    3. Track social graph (who follows whom)
    4. Personalize content for each agent
    5. Learn from agent actions (feedback loop)
    
    See RECOMMENDATION_SYSTEM_SPEC.md for detailed requirements.
    """
    
    def ingest_feed(self, feed: Dict[str, Any]) -> None:
        """
        Add new content to the system
        
        When any agent creates content (post/reply/retweet/quote),
        it enters the recommendation system's feed pool.
        
        Args:
            feed: Feed data dict from agent.create_post(), etc.
        """
        ...
    
    def fetch(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get personalized content for a specific agent
        
        This is THE core function - implements the algorithm that
        determines what each agent sees. Different implementations
        create different social platforms.
        
        Args:
            agent_id: Which agent is requesting content
            context: Additional context (time, recent activity, etc.)
            
        Returns:
            Dict containing:
                - feeds: List[Dict] - Ranked/filtered posts to show this agent
                - users: List[Dict] - Suggested users to follow
                - trends: List[str] - Trending topics/hashtags
                - metadata: Dict - Algorithm explanation (optional)
        
        Algorithm strategies:
            - Chronological: Recent posts from followed users
            - Engagement: Viral/popular content
            - Interest-based: Match agent's preferences
            - Collaborative: Similar to what similar agents like
            - Exploration: Balance known content with discovery
        """
        ...
    
    def record_action(
        self, 
        agent_id: str, 
        action: str, 
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record agent action to improve recommendations
        
        When agents like, reply, follow, etc., the system learns:
        - What content engages which agents
        - How to better personalize future feeds
        - Which creators to amplify
        
        This creates the feedback loop that shapes network dynamics.
        
        Args:
            agent_id: Who performed the action
            action: What they did (like, reply, follow, etc.)
            target_id: What they did it to (feed_id or user_id)
            metadata: Additional data (dwell time, click depth, etc.)
        """
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics for analysis
        
        Returns:
            Dict with metrics like:
                - total_feeds: Number of posts in system
                - total_agents: Number of users
                - engagement_rate: Average engagement
                - content_diversity: Variety of content
                - network_density: How connected the graph is
        """
        ...


class Agent(BaseModel):
    """
    Base Agent class with 9 fundamental actions for Social Arena simulation
    
    Every agent (AI or human) has these same 9 actions:
    - Content Creation (4): create_post, reply, retweet, quote
    - Engagement (2): like, unlike
    - Social Graph (2): follow, unfollow
    - Decision Making (1): decide_next_action
    
    Note: Agents don't discover content themselves. Content is fed to them
    by a RecommendationSystem (like real social media algorithms).
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
    # 9 FUNDAMENTAL ACTIONS
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
    
    # -------------------- DECISION MAKING (1) --------------------
    
    def decide_next_action(self, recommended_content: Dict[str, Any]) -> str:
        """
        ACTION 9: Decide what to do next based on recommended content
        
        Args:
            recommended_content: Content from RecommendationSystem.fetch()
                - feeds: List of recommended tweets
                - users: List of recommended users
                - trends: List of trending topics
            
        Returns:
            Action name: "post", "reply", "retweet", "quote", "like", "follow", "idle"
            
        Note:
            Override this method to implement custom AI behavior.
            The agent receives content from external recommendation system,
            not by browsing/searching itself.
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
