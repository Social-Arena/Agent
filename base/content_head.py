"""
Content Head - LLM-based content generation for agents
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from Agent.utils.logging_utils import get_logger


class ContentStyle(Enum):
    """Content style types"""
    INFORMATIVE = "informative"
    ENTERTAINING = "entertaining"
    PROMOTIONAL = "promotional"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"


@dataclass
class ContentContext:
    """Context for content generation"""
    target_audience: Optional[Dict[str, Any]]
    trending_topics: List[str]
    personal_brand: str
    previous_performance: Dict[str, Any]
    content_style: ContentStyle = ContentStyle.CONVERSATIONAL
    max_length: int = 280


@dataclass
class GeneratedContent:
    """Generated content output"""
    text: str
    entities: Optional[Dict[str, List[Any]]]
    metadata: Dict[str, Any]
    confidence: float
    generation_timestamp: datetime


@dataclass
class AudienceFeedback:
    """Audience feedback for content adaptation"""
    content_id: str
    likes: int
    shares: int
    comments: int
    sentiment: float  # -1 to 1
    engagement_rate: float
    feedback_text: Optional[List[str]] = None


@dataclass
class ContentConfig:
    """Configuration for content generation"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 280
    enable_hashtags: bool = True
    enable_mentions: bool = True
    style: ContentStyle = ContentStyle.CONVERSATIONAL


class ViralityOptimizer:
    """Optimize content for viral propagation"""
    
    def __init__(self):
        self.logger = get_logger("ViralityOptimizer", component="agent_content")
        self.emotional_triggers = [
            "amazing", "incredible", "shocking", "must-see",
            "breaking", "exclusive", "urgent", "important"
        ]
    
    def enhance_emotional_appeal(self, content: str) -> str:
        """Add emotional triggers to content"""
        # Simple enhancement: add emotion words if missing
        has_emotion = any(trigger in content.lower() for trigger in self.emotional_triggers)
        
        if not has_emotion and len(content) < 250:
            # Add emotional element
            content = f"🔥 {content}"
        
        return content
    
    def align_with_trends(self, content: str, trends: List[str]) -> str:
        """Align content with current trends"""
        # Check if content mentions trends
        mentioned_trends = [t for t in trends if t.lower() in content.lower()]
        
        if not mentioned_trends and trends:
            # Add hashtag for top trend
            if not content.endswith("..."):
                content = f"{content} #{trends[0]}"
        
        return content
    
    def add_social_proof(self, content: str) -> str:
        """Add social proof elements"""
        # Add engagement prompts
        prompts = [
            "What do you think?",
            "Share your thoughts!",
            "Tag someone who needs to see this!",
            "Agree?"
        ]
        
        if "?" not in content and len(content) < 250:
            import random
            content = f"{content} {random.choice(prompts)}"
        
        return content
    
    def optimize(self, content: str, trends: List[str]) -> str:
        """Full optimization pipeline"""
        content = self.enhance_emotional_appeal(content)
        content = self.align_with_trends(content, trends)
        content = self.add_social_proof(content)
        return content


class ContentHead:
    """Content generation head - LLM-based content creation"""
    
    def __init__(self, role, config: ContentConfig):
        self.role = role
        self.config = config
        self.content_templates = self._load_templates(role)
        self.virality_optimizer = ViralityOptimizer()
        self.style_parameters = {}
        
        self.logger = get_logger(f"ContentHead_{role}", component="agent_content")
        
        self.logger.info(f"ContentHead initialized", extra={
            "role": str(role),
            "model": config.model_name,
            "style": config.style.value
        })
    
    async def generate_content(self, context: ContentContext) -> GeneratedContent:
        """
        Generate content based on role, target audience, current trends
        
        Args:
            context: Content generation context
            
        Returns:
            GeneratedContent: Generated content with metadata
        """
        # Prepare generation prompt
        prompt = self._create_generation_prompt(context)
        
        # Generate using LLM (mock for now)
        raw_content = await self._generate_with_llm(prompt, context)
        
        # Post-process
        processed_content = self._post_process_content(raw_content, context)
        
        # Extract entities
        entities = self._extract_entities(processed_content)
        
        generated = GeneratedContent(
            text=processed_content,
            entities=entities,
            metadata={
                "style": context.content_style.value,
                "trending_topics": context.trending_topics,
                "prompt_length": len(prompt)
            },
            confidence=0.8,  # Fixed confidence for now
            generation_timestamp=datetime.now()
        )
        
        self.logger.debug(f"Content generated", extra={
            "length": len(processed_content),
            "style": context.content_style.value,
            "has_hashtags": "#" in processed_content
        })
        
        return generated
    
    async def optimize_for_virality(self, content: str, target_audience: Optional[Dict[str, Any]]) -> str:
        """
        Optimize content for viral propagation
        
        Args:
            content: Original content
            target_audience: Target audience info
            
        Returns:
            Optimized content
        """
        # Get trending topics from audience
        trends = target_audience.get("interests", []) if target_audience else []
        
        # Optimize
        optimized = self.virality_optimizer.optimize(content, trends)
        
        self.logger.debug(f"Content optimized for virality", extra={
            "original_length": len(content),
            "optimized_length": len(optimized)
        })
        
        return optimized
    
    async def adapt_style(self, audience_feedback: AudienceFeedback) -> None:
        """
        Adapt content style based on audience feedback
        
        Args:
            audience_feedback: Feedback from audience
        """
        # Analyze feedback
        if audience_feedback.engagement_rate > 0.05:
            # Good engagement, reinforce current style
            self.style_parameters["success_weight"] = \
                self.style_parameters.get("success_weight", 1.0) * 1.1
        else:
            # Poor engagement, try variation
            self.style_parameters["variation_rate"] = \
                self.style_parameters.get("variation_rate", 0.0) + 0.1
        
        self.logger.info(f"Style adapted", extra={
            "content_id": audience_feedback.content_id,
            "engagement_rate": audience_feedback.engagement_rate,
            "sentiment": audience_feedback.sentiment
        })
    
    def generate_hashtags(self, content: str, trending_tags: List[str]) -> List[str]:
        """
        Generate relevant hashtags
        
        Args:
            content: Content text
            trending_tags: Currently trending tags
            
        Returns:
            List of hashtags
        """
        hashtags = []
        
        # Extract topics from content
        topics = self._extract_topics(content)
        
        # Match with trending tags
        for tag in trending_tags[:3]:
            if any(topic in tag.lower() for topic in topics):
                hashtags.append(tag)
        
        # Generate custom tags from topics
        for topic in topics[:2]:
            if topic not in hashtags:
                hashtags.append(topic.capitalize())
        
        self.logger.debug(f"Hashtags generated", extra={
            "count": len(hashtags),
            "hashtags": hashtags
        })
        
        return hashtags[:5]  # Max 5 hashtags
    
    def _load_templates(self, role) -> Dict[str, str]:
        """Load content templates for role"""
        templates = {
            "creator": [
                "Check out {topic}! {insight}",
                "Here's what I learned about {topic}: {insight}",
                "Breaking: {topic} - {insight}"
            ],
            "brand": [
                "Discover {product}: {benefit}",
                "New from {brand}: {announcement}",
                "{product} - {value_prop}"
            ],
            "audience": [
                "Love this! {reaction}",
                "{opinion} about {topic}",
                "Thoughts on {topic}?"
            ]
        }
        
        role_str = str(role).split(".")[-1].lower() if hasattr(role, 'value') else str(role)
        return templates.get(role_str, templates["creator"])
    
    def _create_generation_prompt(self, context: ContentContext) -> str:
        """Create prompt for LLM generation"""
        prompt_parts = [
            f"Generate a {context.content_style.value} social media post",
            f"for audience interested in: {context.personal_brand}"
        ]
        
        if context.trending_topics:
            prompt_parts.append(f"incorporating trends: {', '.join(context.trending_topics[:3])}")
        
        if context.target_audience:
            prompt_parts.append(f"targeting: {context.target_audience.get('description', 'general audience')}")
        
        prompt_parts.append(f"Maximum {context.max_length} characters.")
        
        return " ".join(prompt_parts)
    
    async def _generate_with_llm(self, prompt: str, context: ContentContext) -> str:
        """
        Generate content using LLM (mock implementation)
        
        Args:
            prompt: Generation prompt
            context: Content context
            
        Returns:
            Generated text
        """
        # Mock generation - in production, this would call an actual LLM
        import random
        
        templates = self.content_templates
        template = random.choice(templates)
        
        # Fill template with context
        if context.trending_topics:
            topic = random.choice(context.trending_topics)
        else:
            topic = context.personal_brand
        
        insights = [
            "This is game-changing!",
            "Here's what everyone needs to know.",
            "The future is here.",
            "This changes everything.",
            "You won't believe this."
        ]
        
        content = template.replace("{topic}", topic)
        content = content.replace("{insight}", random.choice(insights))
        content = content.replace("{brand}", context.personal_brand)
        content = content.replace("{product}", context.personal_brand)
        content = content.replace("{benefit}", "innovation and quality")
        content = content.replace("{announcement}", "exciting updates")
        content = content.replace("{value_prop}", "excellence you can trust")
        content = content.replace("{reaction}", "Amazing content!")
        content = content.replace("{opinion}", "Great insights")
        
        return content[:context.max_length]
    
    def _post_process_content(self, content: str, context: ContentContext) -> str:
        """Post-process generated content"""
        # Clean up extra spaces
        content = " ".join(content.split())
        
        # Add hashtags if enabled and not present
        if context.config.enable_hashtags and "#" not in content:
            if context.trending_topics:
                content = f"{content} #{context.trending_topics[0]}"
        
        # Ensure within length limit
        if len(content) > context.max_length:
            content = content[:context.max_length-3] + "..."
        
        return content
    
    def _extract_entities(self, content: str) -> Dict[str, List[Any]]:
        """Extract entities from content"""
        import re
        
        entities = {
            "hashtags": [],
            "mentions": [],
            "urls": []
        }
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', content)
        entities["hashtags"] = [{"tag": tag} for tag in hashtags]
        
        # Extract mentions
        mentions = re.findall(r'@(\w+)', content)
        entities["mentions"] = [{"username": username} for username in mentions]
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        entities["urls"] = [{"url": url} for url in urls]
        
        return entities
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        # Simple topic extraction - split and filter
        words = content.lower().split()
        
        # Filter stop words and short words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
        topics = [w for w in words if len(w) > 4 and w not in stop_words]
        
        return topics[:5]

