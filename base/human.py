"""
Human User - Human-controlled actor in Social Arena
"""

from pydantic import Field
from Agent.base.actor import Actor, ActorType


class HumanUser(Actor):
    """
    Human-controlled user in Social Arena
    
    Same 12 fundamental actions as AI agents, but controlled by human input
    """
    
    actor_type: str = Field(default=ActorType.HUMAN_USER)
    
    # Human-specific state
    online: bool = Field(default=False, description="Whether user is currently online")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Human users don't need autonomous behavior
        self.logger.info(
            "Human user created",
            extra={"username": self.username}
        )
    
    def decide_next_action(self, context):
        """
        Human users don't decide autonomously - they wait for input
        
        This method should not be called for human users.
        Use the 12 fundamental actions directly based on UI input.
        """
        return "idle"

