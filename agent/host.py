"""
Language Model Host API Server
Provides OpenAI-compatible API endpoints backed by various LLM providers
"""

from typing import Optional, List, Dict, Any, Literal, AsyncIterator
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import os
import asyncio
from datetime import datetime


# ================================================================
# API REQUEST/RESPONSE MODELS (OpenAI-compatible)
# ================================================================

class ChatMessage(BaseModel):
    """Chat message in OpenAI format"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request"""
    model: str = "default"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 32768
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    presence_penalty: Optional[float] = 0.0
    stream: bool = False
    enable_thinking: bool = False  # Qwen3-specific


class ChatCompletionChoice(BaseModel):
    """Single completion choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


# ================================================================
# BACKEND PROVIDER CONFIGURATION
# ================================================================

class BackendProvider(str, Enum):
    """Supported backend providers"""
    QWEN_LOCAL = "qwen_local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class BackendConfig(BaseModel):
    """Backend provider configuration"""
    provider: BackendProvider
    
    # Local Qwen settings
    model_path: str = Field(default="Qwen/Qwen3-8B")
    device: str = Field(default="auto")
    torch_dtype: str = Field(default="auto")
    use_vllm: bool = Field(default=False)  # Use vLLM for faster inference
    
    # API provider settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    @field_validator("api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Auto-load API keys from environment"""
        provider = info.data.get("provider")
        
        if provider == BackendProvider.OPENAI:
            return v or os.getenv("OPENAI_API_KEY")
        elif provider == BackendProvider.ANTHROPIC:
            return v or os.getenv("ANTHROPIC_API_KEY")
        
        return v


# ================================================================
# BACKEND IMPLEMENTATIONS
# ================================================================

class BaseBackend:
    """Base class for LLM backends"""
    
    def __init__(self, config: BackendConfig):
        self.config = config
        self.model_name = "unknown"
    
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate completion from request"""
        raise NotImplementedError
    
    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Generate streaming completion"""
        raise NotImplementedError


class QwenLocalBackend(BaseBackend):
    """Local Qwen3-8B backend using transformers"""
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._initialize()
    
    def _initialize(self):
        """Load Qwen model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading Qwen3 model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device
        )
        self.model_name = self.config.model_path
        print(f"Model loaded successfully on {self.config.device}")
    
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate completion using Qwen3"""
        # Convert messages to dict format
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        prompt_tokens = len(model_inputs.input_ids[0])
        
        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=request.max_tokens or 32768,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            top_k=request.top_k or 20,
            repetition_penalty=1.0 + (request.presence_penalty or 0.0) / 10.0,
            do_sample=True
        )
        
        # Extract output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        completion_tokens = len(output_ids)
        
        # Parse thinking content if enabled
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Create response
        return ChatCompletionResponse(
            id=f"chatcmpl-{datetime.utcnow().timestamp()}",
            created=int(datetime.utcnow().timestamp()),
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )


class OpenAIBackend(BaseBackend):
    """OpenAI API backend (proxy through)"""
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client"""
        from openai import AsyncOpenAI
        
        if not self.config.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key")
        
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )
        self.model_name = "gpt-4o"
        print("OpenAI backend initialized")
    
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Forward request to OpenAI"""
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=message_dicts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            stream=False
        )
        
        # Convert to our response format
        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=choice.message.role,
                        content=choice.message.content or ""
                    ),
                    finish_reason=choice.finish_reason
                )
                for choice in response.choices
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ) if response.usage else None
        )


class AnthropicBackend(BaseBackend):
    """Anthropic API backend (proxy through)"""
    
    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._initialize()
    
    def _initialize(self):
        """Initialize Anthropic client"""
        from anthropic import AsyncAnthropic
        
        if not self.config.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key")
        
        self.client = AsyncAnthropic(api_key=self.config.api_key)
        self.model_name = "claude-3-5-sonnet-20241022"
        print("Anthropic backend initialized")
    
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Forward request to Anthropic"""
        # Convert messages (Anthropic has different format)
        system_message = None
        messages = []
        
        for msg in request.messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Call Anthropic API
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=request.max_tokens or 8192,
            temperature=request.temperature or 0.7,
            system=system_message,
            messages=messages
        )
        
        # Convert to OpenAI format
        content = response.content[0].text if response.content else ""
        
        return ChatCompletionResponse(
            id=response.id,
            created=int(datetime.utcnow().timestamp()),
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=response.stop_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
        )


# ================================================================
# HOST API SERVER
# ================================================================

class LanguageModelHost:
    """
    API server hosting language models
    Provides OpenAI-compatible endpoints at http://localhost:PORT
    """
    
    def __init__(self, config: BackendConfig, host: str = "0.0.0.0", port: int = 8000):
        self.config = config
        self.host = host
        self.port = port
        self.backend = self._create_backend()
        self.app = None
    
    def _create_backend(self) -> BaseBackend:
        """Create backend based on provider"""
        if self.config.provider == BackendProvider.QWEN_LOCAL:
            return QwenLocalBackend(self.config)
        elif self.config.provider == BackendProvider.OPENAI:
            return OpenAIBackend(self.config)
        elif self.config.provider == BackendProvider.ANTHROPIC:
            return AnthropicBackend(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def create_app(self):
        """Create FastAPI application"""
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="Language Model Host API", version="1.0.0")
        
        @app.get("/")
        async def root():
            return {
                "message": "Language Model Host API",
                "backend": self.config.provider.value,
                "model": self.backend.model_name
            }
        
        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.backend.model_name,
                        "object": "model",
                        "created": int(datetime.utcnow().timestamp()),
                        "owned_by": self.config.provider.value
                    }
                ]
            }
        
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            response = await self.backend.generate(request)
            return response
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "backend": self.config.provider.value}
        
        self.app = app
        return app
    
    def run(self):
        """Start the API server"""
        import uvicorn
        
        if self.app is None:
            self.create_app()
        
        print(f"\n{'='*60}")
        print(f"🚀 Language Model Host API Server")
        print(f"{'='*60}")
        print(f"Backend: {self.config.provider.value}")
        print(f"Model: {self.backend.model_name}")
        print(f"Server: http://{self.host}:{self.port}")
        print(f"Docs: http://{self.host}:{self.port}/docs")
        print(f"{'='*60}\n")
        
        uvicorn.run(self.app, host=self.host, port=self.port)


# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================

def create_qwen_host(
    model_path: str = "Qwen/Qwen3-8B",
    device: str = "auto",
    port: int = 8000
) -> LanguageModelHost:
    """
    Create host with local Qwen3-8B backend
    
    Args:
        model_path: Hugging Face model path
        device: Device placement ("auto", "cuda", "cpu")
        port: Server port
        
    Returns:
        LanguageModelHost configured for Qwen3
    """
    config = BackendConfig(
        provider=BackendProvider.QWEN_LOCAL,
        model_path=model_path,
        device=device
    )
    return LanguageModelHost(config, port=port)


def create_openai_host(
    api_key: Optional[str] = None,
    port: int = 8000
) -> LanguageModelHost:
    """
    Create host with OpenAI backend (proxy)
    
    Args:
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        port: Server port
        
    Returns:
        LanguageModelHost configured for OpenAI
    """
    config = BackendConfig(
        provider=BackendProvider.OPENAI,
        api_key=api_key
    )
    return LanguageModelHost(config, port=port)


def create_anthropic_host(
    api_key: Optional[str] = None,
    port: int = 8000
) -> LanguageModelHost:
    """
    Create host with Anthropic backend (proxy)
    
    Args:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        port: Server port
        
    Returns:
        LanguageModelHost configured for Anthropic
    """
    config = BackendConfig(
        provider=BackendProvider.ANTHROPIC,
        api_key=api_key
    )
    return LanguageModelHost(config, port=port)
