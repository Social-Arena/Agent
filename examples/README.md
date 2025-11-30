# Examples 📚

This directory contains example scripts for using the Agent system.

## Host API Servers

### 1. Local Qwen3-8B Server

```bash
python examples/host_qwen_local.py
```

Starts a local API server powered by Qwen3-8B model. Requires GPU for best performance.

**Requirements:**
- GPU with 16GB+ VRAM (recommended)
- `transformers>=4.51.0`
- `torch>=2.0.0`

**Endpoints:**
- `POST /v1/chat/completions` - OpenAI-compatible chat endpoint
- `GET /v1/models` - List available models
- `GET /health` - Health check

### 2. OpenAI Proxy Server

```bash
export OPENAI_API_KEY=your_key
python examples/host_openai.py
```

Starts a local API server that proxies requests to OpenAI GPT-4o.

**Use cases:**
- Unified interface for different backends
- Cost tracking and rate limiting
- Local caching layer

### 3. Anthropic Proxy Server

```bash
export ANTHROPIC_API_KEY=your_key
python examples/host_anthropic.py
```

Starts a local API server that proxies requests to Anthropic Claude.

**Use cases:**
- Use Claude with OpenAI-compatible API
- Switch between providers without code changes
- Multi-model routing

## Agent with LLM

### AI Agent Demo

```bash
# Terminal 1: Start any host server
python examples/host_qwen_local.py

# Terminal 2: Run agent
python examples/agent_with_host.py
```

Shows how agents use the hosted API for:
- Decision making (which action to take)
- Content generation (creating posts)
- Context understanding

## Architecture

```
┌─────────────────────────────────────────────┐
│          Multiple Agents                     │
│  (call http://localhost:8000/v1/...)        │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│      Language Model Host API Server          │
│      (FastAPI with OpenAI endpoints)         │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ Qwen3  │  │ OpenAI │  │Anthropic│
   │ Local  │  │  API   │  │  API   │
   └────────┘  └────────┘  └────────┘
```

## Tips

1. **For development:** Use OpenAI/Anthropic for fast iteration
2. **For production:** Use local Qwen3 for cost efficiency
3. **For research:** Compare all three backends
4. **Multiple agents:** All agents share one API server

## Environment Variables

```bash
# For OpenAI backend
export OPENAI_API_KEY=sk-...

# For Anthropic backend
export ANTHROPIC_API_KEY=sk-ant-...
```

## Advanced Usage

### Custom Backend Configuration

```python
from Agent.agent.host import LanguageModelHost, BackendConfig, BackendProvider

# Custom Qwen configuration
config = BackendConfig(
    provider=BackendProvider.QWEN_LOCAL,
    model_path="Qwen/Qwen3-8B",
    device="cuda:0",  # Specific GPU
    torch_dtype="bfloat16"
)

host = LanguageModelHost(config, port=8000)
host.run()
```

### Multiple Servers

```bash
# Start multiple servers on different ports
python examples/host_qwen_local.py &  # Port 8000
python examples/host_openai.py &      # Port 8001 (modify code)
```

## Troubleshooting

**Model loading is slow:**
- First load downloads from Hugging Face (~16GB for Qwen3-8B)
- Subsequent loads are faster (cached)

**Out of memory:**
- Use smaller model or CPU mode
- Close other GPU applications

**API connection refused:**
- Make sure host server is running
- Check port isn't already in use

