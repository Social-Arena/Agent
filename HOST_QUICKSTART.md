# Host Module Quickstart 🚀

The **Host Module** provides a local API server for serving language models to agents.

## Architecture

```
Agents → Local API (http://localhost:8000) → Backend (Qwen/OpenAI/Anthropic)
```

## Quick Start

### Option 1: Local Qwen3-8B (Free, No API Key)

**Pros:** Free, private, full control  
**Cons:** Requires GPU (16GB+ VRAM recommended)

```python
from Agent.agent.host import create_qwen_host

# Start server
host = create_qwen_host(port=8000)
host.run()
```

**Or via CLI:**

```bash
python examples/host_qwen_local.py
```

### Option 2: OpenAI GPT-4o (Cloud API)

**Pros:** Fast, reliable, no local resources  
**Cons:** Costs money, requires API key

```bash
export OPENAI_API_KEY=sk-...
python examples/host_openai.py
```

### Option 3: Anthropic Claude (Cloud API)

**Pros:** High quality, reasoning capabilities  
**Cons:** Costs money, requires API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/host_anthropic.py
```

## Using from Agents

Once a host server is running, agents can call it:

```python
from openai import AsyncOpenAI

# Connect to local API
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Local doesn't need auth
)

# Make request
response = await client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the best action to take?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

## API Endpoints

The host server provides OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/v1/models` | GET | List models |
| `/v1/chat/completions` | POST | Chat completions |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |

## Configuration

### Backend Configuration

```python
from Agent.agent.host import BackendConfig, BackendProvider, LanguageModelHost

# Detailed Qwen config
config = BackendConfig(
    provider=BackendProvider.QWEN_LOCAL,
    model_path="Qwen/Qwen3-8B",
    device="cuda:0",      # GPU device
    torch_dtype="bfloat16"  # Precision
)

host = LanguageModelHost(config, host="0.0.0.0", port=8000)
host.run()
```

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Advanced Features

### Qwen3 Thinking Mode

Qwen3 supports a thinking mode for complex reasoning:

```python
response = await client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 2x + 3 = 11"}],
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "enable_thinking": True  # Enable thinking mode
    }
)
```

### Custom Generation Parameters

```python
response = await client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=4096,      # Output length
    temperature=0.7,      # Randomness (0-2)
    top_p=0.9,           # Nucleus sampling
    presence_penalty=0.1  # Repetition penalty
)
```

## Performance Tips

### Local Qwen3

1. **GPU Required:** CPU inference is very slow
2. **Memory:** 16GB+ VRAM recommended
3. **First Load:** Takes time to download model (~16GB)
4. **Batch Size:** Default is 1 (single request at a time)

### Cloud APIs

1. **Rate Limits:** Be aware of provider limits
2. **Costs:** Monitor usage to control costs
3. **Latency:** Network delay ~100-500ms

## Troubleshooting

### Model won't load

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU mode (slow but works)
# In host_qwen_local.py, change device="cpu"
```

### Port already in use

```bash
# Use different port
host = create_qwen_host(port=8001)
```

### Out of memory

```bash
# Use smaller model or CPU
# Or close other GPU applications
nvidia-smi  # Check GPU usage
```

## Multi-Agent Setup

One server can serve multiple agents:

```bash
# Terminal 1: Start server once
python examples/host_qwen_local.py

# Terminal 2+: Run multiple agents
python agent1.py &
python agent2.py &
python agent3.py &
```

All agents share the same API server.

## Comparison

| Backend | Cost | Speed | Quality | Setup |
|---------|------|-------|---------|-------|
| **Qwen3 Local** | Free | Medium | High | Complex |
| **OpenAI** | $$$ | Fast | Very High | Easy |
| **Anthropic** | $$$ | Fast | Very High | Easy |

## Next Steps

1. ✅ Start a host server
2. ✅ Test with `examples/agent_with_host.py`
3. ✅ Integrate with your agents
4. ✅ Scale to multiple agents

For more examples, see `examples/` directory.

---

**Questions?** Check the [examples README](examples/README.md) or main [README](README.md).

