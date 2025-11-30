"""
Example: Host local Qwen3-8B API server

This script starts a local API server powered by Qwen3-8B.
Agents can then call http://localhost:8000 to get LLM responses.

Usage:
    python examples/host_qwen_local.py
"""

from Agent.agent.host import create_qwen_host

if __name__ == "__main__":
    # Create host with local Qwen3-8B
    host = create_qwen_host(
        model_path="Qwen/Qwen3-8B",
        device="auto",  # "cuda" for GPU, "cpu" for CPU
        port=8000
    )
    
    # Start server (blocks)
    host.run()
    
    # Server will be available at:
    # - API: http://localhost:8000/v1/chat/completions
    # - Docs: http://localhost:8000/docs
    # - Health: http://localhost:8000/health

