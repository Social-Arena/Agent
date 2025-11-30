"""
Example: Host API server with Anthropic backend

This script starts a local API server that proxies requests to Anthropic.
Useful for using Claude models with OpenAI-compatible API.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python examples/host_anthropic.py
"""

from Agent.agent.host import create_anthropic_host

if __name__ == "__main__":
    # Create host with Anthropic backend
    # API key is read from ANTHROPIC_API_KEY environment variable
    host = create_anthropic_host(port=8000)
    
    # Start server (blocks)
    host.run()
    
    # Server will be available at:
    # - API: http://localhost:8000/v1/chat/completions
    # - Docs: http://localhost:8000/docs

