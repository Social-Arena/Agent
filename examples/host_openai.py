"""
Example: Host API server with OpenAI backend

This script starts a local API server that proxies requests to OpenAI.
Useful for providing a unified interface across different backends.

Usage:
    export OPENAI_API_KEY=your_key
    python examples/host_openai.py
"""

from Agent.agent.host import create_openai_host

if __name__ == "__main__":
    # Create host with OpenAI backend
    # API key is read from OPENAI_API_KEY environment variable
    host = create_openai_host(port=8000)
    
    # Start server (blocks)
    host.run()
    
    # Server will be available at:
    # - API: http://localhost:8000/v1/chat/completions
    # - Docs: http://localhost:8000/docs

