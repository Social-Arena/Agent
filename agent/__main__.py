"""
Agent CLI - Command Line Interface for Social Arena Agent
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from .host import create_qwen_host, create_openai_host, create_anthropic_host, BackendProvider

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Social Arena Agent - LLM Host Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start OpenAI host
  python -m Agent --provider openai --port 8000
  
  # Start Anthropic host
  python -m Agent --provider anthropic --port 8000
  
  # Start local Qwen host
  python -m Agent --provider qwen --model-path Qwen/Qwen3-8B --port 8000
  
  # Use custom API key
  python -m Agent --provider openai --api-key YOUR_KEY --port 8000
"""
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "anthropic", "qwen"],
        help="LLM provider to use (openai, anthropic, qwen)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI/Anthropic (default: from environment variables)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path for Qwen local backend (default: Qwen/Qwen3-8B)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for Qwen local backend (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Create host based on provider
    print(f"\n🚀 Starting {args.provider.upper()} host on port {args.port}...")
    
    if args.provider == "openai":
        host = create_openai_host(api_key=args.api_key, port=args.port)
    elif args.provider == "anthropic":
        host = create_anthropic_host(api_key=args.api_key, port=args.port)
    elif args.provider == "qwen":
        host = create_qwen_host(
            model_path=args.model_path,
            device=args.device,
            port=args.port
        )
    else:
        print(f"Error: Unknown provider '{args.provider}'")
        sys.exit(1)
    
    # Run the host
    host.run()


if __name__ == "__main__":
    main()

