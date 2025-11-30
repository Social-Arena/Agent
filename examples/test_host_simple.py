"""
Simple test script for host module (without running server)

This tests the backend configuration and setup without starting the full API server.
Useful for quick validation.
"""

from Agent.agent.host import BackendConfig, BackendProvider


def test_config_validation():
    """Test configuration validation"""
    print("Testing configuration validation...\n")
    
    # Test 1: Qwen Local Config
    print("✓ Testing Qwen local config...")
    qwen_config = BackendConfig(
        provider=BackendProvider.QWEN_LOCAL,
        model_path="Qwen/Qwen3-8B",
        device="cpu",  # Use CPU for testing
        torch_dtype="float32"
    )
    print(f"  Provider: {qwen_config.provider.value}")
    print(f"  Model: {qwen_config.model_path}")
    print(f"  Device: {qwen_config.device}\n")
    
    # Test 2: OpenAI Config (with mock key)
    print("✓ Testing OpenAI config...")
    import os
    os.environ["OPENAI_API_KEY"] = "test-key"
    openai_config = BackendConfig(
        provider=BackendProvider.OPENAI
    )
    print(f"  Provider: {openai_config.provider.value}")
    print(f"  API Key: {'*' * 8} (loaded from env)\n")
    
    # Test 3: Anthropic Config (with mock key)
    print("✓ Testing Anthropic config...")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    anthropic_config = BackendConfig(
        provider=BackendProvider.ANTHROPIC
    )
    print(f"  Provider: {anthropic_config.provider.value}")
    print(f"  API Key: {'*' * 8} (loaded from env)\n")
    
    print("=" * 60)
    print("✅ All configuration tests passed!")
    print("=" * 60)


def show_usage():
    """Show usage examples"""
    print("\nUsage Examples:")
    print("\n1. Start Qwen3-8B server:")
    print("   python examples/host_qwen_local.py")
    
    print("\n2. Start OpenAI proxy:")
    print("   export OPENAI_API_KEY=your_key")
    print("   python examples/host_openai.py")
    
    print("\n3. Start Anthropic proxy:")
    print("   export ANTHROPIC_API_KEY=your_key")
    print("   python examples/host_anthropic.py")
    
    print("\n4. Use from agent:")
    print("   python examples/agent_with_host.py")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Host Module Configuration Test")
    print("=" * 60)
    print()
    
    test_config_validation()
    show_usage()

