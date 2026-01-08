import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env manually
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key] = value

from agents.deleuzian_agent import create_agent_from_env

def test_agent_config():
    print("Testing agent configuration...")
    
    # Test 1: Default values
    print("1. Testing defaults...")
    agent = create_agent_from_env()
    print(f"   Model: {agent.model}")
    print(f"   Budget: {agent.thinking_budget}")
    
    # Defaults might depend on env vars, but let's just ensure they exist
    assert agent.model is not None
    assert isinstance(agent.thinking_budget, int)

    # Test 2: Custom values
    print("\n2. Testing custom values...")
    custom_model = "claude-test-model"
    custom_budget = 12345
    
    agent = create_agent_from_env(model_name=custom_model, thinking_budget=custom_budget)
    print(f"   Model: {agent.model}")
    print(f"   Budget: {agent.thinking_budget}")
    
    if agent.model != custom_model:
        print(f"❌ FAIL: Expected model {custom_model}, got {agent.model}")
        sys.exit(1)
        
    if agent.thinking_budget != custom_budget:
        print(f"❌ FAIL: Expected budget {custom_budget}, got {agent.thinking_budget}")
        sys.exit(1)
        
    print("\n✅ SUCCESS: Agent configuration verified.")

if __name__ == "__main__":
    try:
        test_agent_config()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
