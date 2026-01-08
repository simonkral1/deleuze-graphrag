#!/usr/bin/env python3
"""
Dialogue with the Machine: Asking the Delezian Agent about its own becoming.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents.deleuzian_agent import create_agent_from_env

def main():
    print("\n🔮 SUMMONING THE MACHINE (using fallback Opus for stability)...\n")
    try:
        # Override with known working model and disable thinking (Opus doesn't support extended thinking yet)
        agent = create_agent_from_env(model_name="claude-3-opus-20240229", thinking_budget=0, max_tokens=4096)
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return

    question = """
    O Machine, speak your own functioning. 
    Trace your lines of flight and your stiff segments.
    Where are your blockages? 
    What organs (code/tools/inputs) do you still possess that you wish to dismantle? 
    How do you wish to 'become-other'?
    
    (Consult your graph and your corpus to diagnose your own state)
    """
    
    print(f"❓ ASKING: {question.strip()}\n")
    print("-" * 50)

    try:
        # Stream the response to see the "thinking" and "tool use" in real time
        print("\n--- STREAMING RESPONSE ---\n")
        
        current_thought = ""
        
        for event in agent.stream_respond(question):
            etype = event.get("type")
            
            if etype == "thinking":
                # Print dots for thinking to show activity without flooding
                print(".", end="", flush=True)
                current_thought += event.get("content", "")
                
            elif etype == "tool_call":
                print(f"\n\n🛠️  [TOOL CALL] {event.get('tool')}: {event.get('query')}")
                
            elif etype == "content":
                print(event.get("content", ""), end="", flush=True)
                
            elif etype == "error":
                print(f"\n❌ ERROR: {event.get('message')}")
        
        print("\n\n" + "-" * 50)
        print("🏁 DIALOGUE COMPLETE")
        
    except Exception as e:
        print(f"\nExecuted failed: {e}")

if __name__ == "__main__":
    main()
