#!/usr/bin/env python3
"""
Test if the Deleuzian Agent autonomously uses the graph traversal tool.
"""
import sys
import json
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
    print("Initializing agent...")
    try:
        agent = create_agent_from_env()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Question specifically designed to trigger traversal
    question = "Who is connected to Freud in the rhizome? Trace the connections."
    
    print(f"\nAsking: '{question}'\n")
    print("--- Streaming Response & Tool Calls ---")
    
    tool_calls_seen = []
    
    try:
        # We use stream_respond to see events in real-time
        for event in agent.stream_respond(question):
            etype = event.get("type")
            
            if etype == "tool_call":
                tname = event.get("tool")
                tquery = event.get("query")
                print(f"\n[TOOL CALL] {tname}: {tquery}")
                tool_calls_seen.append(tname)
                
            elif etype == "content":
                # print chunk (optional, maybe just dots to show aliveness)
                sys.stdout.write(".")
                sys.stdout.flush()
                
            elif etype == "thinking":
                # Print thinking to see reasoning
                tcontent = event.get("content", "")
                sys.stdout.write(f"[THINKING] {tcontent}\n")
                sys.stdout.flush()
                
            elif etype == "error":
                print(f"\n[ERROR] {event.get('message')}")
                
        print("\n\n--- Done ---")
        
        if "traverse_relationships" in tool_calls_seen:
            print("✅ SUCCESS: Agent used 'traverse_relationships'.")
        else:
            print("❌ FAILURE: Agent did NOT use 'traverse_relationships'.")
            print(f"Tools used: {tool_calls_seen}")

    except Exception as e:
        print(f"\nError during response: {e}")

if __name__ == "__main__":
    main()
