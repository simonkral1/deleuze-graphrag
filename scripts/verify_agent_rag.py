#!/usr/bin/env python3
"""
Verify Deleuzian Agent's integration with GraphRAG.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file.")
except ImportError:
    print("python-dotenv not found, assuming env vars are set.")

from agents.deleuzian_agent import create_agent_from_env

def main():
    print("Initializing agent from environment...")
    try:
        agent = create_agent_from_env()
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    print("\n--- Testing Corpus Search (LanceDB) ---")
    if agent.corpus_tool:
        try:
            results = agent.corpus_tool.search("body without organs", n_results=3)
            for r in results:
                print(f"[ID: {r['id']}] {r['text'][:100]}...")
            if not results:
                print("No results found.")
        except Exception as e:
            print(f"Corpus search failed: {e}")
    else:
        print("Corpus tool not initialized.")

    print("\n--- Testing Graph Concept Search ---")
    if agent.graph_tool:
        try:
            results = agent.graph_tool.search("assemblage", top_k=3)
            for r in results:
                print(f"[Concept: {r['title']}] {r['summary'][:100]}...")
            if not results:
                print("No results found.")
        except Exception as e:
            print(f"Graph search failed: {e}")
    else:
        print("Graph tool not initialized.")

    print("\n--- Testing Graph Traversal ---")
    if agent.traversal_tool:
        try:
            # Try a known entity, assuming 'Rhizome' or similar exists
            # We inspected entities earlier, let's try a generic one or 'Deleuze'
            results = agent.traversal_tool.get_connections("Deleuze", top_k=3)
            for r in results:
                print(f"[Related: {r['concept']}] {r['description'][:100]}...")
            if not results:
                print("No connections found for 'Deleuze'.")
        except Exception as e:
            print(f"Traversal failed: {e}")
    else:
        print("Traversal tool not initialized.")

if __name__ == "__main__":
    main()
