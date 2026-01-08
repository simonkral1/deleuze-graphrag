#!/usr/bin/env python3
"""
Verify that CorpusSearchTool correctly resolves book titles from documents.parquet.
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
    print("Initializing Agent...")
    try:
        agent = create_agent_from_env()
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return

    if not agent.corpus_tool:
        print("Corpus tool not loaded.")
        return

    query = "body without organs"
    print(f"\nSearching for: '{query}'")
    
    try:
        # Direct tool search (bypassing agent _handle_tool_call formatting if we want raw dicts)
        # But _handle_tool_call does the formatting string construction.
        # Let's inspect the raw search result from the tool instance first to check the dict keys.
        
        results = agent.corpus_tool.search(query, n_results=3)
        
        print(f"\nFound {len(results)} results.\n")
        
        for i, r in enumerate(results):
            title = r.get("book_title", "MISSING_KEY")
            text_snippet = r['text'][:100].replace('\n', ' ')
            print(f"[{i+1}] Title: '{title}'")
            print(f"     Text: {text_snippet}...")
            print("-" * 40)
            
        # Verify specific expectations
        if any(r.get("book_title") and r.get("book_title") != "Unknown Source" for r in results):
            print("\n✅ SUCCESS: Book titles are being resolved!")
        else:
            print("\n❌ FAILURE: All titles are 'Unknown Source' or missing.")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    main()
