#!/usr/bin/env python3
"""
Flask web server for the Deleuzian Thinking Machine.

Provides a visual interface with real-time streaming responses.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from uuid import uuid4

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, Response, render_template, request, jsonify

import anthropic
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__, static_folder="static", template_folder="static")

# Default settings
VECTOR_STORE = Path("vector_store")
COLLECTION = "deleuze_quotes"
COHERE_MODEL = "embed-v4.0"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250514"
THINKING_BUDGET = 30000
MAX_TOKENS = 64000
GRAPHRAG_OUTPUT = Path("graphrag_project/output")

# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

# In-memory session store (tab session only - lost on server restart)
CONVERSATIONS = {}  # session_id -> {"messages": [], "created_at": datetime, "last_response": str}
MAX_HISTORY_EXCHANGES = 10  # Keep last N user/assistant pairs


def get_or_create_session(session_id: str = None) -> tuple[str, dict]:
    """Get existing session or create new one."""
    if session_id and session_id in CONVERSATIONS:
        return session_id, CONVERSATIONS[session_id]

    # Create new session
    new_id = str(uuid4())
    CONVERSATIONS[new_id] = {
        "messages": [],
        "created_at": datetime.now(),
        "last_response": ""
    }
    return new_id, CONVERSATIONS[new_id]


def truncate_history(messages: list, max_exchanges: int = MAX_HISTORY_EXCHANGES) -> list:
    """Keep only the last N exchanges to manage context window."""
    if len(messages) <= max_exchanges * 2:
        return messages
    # Keep the most recent exchanges
    return messages[-(max_exchanges * 2):]


from agents.deleuzian_agent import create_agent_from_env

def stream_response(question: str, queue: Queue, model_name: str = None, thinking_budget: int = None, history: list = None):
    """
    Stream a response from the Deleuzian agent.
    Sends events to the queue for SSE delivery.

    Args:
        question: The user's question
        queue: Queue for SSE events
        model_name: Optional model override
        thinking_budget: Optional thinking budget override
        history: Optional conversation history
    """
    try:
        # Initialize agent using factory
        agent = create_agent_from_env(model_name=model_name, thinking_budget=thinking_budget)

        # Stream events directly from the agent generator with history
        for event in agent.stream_respond(question, history=history):
            queue.put(event)

    except Exception as e:
        queue.put({"type": "error", "message": str(e)})
        queue.put({"type": "done"})


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main interface."""
    return render_template("index.html")


@app.route("/graph")
def graph_view():
    """Serve the graph visualization interface."""
    return render_template("graph.html")


@app.route("/api/graph")
def get_graph_data():
    """Get graph data (nodes and edges) from parquet files."""
    try:
        limit = request.args.get("limit", 500, type=int)
        
        entities_path = GRAPHRAG_OUTPUT / "entities.parquet"
        relationships_path = GRAPHRAG_OUTPUT / "relationships.parquet"
        
        if not entities_path.exists() or not relationships_path.exists():
            return jsonify({"error": "Graph data not found"}), 404
            
        # Load entities
        df_entities = pd.read_parquet(entities_path)
        
        # Sort by degree (if available) or randomly top N
        if "degree" in df_entities.columns:
            df_entities = df_entities.sort_values("degree", ascending=False)
            
        top_entities = df_entities.head(limit)
        valid_titles = set(top_entities["title"].values)
        
        # Create nodes
        nodes = []
        for _, row in top_entities.iterrows():
            nodes.append({
                "id": row["title"],
                "label": row["title"],
                "value": int(row["degree"]) if "degree" in row else 1,
                "title": row["description"] if "description" in row else "",
                "group": row["type"] if "type" in row else "unknown"
            })
            
        # Load relationships
        df_rels = pd.read_parquet(relationships_path)
        
        # Filter relationships where both source and target are in top entities
        mask = df_rels["source"].isin(valid_titles) & df_rels["target"].isin(valid_titles)
        filtered_rels = df_rels[mask]
        
        edges = []
        for _, row in filtered_rels.iterrows():
            edges.append({
                "from": row["source"],
                "to": row["target"],
                "value": float(row["weight"]) if "weight" in row else 1.0,
                # "title": row["description"] if "description" in row else ""
            })
            
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/ask", methods=["POST"])
def ask():
    """Handle a question and stream the response with conversation memory."""
    data = request.get_json()
    question = data.get("question", "").strip()
    model_name = data.get("model")
    thinking_budget = data.get("thinking_budget")
    session_id = data.get("session_id")

    if thinking_budget is not None:
        try:
            thinking_budget = int(thinking_budget)
        except ValueError:
            thinking_budget = None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Get or create session
    session_id, session = get_or_create_session(session_id)

    # Get truncated history for context
    history = truncate_history(session["messages"].copy())

    # Track response content for saving to history
    response_content = []

    def generate():
        nonlocal response_content
        queue = Queue()
        thread = Thread(
            target=stream_response,
            args=(question, queue, model_name, thinking_budget, history)
        )
        thread.start()

        # First event: send session_id
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        while True:
            event = queue.get()
            yield f"data: {json.dumps(event)}\n\n"

            # Collect response content
            if event.get("type") == "content":
                response_content.append(event.get("content", ""))

            if event.get("type") == "done":
                break

        thread.join()

        # Save to session history after streaming completes
        full_response = "".join(response_content)
        session["messages"].append({"role": "user", "content": question})
        session["messages"].append({"role": "assistant", "content": full_response})
        session["last_response"] = full_response

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting Deleuzian Thinking Machine...")
    print("Open http://localhost:5001 in your browser")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
