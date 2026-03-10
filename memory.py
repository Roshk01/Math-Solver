# memory.py
# This file handles Memory - storing past solved problems and reusing them
# Simple JSON file storage (no database needed!)

import json
import os
from datetime import datetime


MEMORY_FILE = "data/memory.json"


def _load_memory() -> list:
    """Load all memories from JSON file"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def _save_memory(memories: list):
    """Save all memories to JSON file"""
    os.makedirs("data", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def save_to_memory(
    original_input: str,
    parsed_problem: dict,
    solution: dict,
    explanation: str,
    verifier_result: dict,
    user_feedback: str = None
):
    """
    Save a complete solved problem to memory.
    Called after every successful solution.
    
    user_feedback: "correct", "incorrect", or None (not yet rated)
    """
    memories = _load_memory()
    
    entry = {
        "id": len(memories) + 1,
        "timestamp": datetime.now().isoformat(),
        "original_input": original_input,
        "parsed_problem": parsed_problem,
        "solution": solution,
        "explanation": explanation,
        "verifier_result": verifier_result,
        "user_feedback": user_feedback,
        "topic": parsed_problem.get("topic", "unknown")
    }
    
    memories.append(entry)
    _save_memory(memories)
    print(f"✅ Saved to memory (total: {len(memories)} entries)")
    return entry["id"]


def update_feedback(memory_id: int, feedback: str, comment: str = ""):
    """
    Update user feedback for a memory entry.
    Called when user clicks ✅ or ❌ button.
    
    feedback: "correct" or "incorrect"
    comment: optional comment if incorrect
    """
    memories = _load_memory()
    
    for entry in memories:
        if entry["id"] == memory_id:
            entry["user_feedback"] = feedback
            entry["feedback_comment"] = comment
            entry["feedback_time"] = datetime.now().isoformat()
            break
    
    _save_memory(memories)


def find_similar_problems(query: str, topic: str = None, limit: int = 3) -> list:
    """
    Find similar past problems from memory.
    Simple keyword matching (no embeddings needed for memory).
    
    Returns list of similar past entries to help with current problem.
    """
    memories = _load_memory()
    
    if not memories:
        return []
    
    # Filter by topic if given
    if topic:
        memories = [m for m in memories if m.get("topic") == topic]
    
    # Only use memories that were marked as correct (or not rated yet)
    good_memories = [
        m for m in memories 
        if m.get("user_feedback") != "incorrect"
    ]
    
    # Simple keyword matching
    query_words = set(query.lower().split())
    scored = []
    
    for entry in good_memories:
        problem_text = entry.get("parsed_problem", {}).get("problem_text", "")
        problem_words = set(problem_text.lower().split())
        
        # Score = number of common words
        overlap = len(query_words & problem_words)
        if overlap > 0:
            scored.append((overlap, entry))
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return top matches
    return [entry for _, entry in scored[:limit]]


def get_all_memories() -> list:
    """Return all saved memories (for display in UI)"""
    return _load_memory()


def get_memory_stats() -> dict:
    """Return stats about memory (for display)"""
    memories = _load_memory()
    
    if not memories:
        return {"total": 0, "correct": 0, "incorrect": 0, "topics": {}}
    
    correct = sum(1 for m in memories if m.get("user_feedback") == "correct")
    incorrect = sum(1 for m in memories if m.get("user_feedback") == "incorrect")
    
    topics = {}
    for m in memories:
        t = m.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
    
    return {
        "total": len(memories),
        "correct": correct,
        "incorrect": incorrect,
        "unrated": len(memories) - correct - incorrect,
        "topics": topics
    }
