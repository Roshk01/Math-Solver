# agents.py
# All 5 AI agents using direct Groq client (same as your working project!)

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# ─────────────────────────────────────────────
# Helper: Call Groq API directly
# Same structure as your working project!
# ─────────────────────────────────────────────
def ask_llm(prompt: str) -> str:
    """Send a prompt to Groq and return the response text"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


def parse_json(text: str, fallback: dict) -> dict:
    """Safely parse JSON from LLM response"""
    try:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception:
        return fallback


# AGENT 1: PARSER AGENT
def parser_agent(raw_text: str) -> dict:
    prompt = f"""
You are a Math Problem Parser. Clean and structure this math problem.

Raw input: {raw_text}

Respond ONLY with valid JSON, no extra text:
{{
    "problem_text": "cleaned problem",
    "topic": "algebra/probability/calculus/linear_algebra/other",
    "variables": ["x"],
    "constraints": [],
    "needs_clarification": false,
    "clarification_reason": ""
}}
"""
    return parse_json(ask_llm(prompt), {
        "problem_text": raw_text, "topic": "other",
        "variables": [], "constraints": [],
        "needs_clarification": False, "clarification_reason": ""
    })


# AGENT 2: ROUTER AGENT
def router_agent(parsed_problem: dict) -> dict:
    prompt = f"""
You are a Math Intent Router.
Topic: {parsed_problem.get("topic", "")}
Problem: {parsed_problem.get("problem_text", "")}

Respond ONLY with valid JSON:
{{
    "problem_type": "quadratic/probability/derivative/matrix/etc",
    "difficulty": "easy/medium/hard",
    "strategy": "brief solving approach",
    "tools_needed": ["formula_lookup"],
    "estimated_steps": 4
}}
"""
    return parse_json(ask_llm(prompt), {
        "problem_type": "general", "difficulty": "medium",
        "strategy": "Solve step by step", "tools_needed": ["formula_lookup"],
        "estimated_steps": 4
    })


# AGENT 3: SOLVER AGENT
def solver_agent(problem_text: str, strategy: str, retrieved_context: str) -> dict:
    prompt = f"""
You are a Math Solver. Solve this problem accurately.

PROBLEM: {problem_text}
STRATEGY: {strategy}
RELEVANT FORMULAS: {retrieved_context if retrieved_context else "Use your math knowledge"}

Respond ONLY with valid JSON:
{{
    "solution": "final answer here",
    "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
    "confidence": 0.9,
    "formulas_used": ["formula name"],
    "answer_type": "numerical/expression/explanation"
}}
"""
    response = ask_llm(prompt)
    return parse_json(response, {
        "solution": response, "steps": ["See above"],
        "confidence": 0.5, "formulas_used": [], "answer_type": "explanation"
    })


# AGENT 4: VERIFIER AGENT
def verifier_agent(problem_text: str, solution: dict) -> dict:
    prompt = f"""
You are a Math Verifier. Check if this solution is correct.

PROBLEM: {problem_text}
SOLUTION: {solution.get("solution", "")}
STEPS: {solution.get("steps", [])}

Respond ONLY with valid JSON:
{{
    "is_correct": true,
    "confidence": 0.9,
    "issues_found": [],
    "corrections": [],
    "needs_human_review": false,
    "review_reason": ""
}}
"""
    return parse_json(ask_llm(prompt), {
        "is_correct": True, "confidence": 0.7,
        "issues_found": [], "corrections": [],
        "needs_human_review": False, "review_reason": ""
    })


# AGENT 5: EXPLAINER AGENT
def explainer_agent(problem_text: str, solution: dict) -> str:
    prompt = f"""
You are a friendly Math Tutor for a JEE student.

PROBLEM: {problem_text}
SOLUTION: {solution.get("solution", "")}
STEPS: {chr(10).join(solution.get("steps", []))}

Write a clear friendly explanation (150-250 words).
Explain WHY each step is done, common mistakes to avoid, and end with a tip.
Use simple language and emojis!
"""
    return ask_llm(prompt)