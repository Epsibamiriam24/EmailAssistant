from __future__ import annotations

from utils import ollama_generate


def summarize(email: str, reply: str) -> str:
    prompt = f"""Summarize this customer conversation in approximately 50 words.
Mention the customer's issue, the identified response, and the next step. Return only the summary.

Customer email: {email}
Support reply: {reply}
"""
    return ollama_generate(prompt)


def fallback_summary(email: str, reply: str) -> str:
    return f"The customer reported: {email} The support response was: {reply}"
