from __future__ import annotations

from utils import ollama_generate


TONES = ("Friendly", "Formal", "Empathetic")


def improve_tone(reply: str, tone: str) -> str:
    if tone not in TONES:
        raise ValueError(f"Tone must be one of: {', '.join(TONES)}")
    prompt = f"""Rewrite this support reply in a {tone.lower()} tone.
Keep its meaning, facts, and requested actions unchanged. Return only the rewritten reply.

Reply:
{reply}
"""
    return ollama_generate(prompt)


def fallback_tone(reply: str, tone: str) -> str:
    if tone == "Friendly":
        return reply.replace("Dear Customer,", "Hello,")
    if tone == "Empathetic":
        return "We understand how frustrating this situation must be.\n\n" + reply
    return reply
