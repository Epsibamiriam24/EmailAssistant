from __future__ import annotations

from utils import ollama_generate


def generate_reply(email: str, intent: str, previous_reply: str) -> str:
    prompt = f"""Write a concise, professional customer-support reply.
Do not invent order details, guarantees, or policies. Address the customer's issue
and suggest the next practical step. Return only the reply.

Intent: {intent}
Customer email: {email}
Example previous reply: {previous_reply}
"""
    return ollama_generate(prompt)


def fallback_reply(intent: str, previous_reply: str) -> str:
    return (
        f"Dear Customer,\n\n{previous_reply}\n\n"
        f"Please contact our support team if you need further assistance regarding your {intent.lower()}.\n\n"
        "Regards,\nSupport Team"
    )
