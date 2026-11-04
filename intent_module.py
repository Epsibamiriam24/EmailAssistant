from __future__ import annotations

from utils import ollama_generate


def detect_intent(email: str, categories: list[str]) -> str:
    category_text = ", ".join(categories)
    prompt = f"""Classify this customer email into exactly one category.
Return only the category name, with no explanation.
Allowed categories: {category_text}

Customer email:
{email}
"""
    result = ollama_generate(prompt)
    normalized = result.casefold()
    for category in categories:
        if category.casefold() in normalized:
            return category
    raise ValueError(f"Ollama returned an unknown intent: {result}")


def fallback_intent(email: str, categories: list[str]) -> str:
    text = email.casefold()
    keywords = {
        "Delivery Issue": ("delivery", "shipment", "package", "order", "arrived", "tracking"),
        "Refund Request": ("refund", "money back", "reimburse"),
        "Product Complaint": ("damaged", "broken", "defective", "stopped working"),
        "Technical Support": ("login", "log in", "password", "application", "app", "error"),
        "Order Cancellation": ("cancel", "cancellation"),
        "Exchange Request": ("exchange", "wrong size", "wrong color"),
    }
    for category, terms in keywords.items():
        if category in categories and any(term in text for term in terms):
            return category
    return "General Inquiry" if "General Inquiry" in categories else categories[0]
