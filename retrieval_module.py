from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {"Customer_Email", "Previous_Reply", "Intent"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("Dataset contains no records.")
    return data


def available_intents(data: pd.DataFrame) -> list[str]:
    return [str(value) for value in data["Intent"].drop_duplicates() if str(value).strip()]


def find_similar_email(email: str, data: pd.DataFrame) -> dict[str, object]:
    messages = data["Customer_Email"].astype(str).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(messages + [email])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    index = int(scores.argmax())
    row = data.iloc[index]
    return {
        "email": str(row["Customer_Email"]),
        "reply": str(row["Previous_Reply"]),
        "intent": str(row["Intent"]),
        "score": float(scores[index]),
    }
