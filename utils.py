from __future__ import annotations

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


class OllamaError(RuntimeError):
    """Raised when Ollama cannot generate a response."""


def ollama_generate(prompt: str, *, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
    except requests.RequestException as exc:
        raise OllamaError(
            "Could not connect to Ollama. Start Ollama and make sure llama3.2 is installed."
        ) from exc
    except (ValueError, KeyError) as exc:
        raise OllamaError("Ollama returned an invalid response.") from exc

    if not result:
        raise OllamaError("Ollama returned an empty response.")
    return result
