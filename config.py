import os
import time
import json
import random
import requests
from typing import Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()

# --- 1. Gemini Configuration (For bpa_to_markdown.py & data_sets_maker.py) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

FALLBACK_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

# --- 2. Ollama Configuration (For RAG.py) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def generate_with_gemini_retry(
    contents: Any,
    config: Optional[Any] = None,
    max_retries_per_model: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Used by bpa_to_markdown.py and data_sets_maker.py with 503/429 backoff."""
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    last_error = None
    for model_name in FALLBACK_GEMINI_MODELS:
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except APIError as e:
                last_error = e
                if e.code in (429, 500, 503, 504):
                    sleep_time = (base_delay ** attempt) + random.uniform(0.2, 1.0)
                    print(f"[Gemini {e.code}] {model_name} busy (Attempt {attempt}). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    raise e
        print(f"[Gemini Fallback] Switching from {model_name} to next model...")

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def get_installed_ollama_models() -> list[str]:
    """Queries Ollama /api/tags to list all locally pulled models."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def generate_with_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.1,
    num_ctx: int = 4096,
) -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    for attempt in range(1, 4):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 404:
                installed = get_installed_ollama_models()
                if not installed:
                    raise RuntimeError(
                        f"Ollama returned 404 because no models are installed locally. "
                        f"Run 'ollama pull {model}' in your terminal first."
                    )
                # Auto-fallback to the first available local model (skipping embedding-only models)
                chat_models = [m for m in installed if "embed" not in m.lower()] or installed
                fallback_model = chat_models[0]
                print(
                    f"[Ollama 404] Model '{model}' not found locally. "
                    f"Installed models: {installed}. Auto-switching to '{fallback_model}'..."
                )
                payload["model"] = fallback_model
                resp = requests.post(url, json=payload, timeout=120)

            if resp.status_code >= 500 and attempt < 3:
                print(f"[Ollama {resp.status_code}] Transient error on attempt {attempt}. Retrying in 2s...")
                time.sleep(2.0)
                continue

            resp.raise_for_status()
            return resp.json().get("response", "").strip()

        except requests.exceptions.ConnectionError:
            if attempt < 3:
                time.sleep(2.0)
                continue
            raise RuntimeError(
                f"Cannot connect to Ollama at {base}. "
                "Make sure Ollama is running ('ollama serve')."
            )


def stream_with_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.1,
    num_ctx: int = 4096,
):
    """Yields streaming tokens from Ollama /api/generate."""
    base = OLLAMA_BASE_URL.rstrip("/")
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        if resp.status_code == 404:
            installed = get_installed_ollama_models()
            chat_models = [m for m in installed if "embed" not in m.lower()] or installed
            if chat_models:
                payload["model"] = chat_models[0]
                resp = requests.post(url, json=payload, stream=True, timeout=120)

        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    except Exception as e:
        yield f"\n[Error streaming from Ollama: {e}]"
