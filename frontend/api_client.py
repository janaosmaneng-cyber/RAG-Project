"""Thin wrapper around the backend API. Keeps all HTTP details (URL,
timeouts, multipart encoding) out of app.py so the UI code stays simple."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Generous timeout: local Ollama generation can take a while, especially
# on CPU-only setups or with a larger num_ctx.
REQUEST_TIMEOUT_SECONDS = 120


class APIClientError(Exception):
    """Raised whenever the backend can't be reached or returns an error.
    Caught in app.py to show a friendly message instead of a raw traceback."""


def check_health() -> bool:
    """Returns True if the backend is reachable and healthy."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def ask_question(question: str, image_bytes: bytes | None = None, image_filename: str | None = None) -> dict:
    """Send a question (and optional image) to POST /query.

    Returns the parsed JSON response:
        {"answer": str, "sources": list[str], "image_classification": dict | None}

    Raises APIClientError with a human-readable message on any failure —
    connection refused, timeout, non-200 status, or malformed response.
    """

    if not question or not question.strip():
        raise APIClientError("Please enter a question before sending.")

    data = {"question": question}
    files = None

    if image_bytes is not None:
        files = {
            "image": (
                image_filename or "upload.jpg",
                image_bytes,
            )
        }

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError as exc:
        raise APIClientError(
            f"Could not connect to the backend at {API_BASE_URL}. "
            "Is it running? (uvicorn app.main:app --reload)"
        ) from exc
    except requests.Timeout as exc:
        raise APIClientError(
            "The backend took too long to respond. The model may still be "
            "generating — try again in a moment."
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError(f"Request to the backend failed: {exc}") from exc

    if response.status_code == 422:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIClientError(f"The backend rejected the request: {detail}")

    if response.status_code >= 500:
        raise APIClientError(
            f"The backend hit an internal error (status {response.status_code}). "
            "Check the backend logs for details."
        )

    if response.status_code != 200:
        raise APIClientError(
            f"Unexpected response from backend (status {response.status_code})."
        )

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("The backend returned a response that wasn't valid JSON.") from exc