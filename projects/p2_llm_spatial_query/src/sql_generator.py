"""Generate SQL from natural language via a local LLM endpoint.

Calls an OpenAI-compatible ``/v1/chat/completions`` API (e.g. vLLM, Ollama,
text-generation-inference) and parses the SQL from the response.
"""

from __future__ import annotations

import re
from typing import Any

import requests

from shared.utils.logging import get_logger

logger = get_logger("p2_sql_gen")


def generate_sql(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> str:
    """Call the local LLM endpoint and return generated SQL.

    Parameters
    ----------
    system_prompt : str
        System-level prompt with schema and safety context.
    user_prompt : str
        User-level prompt with few-shot examples and query.
    config : dict
        Project configuration dict.  Expected keys under ``config["llm"]``:
        ``endpoint``, ``model``, ``max_tokens``, ``temperature``.

    Returns
    -------
    str
        The generated SQL string.

    Raises
    ------
    ConnectionError
        If the LLM endpoint is unreachable.
    RuntimeError
        If the LLM returns an unexpected response format.
    """
    llm_cfg = config.get("llm", {})
    endpoint = llm_cfg.get("endpoint", "http://localhost:8080/v1")
    model = llm_cfg.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
    max_tokens = llm_cfg.get("max_tokens", 512)
    temperature = llm_cfg.get("temperature", 0.1)

    url = f"{endpoint}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        logger.error("Cannot reach LLM endpoint at %s", url)
        raise ConnectionError(
            f"LLM endpoint unreachable at {url}. "
            "Ensure the local model server is running."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        logger.error("LLM endpoint returned HTTP error: %s", exc)
        raise RuntimeError(f"LLM endpoint error: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        logger.error("LLM request timed out")
        raise ConnectionError("LLM request timed out after 60s") from exc

    data = response.json()
    try:
        raw_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected LLM response format: {data}"
        ) from exc

    sql = parse_sql_from_response(raw_text)
    logger.info("Generated SQL: %s", sql)
    return sql


def parse_sql_from_response(response_text: str) -> str:
    """Extract SQL from an LLM response that may contain markdown fences.

    Handles responses wrapped in triple-backtick code blocks (with or
    without a ``sql`` language tag) as well as bare SQL text.

    Parameters
    ----------
    response_text : str
        Raw text from the LLM response.

    Returns
    -------
    str
        Cleaned SQL string.
    """
    # Try to extract from markdown code block
    pattern = r"```(?:sql)?\s*\n?(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        sql = response_text.strip()

    # Remove trailing semicolons
    sql = sql.rstrip(";").strip()

    return sql
