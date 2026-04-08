"""Build prompts for the LLM-powered spatial query pipeline.

Constructs system and user prompts that include schema context, spatial
function references, safety constraints, and few-shot examples selected
by keyword relevance.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from shared.utils.logging import get_logger

logger = get_logger("p2_prompt")


def build_system_prompt(schema_meta: dict[str, Any], config: dict[str, Any]) -> str:
    """Create the system prompt with schema context and safety rules.

    Parameters
    ----------
    schema_meta : dict
        Schema metadata dict (``layers`` key with table definitions).
    config : dict
        Project configuration dict.

    Returns
    -------
    str
        Formatted system prompt for the LLM.
    """
    safety = config.get("safety", {})
    allowed = safety.get("allowed_operations", [])
    blocked = safety.get("blocked_operations", [])

    # Build schema description
    schema_lines = ["## Database Schema\n"]
    for layer_name, layer_info in schema_meta.get("layers", {}).items():
        desc = layer_info.get("description", "")
        schema_lines.append(f"### {layer_name}")
        schema_lines.append(f"Description: {desc}")
        schema_lines.append("Columns:")
        for col_name, col_info in layer_info.get("columns", {}).items():
            col_type = col_info.get("type", "unknown")
            col_desc = col_info.get("description", "")
            srid = col_info.get("srid")
            if srid:
                schema_lines.append(
                    f"  - {col_name} ({col_type}, SRID:{srid}): {col_desc}"
                )
            else:
                schema_lines.append(f"  - {col_name} ({col_type}): {col_desc}")
        schema_lines.append("")

    schema_text = "\n".join(schema_lines)

    prompt = f"""You are a spatial SQL assistant for a Sierra Nevada forestry GeoPackage database.
Your role is to translate natural language questions into valid SpatiaLite SQL queries.

{schema_text}
## Allowed Spatial Functions
{', '.join(allowed)}

## Safety Constraints
- Generate ONLY SELECT queries.
- NEVER use: {', '.join(blocked)}
- Return exactly one SQL statement with no trailing semicolons.
- All distances are in meters (the CRS is EPSG:3310, California Albers).
- Convert feet to meters when the user specifies distances in feet (1 foot = 0.3048 m).

## Output Format
Return only the SQL query, with no explanation or markdown formatting."""

    return prompt


def build_user_prompt(
    user_query: str,
    few_shots: list[dict[str, str]],
    config: dict[str, Any],
) -> str:
    """Assemble the user prompt with few-shot examples and the query.

    Parameters
    ----------
    user_query : str
        Natural language question from the user.
    few_shots : list[dict]
        Selected few-shot examples, each with ``question`` and ``sql`` keys.
    config : dict
        Project configuration dict.

    Returns
    -------
    str
        Formatted user prompt.
    """
    parts: list[str] = []

    if few_shots:
        parts.append("Here are some example queries for reference:\n")
        for i, example in enumerate(few_shots, 1):
            parts.append(f"Example {i}:")
            parts.append(f"  Question: {example['question']}")
            parts.append(f"  SQL: {example['sql']}")
            parts.append("")

    parts.append(f"Now translate this question to SQL:\n{user_query}")

    return "\n".join(parts)


def select_few_shots(
    user_query: str,
    all_examples: list[dict[str, str]],
    top_k: int = 5,
) -> list[dict[str, str]]:
    """Select the most relevant few-shot examples by keyword overlap.

    Scores each example by the number of overlapping keywords (words
    longer than 2 characters) between the user query and the example
    question, then returns the top *top_k* by score.

    Parameters
    ----------
    user_query : str
        Natural language question from the user.
    all_examples : list[dict]
        All available few-shot examples with ``question`` and ``sql`` keys.
    top_k : int
        Number of examples to return.

    Returns
    -------
    list[dict]
        The *top_k* most relevant examples, ordered by descending score.
    """
    if not all_examples:
        return []

    # Tokenize query into lowercase keywords (length > 2)
    query_tokens = set(
        w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 2
    )

    scored: list[tuple[int, int, dict[str, str]]] = []
    for idx, example in enumerate(all_examples):
        example_tokens = set(
            w.lower() for w in re.findall(r"\w+", example["question"]) if len(w) > 2
        )
        overlap = len(query_tokens & example_tokens)
        scored.append((overlap, -idx, example))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    selected = [item[2] for item in scored[:top_k]]
    logger.debug("Selected %d few-shot examples for query: %s", len(selected), user_query)
    return selected


def load_few_shot_examples(config: dict[str, Any]) -> list[dict[str, str]]:
    """Load few-shot examples from the YAML file specified in config.

    Parameters
    ----------
    config : dict
        Project configuration dict.  Expected key:
        ``config["rag"]["few_shot_examples"]``.

    Returns
    -------
    list[dict]
        List of example dicts with ``question`` and ``sql`` keys.
    """
    from pathlib import Path

    examples_path = Path(config["rag"]["few_shot_examples"])
    if not examples_path.exists():
        logger.warning("Few-shot examples file not found: %s", examples_path)
        return []

    with open(examples_path) as f:
        data = yaml.safe_load(f)

    return data.get("examples", [])
