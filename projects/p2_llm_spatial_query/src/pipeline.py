"""End-to-end LLM spatial query pipeline.

Orchestrates schema loading, prompt construction, SQL generation,
validation, execution, and result formatting into a single
``run_query`` entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.utils.config import load_config
from shared.utils.logging import get_logger

logger = get_logger("p2_pipeline")


def run_query(
    user_query: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """Execute the full natural-language-to-spatial-SQL pipeline.

    Steps:
    1. Load configuration and schema metadata.
    2. Select relevant few-shot examples.
    3. Build system and user prompts.
    4. Generate SQL via the local LLM.
    5. Validate the generated SQL.
    6. Execute the validated SQL against the GeoPackage.
    7. Format and return results.

    Parameters
    ----------
    user_query : str
        Natural language spatial question.
    config_path : str or Path
        Path to the project YAML configuration file.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``sql`` (str): The generated SQL statement.
        - ``is_valid`` (bool): Whether the SQL passed validation.
        - ``validation_reason`` (str): Reason if validation failed.
        - ``results_summary`` (str): Human-readable results or error message.
        - ``raw_results`` (DataFrame | GeoDataFrame | None): Query results.
    """
    from projects.p2_llm_spatial_query.src.executor import execute_query
    from projects.p2_llm_spatial_query.src.formatter import (
        format_error,
        format_results,
    )
    from projects.p2_llm_spatial_query.src.prompt_builder import (
        build_system_prompt,
        build_user_prompt,
        load_few_shot_examples,
        select_few_shots,
    )
    from projects.p2_llm_spatial_query.src.schema_extractor import (
        load_schema_metadata,
    )
    from projects.p2_llm_spatial_query.src.sql_generator import generate_sql
    from projects.p2_llm_spatial_query.src.sql_validator import validate_sql

    config = load_config(config_path)
    logger.info("Loaded config from %s", config_path)

    result: dict[str, Any] = {
        "sql": "",
        "is_valid": False,
        "validation_reason": "",
        "results_summary": "",
        "raw_results": None,
    }

    # 1. Load schema metadata
    try:
        schema_meta = load_schema_metadata(config)
    except FileNotFoundError as exc:
        result["results_summary"] = format_error(str(exc), user_query)
        return result

    # 2. Load and select few-shot examples
    all_examples = load_few_shot_examples(config)
    top_k = config.get("rag", {}).get("top_k", 5)
    few_shots = select_few_shots(user_query, all_examples, top_k=top_k)

    # 3. Build prompts
    system_prompt = build_system_prompt(schema_meta, config)
    user_prompt = build_user_prompt(user_query, few_shots, config)

    # 4. Generate SQL
    try:
        sql = generate_sql(system_prompt, user_prompt, config)
    except (ConnectionError, RuntimeError) as exc:
        result["results_summary"] = format_error(str(exc), user_query)
        return result

    result["sql"] = sql

    # 5. Validate SQL
    is_valid, reason = validate_sql(sql, config)
    result["is_valid"] = is_valid
    result["validation_reason"] = reason

    if not is_valid:
        result["results_summary"] = format_error(
            f"Generated SQL failed validation: {reason}", user_query
        )
        logger.warning("SQL validation failed: %s", reason)
        return result

    # 6. Execute query
    gpkg_path = Path(config["geopackage"]["path"])
    try:
        raw_results = execute_query(sql, gpkg_path, config)
    except Exception as exc:
        result["results_summary"] = format_error(str(exc), user_query)
        logger.error("Query execution failed: %s", exc)
        return result

    result["raw_results"] = raw_results

    # 7. Format results
    result["results_summary"] = format_results(raw_results, user_query)
    logger.info("Pipeline complete for query: %s", user_query)
    return result
