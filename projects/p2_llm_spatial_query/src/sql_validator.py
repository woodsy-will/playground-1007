"""SQL validation and sanitization for safe query execution.

This is the CRITICAL safety component of the pipeline.  Every generated
SQL statement must pass validation before being executed against the
GeoPackage.  The validator enforces a strict whitelist/blocklist approach:

- Only SELECT statements are allowed.
- Destructive operations (DELETE, DROP, UPDATE, INSERT, ALTER, TRUNCATE)
  are explicitly blocked.
- Only whitelisted spatial functions may appear.
- Multi-statement injection via semicolons is detected and rejected.
"""

from __future__ import annotations

import re
from typing import Any

from shared.utils.logging import get_logger

logger = get_logger("p2_validator")

# Pre-compiled patterns for performance
_COMMENT_PATTERN = re.compile(
    r"--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_sql(sql: str) -> str:
    """Strip comments and normalize whitespace in a SQL string.

    Parameters
    ----------
    sql : str
        Raw SQL string.

    Returns
    -------
    str
        Cleaned SQL with comments removed and whitespace collapsed.
    """
    # Remove single-line (--) and multi-line (/* */) comments
    cleaned = _COMMENT_PATTERN.sub(" ", sql)
    # Collapse whitespace
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def validate_sql(
    sql: str,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Validate a SQL statement against safety rules.

    Checks are applied in order of severity:

    1. Statement must begin with SELECT (case-insensitive).
    2. No blocked operations (DELETE, DROP, UPDATE, INSERT, ALTER, TRUNCATE).
    3. Only allowed spatial functions are present.
    4. No multi-statement injection (semicolons followed by more text).

    Parameters
    ----------
    sql : str
        The SQL statement to validate.
    config : dict
        Project configuration dict with ``safety.blocked_operations``
        and ``safety.allowed_operations`` keys.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` if valid, or ``(False, reason)`` if invalid.
    """
    safety = config.get("safety", {})
    blocked_ops = [op.upper() for op in safety.get("blocked_operations", [])]
    allowed_ops = [op.upper() for op in safety.get("allowed_operations", [])]

    # Sanitize first
    cleaned = sanitize_sql(sql)

    if not cleaned:
        return False, "Empty SQL statement"

    # 1. Must start with SELECT
    if not cleaned.upper().lstrip().startswith("SELECT"):
        return False, "SQL must start with SELECT"

    # 2. Check for blocked operations
    # Use word-boundary matching to avoid false positives (e.g. "UPDATED" in a
    # column alias should not trip "UPDATE", but we err on the side of caution
    # for safety — a standalone keyword is blocked).
    upper_sql = cleaned.upper()
    for op in blocked_ops:
        # Match the blocked keyword as a standalone word
        pattern = rf"\b{re.escape(op)}\b"
        if re.search(pattern, upper_sql):
            return False, f"Blocked operation detected: {op}"

    # 3. Check spatial functions — only whitelisted ones are allowed
    # Find all function-call patterns that start with ST_
    found_spatial = set(re.findall(r"\b(ST_\w+)\s*\(", cleaned, re.IGNORECASE))
    allowed_upper = {op.upper() for op in allowed_ops if op.upper().startswith("ST_")}
    for func in found_spatial:
        if func.upper() not in allowed_upper:
            return (
                False,
                f"Spatial function not in whitelist: {func}. "
                f"Allowed: {', '.join(sorted(allowed_upper))}",
            )

    # 4. Multi-statement injection detection
    # Remove content inside string literals before checking for semicolons
    # to avoid false positives on quoted semicolons.
    no_strings = re.sub(r"'[^']*'", "''", cleaned)
    if ";" in no_strings:
        # Check if there is meaningful content after the semicolon
        parts = no_strings.split(";")
        trailing = ";".join(parts[1:]).strip()
        if trailing:
            return False, "Multiple SQL statements detected (possible injection)"
        # A bare trailing semicolon is just sloppy but not dangerous;
        # we still reject it for strictness.
        return False, "Trailing semicolon not allowed"

    logger.info("SQL validation passed")
    return True, ""
