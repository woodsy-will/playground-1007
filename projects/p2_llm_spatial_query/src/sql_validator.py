"""SQL validation and sanitization for safe query execution.

This is the CRITICAL safety component of the pipeline.  Every generated
SQL statement must pass validation before being executed against the
GeoPackage.  The validator enforces a strict whitelist/blocklist approach:

- Only SELECT statements are allowed.
- Destructive operations (DELETE, DROP, UPDATE, INSERT, ALTER, TRUNCATE)
  are explicitly blocked.
- Only whitelisted spatial functions may appear.
- Multi-statement injection via semicolons is detected and rejected.

Validation uses ``sqlparse`` for tokenization so that keywords appearing
inside string literals are correctly ignored (no false positives).
"""

from __future__ import annotations

import re
from typing import Any

import sqlparse
from sqlparse import tokens as token_types

from shared.utils.logging import get_logger

logger = get_logger("p2_validator")

# Pre-compiled patterns for comment/whitespace sanitization
_COMMENT_PATTERN = re.compile(
    r"--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")

# SQLite-specific dangerous keywords that are ALWAYS blocked regardless
# of configuration.  These can bypass the SELECT-only check when used as
# function calls or stand-alone statements.
_ALWAYS_BLOCKED = frozenset({
    "PRAGMA", "ATTACH", "DETACH", "VACUUM", "LOAD_EXTENSION",
})


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

    Uses ``sqlparse`` for tokenization to properly distinguish keywords
    from string literal content, eliminating false positives from blocked
    keywords that appear inside quoted text.

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
    blocked_ops = {op.upper() for op in safety.get("blocked_operations", [])}
    allowed_ops = [op.upper() for op in safety.get("allowed_operations", [])]
    allowed_spatial = {op for op in allowed_ops if op.startswith("ST_")}

    # Sanitize first
    cleaned = sanitize_sql(sql)

    if not cleaned:
        return False, "Empty SQL statement"

    # 1. Must start with SELECT
    if not cleaned.upper().lstrip().startswith("SELECT"):
        return False, "SQL must start with SELECT"

    # Parse with sqlparse — handles string literals, multi-statement, etc.
    statements = sqlparse.parse(cleaned)
    non_empty = [s for s in statements if str(s).strip()]

    # 4. Multi-statement detection (checked early for security)
    if len(non_empty) > 1:
        return False, "Multiple SQL statements detected (possible injection)"

    stmt = non_empty[0]
    tokens = list(stmt.flatten())

    for i, token in enumerate(tokens):
        if token.is_whitespace:
            continue

        # Skip string literals — the core improvement over regex.
        # sqlparse correctly tokenizes 'DELETE old records' as a single
        # Literal.String.Single token, so blocked keywords inside quoted
        # text are never inspected.
        if token.ttype is not None and token.ttype in token_types.Literal.String:
            continue

        upper_val = token.value.upper()

        # 2a. Check for always-blocked SQLite keywords (security-critical,
        #     independent of user config).
        if upper_val in _ALWAYS_BLOCKED:
            return False, f"Blocked keyword detected: {upper_val}"

        # 2b. Check for config-driven blocked operations
        if upper_val in blocked_ops:
            return False, f"Blocked operation detected: {upper_val}"

        # 3. Check spatial functions — only whitelisted ones are allowed.
        #    Only flag function calls (Name token followed by '(').
        if upper_val.startswith("ST_"):
            for j in range(i + 1, len(tokens)):
                if tokens[j].is_whitespace:
                    continue
                if tokens[j].ttype is token_types.Punctuation and tokens[j].value == "(":
                    if upper_val not in allowed_spatial:
                        return (
                            False,
                            f"Spatial function not in whitelist: {token.value}. "
                            f"Allowed: {', '.join(sorted(allowed_spatial))}",
                        )
                break

        # 4. Semicolon detection (trailing or separating)
        if token.ttype is token_types.Punctuation and token.value == ";":
            has_trailing = any(
                not tokens[j].is_whitespace
                for j in range(i + 1, len(tokens))
            )
            if has_trailing:
                return False, "Multiple SQL statements detected (possible injection)"
            return False, "Trailing semicolon not allowed"

    logger.info("SQL validation passed")
    return True, ""
