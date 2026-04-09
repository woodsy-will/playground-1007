"""Exhaustive tests for the SQL validator -- the critical safety component.

Covers valid queries, blocked operations, injection attempts,
case-insensitive detection, spatial function whitelist enforcement,
and edge cases.
"""

from __future__ import annotations

import pytest

from projects.p2_llm_spatial_query.src.sql_validator import sanitize_sql, validate_sql


@pytest.fixture()
def safety_config() -> dict:
    """Standard safety configuration for all validator tests."""
    return {
        "safety": {
            "allowed_operations": [
                "SELECT",
                "ST_Buffer",
                "ST_Intersects",
                "ST_Within",
                "ST_Contains",
                "ST_Area",
                "ST_Length",
            ],
            "blocked_operations": [
                "DELETE",
                "DROP",
                "UPDATE",
                "INSERT",
                "ALTER",
                "TRUNCATE",
            ],
        },
    }


# -----------------------------------------------------------------------
# Valid queries that SHOULD pass
# -----------------------------------------------------------------------

class TestValidQueries:
    """Queries that must be accepted by the validator."""

    def test_simple_select(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_where(self, safety_config: dict) -> None:
        sql = "SELECT unit_name FROM harvest_units WHERE acres > 40"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_join(self, safety_config: dict) -> None:
        sql = (
            "SELECT h.unit_name, s.stream_class "
            "FROM harvest_units h, streams s "
            "WHERE ST_Intersects(h.geometry, s.geometry)"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_st_buffer(self, safety_config: dict) -> None:
        sql = (
            "SELECT h.* FROM harvest_units h, streams s "
            "WHERE ST_Intersects(h.geometry, ST_Buffer(s.geometry, 60.96))"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_st_area(self, safety_config: dict) -> None:
        sql = "SELECT unit_name, ST_Area(geometry) AS area FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_st_length(self, safety_config: dict) -> None:
        sql = (
            "SELECT stream_class, SUM(ST_Length(geometry)) AS total "
            "FROM streams GROUP BY stream_class"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_st_within(self, safety_config: dict) -> None:
        sql = (
            "SELECT r.* FROM roads r, harvest_units h "
            "WHERE ST_Within(r.geometry, h.geometry)"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_st_contains(self, safety_config: dict) -> None:
        sql = (
            "SELECT h.* FROM harvest_units h, sensitive_habitats sh "
            "WHERE ST_Contains(h.geometry, sh.geometry)"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_aggregate(self, safety_config: dict) -> None:
        sql = "SELECT COUNT(*) AS cnt FROM harvest_units WHERE prescription = 'clearcut'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_subquery(self, safety_config: dict) -> None:
        sql = (
            "SELECT * FROM harvest_units "
            "WHERE unit_id IN (SELECT unit_id FROM harvest_units WHERE acres > 30)"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_case_insensitive(self, safety_config: dict) -> None:
        sql = "select * from harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_leading_whitespace(self, safety_config: dict) -> None:
        sql = "   SELECT * FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_select_with_multiple_spatial_funcs(self, safety_config: dict) -> None:
        sql = (
            "SELECT h.unit_name, ST_Area(h.geometry), ST_Length(s.geometry) "
            "FROM harvest_units h, streams s "
            "WHERE ST_Intersects(h.geometry, ST_Buffer(s.geometry, 100))"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason


# -----------------------------------------------------------------------
# Blocked operations that MUST be caught
# -----------------------------------------------------------------------

class TestBlockedOperations:
    """Every blocked operation must be detected and rejected."""

    def test_delete(self, safety_config: dict) -> None:
        sql = "DELETE FROM harvest_units WHERE unit_id = 1"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "DELETE" in reason or "SELECT" in reason

    def test_drop_table(self, safety_config: dict) -> None:
        sql = "DROP TABLE harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_update(self, safety_config: dict) -> None:
        sql = "UPDATE harvest_units SET acres = 0 WHERE unit_id = 1"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_insert(self, safety_config: dict) -> None:
        sql = "INSERT INTO harvest_units (unit_id) VALUES (99)"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_alter_table(self, safety_config: dict) -> None:
        sql = "ALTER TABLE harvest_units ADD COLUMN evil TEXT"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_truncate(self, safety_config: dict) -> None:
        sql = "TRUNCATE TABLE harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_blocked_ops_case_insensitive(self, safety_config: dict) -> None:
        """Blocked keywords must be caught regardless of case."""
        for op in ["delete", "Delete", "DELETE", "DeLeTe"]:
            sql = f"{op} FROM harvest_units"
            is_valid, _ = validate_sql(sql, safety_config)
            assert not is_valid, f"Failed to block: {sql}"

    def test_drop_case_insensitive(self, safety_config: dict) -> None:
        sql = "drop table harvest_units"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_update_case_insensitive(self, safety_config: dict) -> None:
        sql = "update harvest_units set acres = 0"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid


# -----------------------------------------------------------------------
# SQL injection attempts that MUST be caught
# -----------------------------------------------------------------------

class TestInjectionPrevention:
    """Multi-statement injection and related attacks."""

    def test_semicolon_followed_by_drop(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units; DROP TABLE harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_semicolon_followed_by_delete(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units; DELETE FROM streams"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_semicolon_followed_by_insert(self, safety_config: dict) -> None:
        sql = "SELECT 1; INSERT INTO harvest_units VALUES (99, 'evil', 0, 'x')"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_trailing_semicolon_rejected(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units;"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_multiple_semicolons(self, safety_config: dict) -> None:
        sql = "SELECT 1; SELECT 2; SELECT 3"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_comment_hiding_injection(self, safety_config: dict) -> None:
        """Blocked op inside a comment should be stripped, but the multi-statement
        semicolon should still be caught."""
        sql = "SELECT * FROM harvest_units; -- DROP TABLE harvest_units"
        is_valid, _ = validate_sql(sql, safety_config)
        # After comment stripping, this becomes:
        # "SELECT * FROM harvest_units;"
        assert not is_valid

    def test_union_based_injection_with_delete(self, safety_config: dict) -> None:
        sql = (
            "SELECT * FROM harvest_units "
            "UNION SELECT * FROM harvest_units; DELETE FROM streams"
        )
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_empty_sql(self, safety_config: dict) -> None:
        sql = ""
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "Empty" in reason

    def test_whitespace_only(self, safety_config: dict) -> None:
        sql = "   \t\n  "
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_select_into_blocked(self, safety_config: dict) -> None:
        """SELECT that starts as valid but contains INSERT via subquery."""
        sql = "SELECT * FROM harvest_units WHERE 1=1; INSERT INTO streams VALUES (99)"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid


# -----------------------------------------------------------------------
# Spatial function whitelist
# -----------------------------------------------------------------------

class TestSpatialFunctionWhitelist:
    """Only allowed spatial functions should pass."""

    def test_allowed_st_buffer(self, safety_config: dict) -> None:
        sql = "SELECT ST_Buffer(geometry, 100) FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_allowed_st_intersects(self, safety_config: dict) -> None:
        sql = (
            "SELECT * FROM harvest_units h, streams s "
            "WHERE ST_Intersects(h.geometry, s.geometry)"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_blocked_st_union(self, safety_config: dict) -> None:
        sql = "SELECT ST_Union(geometry) FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "ST_Union" in reason

    def test_blocked_st_difference(self, safety_config: dict) -> None:
        sql = "SELECT ST_Difference(h.geometry, s.geometry) FROM harvest_units h, streams s"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "ST_Difference" in reason

    def test_blocked_st_transform(self, safety_config: dict) -> None:
        sql = "SELECT ST_Transform(geometry, 4326) FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "ST_Transform" in reason

    def test_spatial_func_case_insensitive(self, safety_config: dict) -> None:
        sql = "SELECT st_buffer(geometry, 100) FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_non_spatial_functions_ok(self, safety_config: dict) -> None:
        """Built-in SQL functions (COUNT, SUM, etc.) are not spatial
        and should not be blocked."""
        sql = "SELECT COUNT(*), SUM(acres), AVG(acres) FROM harvest_units"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason


# -----------------------------------------------------------------------
# Sanitize helper
# -----------------------------------------------------------------------

class TestSanitizeSql:
    """Tests for the sanitize_sql helper."""

    def test_strip_single_line_comment(self) -> None:
        sql = "SELECT * FROM t -- this is a comment\nWHERE x = 1"
        result = sanitize_sql(sql)
        assert "--" not in result
        assert "WHERE x = 1" in result

    def test_strip_multiline_comment(self) -> None:
        sql = "SELECT /* block */ * FROM t"
        result = sanitize_sql(sql)
        assert "/*" not in result
        assert "*/" not in result

    def test_normalize_whitespace(self) -> None:
        sql = "SELECT  *   FROM   harvest_units"
        result = sanitize_sql(sql)
        assert result == "SELECT * FROM harvest_units"

    def test_strip_leading_trailing(self) -> None:
        sql = "  \n  SELECT * FROM t  \n  "
        result = sanitize_sql(sql)
        assert result == "SELECT * FROM t"


# -----------------------------------------------------------------------
# SQLite-specific dangerous keywords (always blocked)
# -----------------------------------------------------------------------

class TestSQLiteDangerousKeywords:
    """PRAGMA, ATTACH, DETACH, VACUUM, and load_extension must always
    be blocked, regardless of the user-configurable blocklist."""

    def test_pragma_blocked(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units; PRAGMA table_info(harvest_units)"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_pragma_standalone_blocked(self, safety_config: dict) -> None:
        sql = "PRAGMA table_info(harvest_units)"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid

    def test_attach_blocked(self, safety_config: dict) -> None:
        sql = "ATTACH DATABASE ':memory:' AS evil"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_detach_blocked(self, safety_config: dict) -> None:
        sql = "DETACH DATABASE evil"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_vacuum_blocked(self, safety_config: dict) -> None:
        sql = "VACUUM"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_load_extension_in_select_blocked(self, safety_config: dict) -> None:
        """load_extension() called inside a SELECT must be caught."""
        sql = "SELECT load_extension('evil.so')"
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "LOAD_EXTENSION" in reason

    def test_load_extension_case_insensitive(self, safety_config: dict) -> None:
        sql = "SELECT Load_Extension('mod_spatialite') FROM harvest_units"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid

    def test_pragma_case_insensitive(self, safety_config: dict) -> None:
        sql = "pragma table_info(harvest_units)"
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid


# -----------------------------------------------------------------------
# String literal edge cases
# -----------------------------------------------------------------------

class TestStringLiteralEdgeCases:
    """Ensure the validator handles tricky string literals correctly."""

    def test_escaped_quotes_pass(self, safety_config: dict) -> None:
        """SQL escaped quotes (O''Brien) should not break validation."""
        sql = "SELECT * FROM harvest_units WHERE unit_name = 'O''Brien'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_semicolon_inside_string_literal(self, safety_config: dict) -> None:
        """Semicolons inside string literals should not trigger injection
        detection."""
        sql = "SELECT * FROM harvest_units WHERE unit_name = 'a;b'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_delete_keyword_in_string_literal(self, safety_config: dict) -> None:
        """Blocked keywords inside string literals must not cause false
        positives."""
        sql = "SELECT * FROM harvest_units WHERE notes = 'DELETE old records'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_drop_keyword_in_string_literal(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units WHERE notes = 'will DROP by later'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_update_keyword_in_string_literal(self, safety_config: dict) -> None:
        sql = "SELECT * FROM harvest_units WHERE notes = 'UPDATE pending'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_pragma_keyword_in_string_literal(self, safety_config: dict) -> None:
        """Always-blocked keywords inside string literals must also be
        ignored."""
        sql = "SELECT * FROM harvest_units WHERE notes = 'see PRAGMA docs'"
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason


# -----------------------------------------------------------------------
# Nested subquery validation
# -----------------------------------------------------------------------

class TestNestedSubqueries:
    """Subqueries must be validated recursively."""

    def test_nested_select_valid(self, safety_config: dict) -> None:
        sql = (
            "SELECT * FROM (SELECT unit_name, ST_Area(geometry) "
            "FROM harvest_units) sub"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_deeply_nested_subquery_valid(self, safety_config: dict) -> None:
        sql = (
            "SELECT * FROM ("
            "  SELECT * FROM ("
            "    SELECT unit_name FROM harvest_units WHERE acres > 10"
            "  ) inner_q"
            ") outer_q"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert is_valid, reason

    def test_nested_blocked_spatial_func(self, safety_config: dict) -> None:
        """A blocked spatial function inside a subquery must be caught."""
        sql = (
            "SELECT * FROM ("
            "  SELECT ST_Union(geometry) FROM harvest_units"
            ") sub"
        )
        is_valid, reason = validate_sql(sql, safety_config)
        assert not is_valid
        assert "ST_Union" in reason

    def test_nested_subquery_with_blocked_keyword(self, safety_config: dict) -> None:
        """A blocked keyword inside a subquery must be caught."""
        sql = (
            "SELECT * FROM harvest_units "
            "WHERE unit_id IN ("
            "  SELECT unit_id FROM harvest_units; DELETE FROM streams"
            ")"
        )
        is_valid, _ = validate_sql(sql, safety_config)
        assert not is_valid
