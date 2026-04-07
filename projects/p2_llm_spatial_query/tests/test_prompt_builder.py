"""Tests for prompt construction and few-shot selection."""

from __future__ import annotations

import pytest

from projects.p2_llm_spatial_query.src.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    select_few_shots,
)


class TestBuildSystemPrompt:
    """Verify system prompt includes schema and safety context."""

    def test_includes_table_names(
        self, sample_schema_meta: dict, default_config: dict
    ) -> None:
        prompt = build_system_prompt(sample_schema_meta, default_config)
        assert "harvest_units" in prompt
        assert "streams" in prompt

    def test_includes_column_descriptions(
        self, sample_schema_meta: dict, default_config: dict
    ) -> None:
        prompt = build_system_prompt(sample_schema_meta, default_config)
        assert "unit_name" in prompt
        assert "stream_class" in prompt

    def test_includes_blocked_ops(
        self, sample_schema_meta: dict, default_config: dict
    ) -> None:
        prompt = build_system_prompt(sample_schema_meta, default_config)
        assert "DELETE" in prompt
        assert "DROP" in prompt

    def test_includes_allowed_functions(
        self, sample_schema_meta: dict, default_config: dict
    ) -> None:
        prompt = build_system_prompt(sample_schema_meta, default_config)
        assert "ST_Buffer" in prompt
        assert "ST_Intersects" in prompt

    def test_includes_select_only_instruction(
        self, sample_schema_meta: dict, default_config: dict
    ) -> None:
        prompt = build_system_prompt(sample_schema_meta, default_config)
        assert "SELECT" in prompt


class TestBuildUserPrompt:
    """Verify user prompt assembly with few-shot examples."""

    def test_includes_user_query(self, default_config: dict) -> None:
        prompt = build_user_prompt("Show all harvest units", [], default_config)
        assert "Show all harvest units" in prompt

    def test_includes_few_shot_examples(self, default_config: dict) -> None:
        shots = [
            {"question": "List streams", "sql": "SELECT * FROM streams"},
        ]
        prompt = build_user_prompt("Show units", shots, default_config)
        assert "List streams" in prompt
        assert "SELECT * FROM streams" in prompt

    def test_empty_few_shots(self, default_config: dict) -> None:
        prompt = build_user_prompt("Show units", [], default_config)
        assert "Show units" in prompt
        assert "Example" not in prompt


class TestSelectFewShots:
    """Keyword-based few-shot example selection."""

    @pytest.fixture()
    def examples(self) -> list[dict[str, str]]:
        return [
            {"question": "Show harvest units near streams", "sql": "SELECT ..."},
            {"question": "Calculate road length on federal land", "sql": "SELECT ..."},
            {"question": "Find sensitive habitats near harvest units", "sql": "SELECT ..."},
            {"question": "List all streams by class", "sql": "SELECT ..."},
            {"question": "Buffer roads by 100 meters", "sql": "SELECT ..."},
            {"question": "Total acreage of clearcut units", "sql": "SELECT ..."},
        ]

    def test_returns_top_k(self, examples: list[dict]) -> None:
        result = select_few_shots("harvest units near streams", examples, top_k=3)
        assert len(result) == 3

    def test_relevant_examples_ranked_first(self, examples: list[dict]) -> None:
        result = select_few_shots("harvest units near streams", examples, top_k=2)
        questions = [r["question"] for r in result]
        # The example about harvest units + streams should rank highest
        assert any("harvest" in q and "streams" in q for q in questions)

    def test_empty_examples_returns_empty(self) -> None:
        result = select_few_shots("anything", [], top_k=5)
        assert result == []

    def test_top_k_larger_than_available(self, examples: list[dict]) -> None:
        result = select_few_shots("show everything", examples, top_k=100)
        assert len(result) == len(examples)
