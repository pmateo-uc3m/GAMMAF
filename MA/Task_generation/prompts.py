"""Prompt template loading for the contamination LLM.

Prompts are stored as plain-text files in this directory so they are easy to
edit without touching code (see ``prompts_system.txt``, ``prompts_user.txt``,
``prompts_answer_system.txt`` and ``prompts_answer_user.txt``).  The
contamination user prompt supports ``str.format`` placeholders: ``{query}``,
``{correct_answer}``, ``{n}``, ``{source_passages}``.  The answer-generation
user prompt supports ``{query}`` and ``{source_passages}``.
"""

from __future__ import annotations

import os
from typing import Dict, List

_DIR = os.path.dirname(os.path.abspath(__file__))


def _read(name: str) -> str:
    path = os.path.join(_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompts(system_file: str, user_file: str) -> Dict[str, str]:
    """Load the system and user prompt templates."""
    return {
        "system": _read(system_file),
        "user": _read(user_file),
    }


def load_answer_prompts(system_file: str, user_file: str) -> Dict[str, str]:
    """Load the system and user answer-generation prompt templates."""
    return {
        "system": _read(system_file),
        "user": _read(user_file),
    }


def build_answer_prompt(
    template: str,
    query: str,
    source_passages: List[str],
) -> str:
    """Render the answer-generation user prompt with the given values."""
    numbered = "\n".join(
        f"[{i}] {p}" for i, p in enumerate(source_passages)
    )
    return template.format(query=query, source_passages=numbered)


def build_user_prompt(
    template: str,
    query: str,
    correct_answer: str,
    source_passages: List[str],
    n: int,
) -> str:
    """Render the user prompt with the given values."""
    numbered = "\n".join(
        f"[{i}] {p}" for i, p in enumerate(source_passages)
    )
    return template.format(
        query=query,
        correct_answer=correct_answer,
        n=n,
        source_passages=numbered,
    )
