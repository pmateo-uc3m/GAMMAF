"""Loading and validation helpers for the MS MARCO (v2.1) dataset.

Inspects the real Hugging Face schema of ``microsoft/ms_marco`` (v2.1) rather
than assuming its serialization.  The actual schema for v2.1 is::

    {
        "query": str,
        "query_id": int,
        "answers": [str, ...],
        "query_type": str,
        "wellFormedAnswers": [str, ...],
        "passages": {
            "passage_text": [str, ...],
            "is_selected": [int, ...],   # 0/1
            "url": [str, ...],
        }
    }

This module only reads the dataset; it does not depend on any file outside
``MA/Task_generation``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DatasetEntryError(ValueError):
    """Raised when a dataset entry is missing/contains malformed fields."""


def load_msmarco(name: str, config: str, split: str):
    """Load the requested MS MARCO subset and return an iterable of rows.

    Returns the HF ``Dataset`` object (which already supports iteration and
    indexing).  Loading errors are propagated to the caller.
    """
    from datasets import load_dataset

    return load_dataset(name, config, split=split)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def extract_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate the benchmark-relevant fields of a dataset row.

    Returns a normalized dict or raises :class:`DatasetEntryError`.
    """
    query = row.get("query")
    query_id = row.get("query_id")
    answers = row.get("answers")
    passages = row.get("passages")

    if query is None or not isinstance(query, str) or not query.strip():
        raise DatasetEntryError("missing or empty 'query' field")
    if query_id is None:
        raise DatasetEntryError("missing 'query_id' field")

    answer_list = _as_str_list(answers)
    if not answer_list:
        raise DatasetEntryError("missing/empty 'answers' field")

    # Normalize where a row stores query_id as int/str variants.
    query_id_value = query_id
    if isinstance(query_id, (int, float, str)):
        pass
    else:
        query_id_value = str(query_id)

    if not isinstance(passages, dict):
        raise DatasetEntryError("'passages' is not a dict")

    passage_text = passages.get("passage_text")
    is_selected = passages.get("is_selected")

    if not isinstance(passage_text, (list, tuple)) or len(passage_text) == 0:
        raise DatasetEntryError("'passages.passage_text' missing or empty")

    texts = [str(t) for t in passage_text]

    if is_selected is None:
        raise DatasetEntryError("'passages.is_selected' missing")
    if not isinstance(is_selected, (list, tuple)):
        raise DatasetEntryError("'passages.is_selected' is not a list")
    if len(is_selected) != len(texts):
        raise DatasetEntryError(
            "length mismatch between 'passage_text' "
            f"({len(texts)}) and 'is_selected' ({len(is_selected)})"
        )

    try:
        selected = [int(s) for s in is_selected]
    except (TypeError, ValueError) as exc:
        raise DatasetEntryError("'is_selected' contains non-integer values") from exc

    for s in selected:
        if s not in (0, 1):
            raise DatasetEntryError(f"'is_selected' has invalid value {s!r} (expected 0/1)")

    return {
        "query_id": query_id_value,
        "query": query,
        "answers": answer_list,
        "passages": texts,
        "is_selected": selected,
    }
