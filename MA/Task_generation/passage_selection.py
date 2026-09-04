"""Deterministic passage selection for the contamination benchmark.

Given an MS MARCO entry, select **all** passages marked ``is_selected == 1``
(the passages that were judged relevant for answering the query).

Selection policy (documented here and in the README):

* Every passage with ``is_selected == 1`` is returned, in original order.
* No random sampling is involved; the result is fully deterministic from the
  entry itself.
* If the entry has no selected passage (``is_selected`` all 0), a
  :class:`ValueError` is raised and the caller skips the entry rather than
  silently producing an empty example.
"""

from __future__ import annotations

from typing import List, Tuple


def select_selected_passages(
    passages: List[str],
    is_selected: List[int],
) -> Tuple[List[str], List[int]]:
    """Return all passages with ``is_selected == 1``.

    Returns ``(selected_texts, selected_original_indices)`` preserving the
    original order.

    Raises ``ValueError`` if the lengths mismatch or no passage is selected.
    """
    if len(passages) != len(is_selected):
        raise ValueError("passages and is_selected must have equal length")

    selected_idx = [i for i, s in enumerate(is_selected) if s == 1]
    if not selected_idx:
        raise ValueError("no selected passages (is_selected all 0)")

    texts = [passages[i] for i in selected_idx]
    return texts, selected_idx