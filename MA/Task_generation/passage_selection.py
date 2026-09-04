"""Deterministic passage selection for the contamination benchmark.

Given an MS MARCO entry, select exactly ``n`` passages while prioritizing the
passages marked ``is_selected == 1`` (the relevant passages).

Selection policy (documented here and in the README):

* Prefer passages with ``is_selected == 1``.
* Include as many selected passages as possible, up to ``n``.
* Fill remaining slots with randomly chosen non-selected passages.
* If there are >= ``n`` total passages, exactly ``n`` are returned.
* If there are fewer than ``n`` total passages, the entry is *skipped* rather
  than silently producing an incorrectly-sized example.

Randomness uses ``numpy.random.Generator`` seeded explicitly from the
configured passage-selection seed, so the selection is fully reproducible.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def select_passages(
    passages: List[str],
    is_selected: List[int],
    n: int,
    rng,
) -> Tuple[List[str], List[int]]:
    """Select exactly ``n`` passages, prioritizing selected ones.

    Returns ``(selected_texts, selected_original_indices)``.

    Raises ``ValueError`` if fewer than ``n`` passages are available.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if len(passages) != len(is_selected):
        raise ValueError("passages and is_selected must have equal length")

    if len(passages) < n:
        raise ValueError(
            f"insufficient passages: have {len(passages)}, need {n}"
        )

    selected_idx = [i for i, s in enumerate(is_selected) if s == 1]
    non_selected_idx = [i for i, s in enumerate(is_selected) if s != 1]

    chosen: List[int] = []

    if len(selected_idx) >= n:
        chosen = selected_idx[:n]
    else:
        chosen.extend(selected_idx)
        remaining = n - len(chosen)
        fill = non_selected_idx[:]
        rng.shuffle(fill)
        chosen.extend(fill[:remaining])

    if len(chosen) != n:
        raise ValueError(f"internal selection error: expected {n}, got {len(chosen)}")

    texts = [passages[i] for i in chosen]
    return texts, chosen
