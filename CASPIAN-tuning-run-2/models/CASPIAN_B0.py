"""CASPIAN baseline adapted to GAMMAF (B0).

Thin wrapper around the shared base in ``caspian_base.py`` (an exact copy of
``defense-models/CASPIAN.py``).  B0 keeps the paper's LI-CTE influence,
degree-aware normalization, spectral signals, cascade rules and attribution,
and only changes the framework contract:

* flags are emitted every round (including round 1) as the top-``top_k``
  agents of the single-turn attribution score;
* the score is a standardised sum of origin/amplifier/bridge statistics so it
  is continuous and yields a well-defined AUROC.

This file exists so the baseline and the five variants can be evaluated from a
single ``models_directory`` and a single results JSON.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caspian_base import CASPIANDetector, Master  # noqa: F401  (re-exported)
