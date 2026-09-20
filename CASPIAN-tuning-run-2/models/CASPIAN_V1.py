"""CASPIAN V1 — reciprocity-aware influence score.

V1 keeps the whole B0 pipeline (LI-CTE influence, degree-aware normalization,
spectral cascade detection, attribution) and only changes the per-agent score
used for flagging and AUROC.

Motivation (paper-faithful): the paper defines a cascade as self-reinforcing
propagation.  A self-reinforcing pair necessarily exchanges influence in both
directions, i.e. the influence matrix has a reciprocal edge ``i -> j`` and
``j -> i``.  B0's origin/amplifier/bridge statistics are row/column aggregates
and a single strong one-way edge can dominate them.  V1 therefore adds an
edge-level reciprocity statistic, computed directly from the LI-CTE influence
matrix that CASPIAN already estimates:

    R_ij = min(A_ij, A_ji)          (reciprocal influence on edge {i, j})
    r_i  = max_j R_ij               (strongest reciprocal partner)
    s_i  = sum_j R_ij               (total reciprocal coupling)

The final score is B0's attribution score plus a standardised combination of
``r_i`` and ``s_i``.  No new data channel is used.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caspian_base import CASPIANDetector as _BaseDetector
from caspian_base import Master as _BaseMaster
from LoggingUtils import log_done, log_info
from Utils import load_config_from_path


class CASPIANDetector(_BaseDetector):
    def _reciprocity_features(self, raw):
        reciprocal = np.minimum(raw, raw.T)
        np.fill_diagonal(reciprocal, 0.0)
        return reciprocal.max(axis=1), reciprocal.sum(axis=1)

    def _score_agents(self, state, raw, normalised, embeddings, debate_round):
        base = self._node_scores(raw, normalised)
        n = raw.shape[0]
        if n < 2:
            return base
        weight = float(getattr(self.config, "reciprocity_weight", 0.5))
        strongest, total = self._reciprocity_features(raw)
        score = base + weight * (
            self._standardise(strongest, self.epsilon)
            + self._standardise(total, self.epsilon)
        )
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


class Master(_BaseMaster):
    def _run(self, train_pkl_path=None):
        detector = CASPIANDetector(self.args)
        log_info("Initialized CASPIAN V1 (reciprocity-aware influence score).")
        log_done("CASPIAN V1 ready.")
        return {}, detector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASPIAN V1 detector")
    parser.add_argument("--config", required=True)
    parsed = parser.parse_args()
    Master(parsed.config)._run()
