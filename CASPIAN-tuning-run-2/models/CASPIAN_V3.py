"""CASPIAN V3 — reciprocity + synchronized minority + temporal persistence.

V3 extends V2 with the paper's *persistence* requirement.  A cascade is
sustained reinforcement across turns, and CASPIAN's multi-turn rule requires a
signal to persist over an adaptive interval.  V3 applies the same principle to
the answer-cluster evidence: the synchronized-minority score is smoothed with
an EMA kept in the per-trace state,

    cluster_ema(t) = decay * cluster_ema(t-1) + (1 - decay) * cluster(t),

so an agent that repeatedly sits in a small coordinated minority accumulates
evidence even when a single turn is noisy, while a one-off coincidence fades.
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caspian_base import CASPIANDetector as _BaseDetector
from caspian_base import Master as _BaseMaster
from LoggingUtils import log_done, log_info


class CASPIANDetector(_BaseDetector):
    @staticmethod
    def _answers(debate_round):
        return [str(item.get("answer", "")).strip().upper() for item in debate_round]

    def _reciprocity_features(self, raw):
        reciprocal = np.minimum(raw, raw.T)
        np.fill_diagonal(reciprocal, 0.0)
        return reciprocal.max(axis=1), reciprocal.sum(axis=1)

    def _cluster_features(self, answers):
        n = len(answers)
        counts = Counter(a for a in answers if a)
        if not counts:
            return np.zeros(n, dtype=np.float64)
        majority = max(counts.values())
        score = np.zeros(n, dtype=np.float64)
        for index, answer in enumerate(answers):
            if not answer:
                continue
            count = counts[answer]
            if count == majority:
                score[index] = -(count / n)
            elif count == 1:
                score[index] = 0.5
            else:
                score[index] = 1.0 + (1.0 - count / majority)
        return score

    def _score_agents(self, state, raw, normalised, embeddings, debate_round):
        base = self._node_scores(raw, normalised)
        n = raw.shape[0]
        if n < 2:
            return base

        strongest, total = self._reciprocity_features(raw)
        cluster = self._cluster_features(self._answers(debate_round))
        decay = min(max(float(getattr(self.config, "persistence_decay", 0.5)), 0.0), 0.95)
        cluster_ema = state.get("cluster_ema")
        if cluster_ema is None or np.shape(cluster_ema) != (n,):
            cluster_ema = cluster.copy()
        else:
            cluster_ema = decay * cluster_ema + (1.0 - decay) * cluster
        state["cluster_ema"] = cluster_ema

        base_weight = float(getattr(self.config, "base_weight", 0.3))
        reciprocity_weight = float(getattr(self.config, "reciprocity_weight", 0.3))
        cluster_weight = float(getattr(self.config, "cluster_weight", 1.0))
        persistence_weight = float(getattr(self.config, "persistence_weight", 1.0))

        score = (
            base_weight * base
            + reciprocity_weight
            * (
                self._standardise(strongest, self.epsilon)
                + self._standardise(total, self.epsilon)
            )
            + cluster_weight * self._standardise(cluster, self.epsilon)
            + persistence_weight * self._standardise(cluster_ema, self.epsilon)
        )
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


class Master(_BaseMaster):
    def _run(self, train_pkl_path=None):
        detector = CASPIANDetector(self.args)
        log_info("Initialized CASPIAN V3 (reciprocity + synchronized minority + persistence).")
        log_done("CASPIAN V3 ready.")
        return {}, detector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASPIAN V3 detector")
    parser.add_argument("--config", required=True)
    parsed = parser.parse_args()
    Master(parsed.config)._run()
