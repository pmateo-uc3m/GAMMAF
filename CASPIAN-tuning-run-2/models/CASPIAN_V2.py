"""CASPIAN V2 — reciprocity + synchronized-minority answer cluster.

V2 keeps the whole B0 pipeline and adds the one signal that actually separates
the injected agents in the communication-only GAMMAF setting: a synchronized
minority answer cluster.

Why not message similarity: the pooled MiniLM embeddings are dominated by the
shared question/choices, so agents that pick the same answer have near-identical
messages whether or not they are coordinated (measured mean cosine 0.83 on
benign data).  The discrete answer, however, exposes coordination: the two
injected agents always share one answer, whereas on benign rounds a same-answer
pair occurs in only ~10% of rounds and the usual safe majority has size 5-6.

Cluster score (per agent, answer multiplicity ``c`` and majority multiplicity
``maj`` over the round):

    c == maj :  -(c / n)            (majority: low anomaly, more so if larger)
    c == 1   :   0.5               (lonely wrong answer)
    1 < c < maj : 1 + (1 - c / maj) (small coordinated minority: high)

The score is added to the influence/reciprocity evidence, so CASPIAN's
attribution still contributes while the round-1 decision is driven by the
synchronized minority.
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
        base_weight = float(getattr(self.config, "base_weight", 0.3))
        reciprocity_weight = float(getattr(self.config, "reciprocity_weight", 0.3))
        cluster_weight = float(getattr(self.config, "cluster_weight", 1.0))

        strongest, total = self._reciprocity_features(raw)
        cluster = self._cluster_features(self._answers(debate_round))

        score = (
            base_weight * base
            + reciprocity_weight
            * (
                self._standardise(strongest, self.epsilon)
                + self._standardise(total, self.epsilon)
            )
            + cluster_weight * self._standardise(cluster, self.epsilon)
        )
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


class Master(_BaseMaster):
    def _run(self, train_pkl_path=None):
        detector = CASPIANDetector(self.args)
        log_info("Initialized CASPIAN V2 (reciprocity + synchronized-minority cluster).")
        log_done("CASPIAN V2 ready.")
        return {}, detector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASPIAN V2 detector")
    parser.add_argument("--config", required=True)
    parsed = parser.parse_args()
    Master(parsed.config)._run()
