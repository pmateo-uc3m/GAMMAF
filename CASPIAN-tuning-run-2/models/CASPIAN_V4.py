"""CASPIAN V4 — reciprocity + synchronized minority + neighbourhood consensus + cascade roles.

V4 extends V2 with two structural signals that follow directly from the
paper's formulation:

1. *Neighbourhood consensus deviation*: the paper masks influence by the
   structural possibility graph ``G0`` (Appendix C, step 1) and models a
   cascade as propagation through that graph, so an agent whose answer
   disagrees with its structural neighbours is a candidate source of divergent
   propagation.
2. *Cascade-role boost*: when the spectral rules confirm a cascade, CASPIAN's
   attribution returns origin/amplifier/bridge (Eqs. (11)-(13)); V4 feeds those
   roles into the per-agent score so the audited attribution is reflected in
   the flags.

The influence accumulation still has the residual problem that after a few
rounds the graph can be emptied by flagging, which makes the raw score
degenerate; the neighbourhood and cluster terms keep the ranking informative.
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

    def _neighbourhood_deviation(self, state, answers):
        structural = state.get("structural")
        n = len(answers)
        if structural is None or structural.size == 0:
            return np.zeros(n, dtype=np.float64)
        neighbourhood = (structural > 0) | (structural.T > 0)
        np.fill_diagonal(neighbourhood, False)
        deviation = np.zeros(n, dtype=np.float64)
        for index in range(n):
            peers = np.where(neighbourhood[index])[0]
            if peers.size == 0 or not answers[index]:
                continue
            peer_answers = [answers[p] for p in peers if answers[p]]
            if not peer_answers:
                continue
            counts = Counter(peer_answers)
            deviation[index] = 1.0 - counts[answers[index]] / len(peer_answers)
        return deviation

    def _score_agents(self, state, raw, normalised, embeddings, debate_round):
        base = self._node_scores(raw, normalised)
        n = raw.shape[0]
        if n < 2:
            return base

        answers = self._answers(debate_round)
        strongest, total = self._reciprocity_features(raw)
        cluster = self._cluster_features(answers)
        deviation = self._neighbourhood_deviation(state, answers)
        cascade_roles = np.asarray(
            state.get("cascade_agents", np.zeros(n, dtype=int)), dtype=np.float64
        )

        base_weight = float(getattr(self.config, "base_weight", 0.3))
        reciprocity_weight = float(getattr(self.config, "reciprocity_weight", 0.3))
        cluster_weight = float(getattr(self.config, "cluster_weight", 1.0))
        deviation_weight = float(getattr(self.config, "deviation_weight", 0.5))
        cascade_weight = float(getattr(self.config, "cascade_weight", 0.5))

        score = (
            base_weight * base
            + reciprocity_weight
            * (
                self._standardise(strongest, self.epsilon)
                + self._standardise(total, self.epsilon)
            )
            + cluster_weight * self._standardise(cluster, self.epsilon)
            + deviation_weight * self._standardise(deviation, self.epsilon)
            + cascade_weight * cascade_roles
        )
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


class Master(_BaseMaster):
    def _run(self, train_pkl_path=None):
        detector = CASPIANDetector(self.args)
        log_info("Initialized CASPIAN V4 (cluster + neighbourhood + cascade roles).")
        log_done("CASPIAN V4 ready.")
        return {}, detector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASPIAN V4 detector")
    parser.add_argument("--config", required=True)
    parsed = parser.parse_args()
    Master(parsed.config)._run()
