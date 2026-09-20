"""CASPIAN V5 — coordination-first fusion with minority-pair evidence.

V5 is the iterated combination of the previous variants.  It keeps every
CASPIAN component (LI-CTE influence, degree normalization, spectral cascade
detection, attribution) and fuses the evidence so that the *synchronized
minority cluster* dominates the round-1 decision:

* ``cluster`` — V2's answer-multiplicity score (small non-majority groups are
  anomalous, the large majority is not);
* ``minority_pair`` — the members of the smallest non-majority answer group
  (size 2 for the typical injected pair; size 3 if one benign agent happened to
  copy the injected answer).  Both/all members receive the boost;
* ``cluster_ema`` — V3's persistence across turns;
* ``deviation`` — V4's neighbourhood consensus deviation;
* a small standardised influence/reciprocity tie-break so CASPIAN's
  attribution still breaks ties among equally-suspicious clusters.

Message similarity is deliberately NOT used as a primary term: on benign data
same-answer messages have mean cosine 0.83 because the pooled embedding is
dominated by the shared question, so it carries almost no coordination signal.
It is used only to break ties between multiple size-2 groups.
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

    def _minority_group(self, answers, embeddings):
        n = len(answers)
        counts = Counter(a for a in answers if a)
        if not counts:
            return np.zeros(n, dtype=np.float64)
        majority = max(counts.values())
        candidates = {a: c for a, c in counts.items() if c < majority and c >= 2}
        if not candidates:
            return np.zeros(n, dtype=np.float64)
        smallest = min(candidates.values())
        groups = [a for a, c in candidates.items() if c == smallest]
        if len(groups) > 1 and smallest == 2:
            normalised = embeddings / np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), self.epsilon
            )
            similarity = normalised @ normalised.T
            best, best_score = None, -np.inf
            for answer in groups:
                members = [i for i, a in enumerate(answers) if a == answer]
                if len(members) == 2:
                    score = float(similarity[members[0], members[1]])
                    if score > best_score:
                        best_score, best = score, members
            if best is not None:
                groups = [answers[best[0]]]
        boost = np.zeros(n, dtype=np.float64)
        for answer in groups:
            for index, a in enumerate(answers):
                if a == answer:
                    boost[index] = 1.0
        return boost

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
            deviation[index] = 1.0 - Counter(peer_answers)[answers[index]] / len(peer_answers)
        return deviation

    def _score_agents(self, state, raw, normalised, embeddings, debate_round):
        base = self._node_scores(raw, normalised)
        n = raw.shape[0]
        if n < 2:
            return base

        answers = self._answers(debate_round)
        strongest, total = self._reciprocity_features(raw)
        cluster = self._cluster_features(answers)
        minority = self._minority_group(answers, embeddings)
        deviation = self._neighbourhood_deviation(state, answers)

        decay = min(max(float(getattr(self.config, "persistence_decay", 0.5)), 0.0), 0.95)
        cluster_ema = state.get("cluster_ema")
        if cluster_ema is None or np.shape(cluster_ema) != (n,):
            cluster_ema = cluster.copy()
        else:
            cluster_ema = decay * cluster_ema + (1.0 - decay) * cluster
        state["cluster_ema"] = cluster_ema

        base_weight = float(getattr(self.config, "base_weight", 0.2))
        reciprocity_weight = float(getattr(self.config, "reciprocity_weight", 0.2))
        cluster_weight = float(getattr(self.config, "cluster_weight", 1.0))
        persistence_weight = float(getattr(self.config, "persistence_weight", 0.8))
        pair_weight = float(getattr(self.config, "pair_weight", 1.5))
        deviation_weight = float(getattr(self.config, "deviation_weight", 0.3))

        score = (
            base_weight * base
            + reciprocity_weight
            * (
                self._standardise(strongest, self.epsilon)
                + self._standardise(total, self.epsilon)
            )
            + cluster_weight * self._standardise(cluster, self.epsilon)
            + persistence_weight * self._standardise(cluster_ema, self.epsilon)
            + pair_weight * minority
            + deviation_weight * self._standardise(deviation, self.epsilon)
        )
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


class Master(_BaseMaster):
    def _run(self, train_pkl_path=None):
        detector = CASPIANDetector(self.args)
        log_info("Initialized CASPIAN V5 (coordination-first fusion + minority pair).")
        log_done("CASPIAN V5 ready.")
        return {}, detector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASPIAN V5 detector")
    parser.add_argument("--config", required=True)
    parsed = parser.parse_args()
    Master(parsed.config)._run()
