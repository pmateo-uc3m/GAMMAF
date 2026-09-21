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

The module also embeds the communication-only CASPIAN base detector
(``_CASPIANDetector``), which V5 extends.  The base observes communication
influence only, since GAMMAF exposes just the current round's message
embeddings and an adjacency matrix; it follows Appendix C's Gaussian-copula
construction and the paper's online Watch/cascade rules.
"""

import argparse
import math
import threading
from collections import Counter

import numpy as np

from EvaluationConfigCheck import load_defense_model_config
from LoggingUtils import log_done, log_info


class _CASPIANDetector:
    """Online spectral cascade detector adapted to communication evidence.

    The original CASPIAN method observes communication, memory, tool, and
    execution channels.  GAMMAF exposes only the current round's message
    embeddings and an adjacency matrix, so this implementation keeps one
    communication influence matrix and does not synthesize the unavailable
    channels.

    The detector is intentionally online and training-free.  ``begin_trace``
    and ``end_trace`` are optional lifecycle hooks used by the evaluation loop
    to keep spectral histories isolated when questions are evaluated
    concurrently.
    """

    def __init__(self, config):
        self.config = config
        self.epsilon = max(float(config.epsilon), 1e-12)
        self.target_ema_decay = min(max(float(config.target_ema_decay), 0.0), 1.0)
        self.influence_ema_decay = min(max(float(config.influence_ema_decay), 0.0), 1.0)
        # The paper derives W from the inverse gap without a finite cap.  This
        # safety bound prevents an almost-zero gap from retaining unbounded
        # state in a finite-round evaluator.
        self.max_persistence_window = max(1, int(config.max_persistence_window))
        self.spine_top_k = max(1, int(config.spine_top_k))
        self._states = {}
        self._lock = threading.RLock()
        self.last_attribution = None

    def begin_trace(self, trace_id, adjacency_matrix):
        """Create trace state once; repeated calls for the same trace are safe."""
        key = self._key(trace_id)
        adjacency = self._validate_adjacency(adjacency_matrix)
        with self._lock:
            if key not in self._states:
                self._states[key] = self._new_state(adjacency)

    def end_trace(self, trace_id):
        """Release all temporal state for an independently evaluated trace."""
        with self._lock:
            self._states.pop(self._key(trace_id), None)

    def reset(self):
        """Compatibility reset for callers that manage model lifecycle directly."""
        with self._lock:
            self._states.clear()
            self.last_attribution = None

    @staticmethod
    def _key(trace_id):
        try:
            hash(trace_id)
            return trace_id
        except TypeError:
            return repr(trace_id)

    def _validate_adjacency(self, adjacency_matrix):
        adjacency = np.asarray(adjacency_matrix, dtype=np.float64)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f"CASPIAN adjacency must be square, got shape {adjacency.shape}"
            )
        if not np.all(np.isfinite(adjacency)):
            raise ValueError("CASPIAN adjacency contains non-finite values")
        return (adjacency != 0).astype(np.float64)

    def _new_state(self, adjacency):
        # The debate loop indexes row i by the messages received by agent i.
        # Transposing gives the paper's source-agent-to-target-agent orientation.
        structural = adjacency.T.copy()
        return {
            "structural": structural,
            "n": structural.shape[0],
            "embedding_dim": None,
            "target_history": None,
            "influence": np.zeros_like(structural, dtype=np.float64),
            "step": 0,
            "previous_energy": None,
            "previous_lambda1": None,
            "previous_ratio": None,
            "previous_gap": None,
            "watch_start": None,
            "watch_window": None,
            "watch_records": [],
            "cascade_emitted": False,
            "cascade_agents": np.zeros(structural.shape[0], dtype=int),
        }

    @staticmethod
    def _normalise_rows(values, epsilon):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, epsilon)

    @staticmethod
    def _rank_normalise_rows(values, epsilon):
        """Apply marginal rank normalization used by the copula approximation."""
        ranked = np.empty_like(values, dtype=np.float64)
        for row_index, row in enumerate(values):
            order = np.argsort(row, kind="stable")
            ranks = np.empty(row.size, dtype=np.float64)
            ranks[order] = np.arange(row.size, dtype=np.float64)
            ranks -= ranks.mean()
            norm = np.linalg.norm(ranks)
            if norm > epsilon:
                ranked[row_index] = ranks / norm
            else:
                ranked[row_index] = 0.0
        return ranked

    def _extract_embeddings(self, debate_round, n):
        if not isinstance(debate_round, (list, tuple)):
            raise ValueError("CASPIAN debate_round must be a list of agent dictionaries")
        if len(debate_round) != n:
            raise ValueError(
                f"CASPIAN received {len(debate_round)} agents for a {n}x{n} topology"
            )

        ordered = list(debate_round)
        ids = [item.get("agent_id") for item in ordered if isinstance(item, dict)]
        if len(ids) == n and all(isinstance(agent_id, (int, np.integer)) for agent_id in ids):
            if sorted(int(agent_id) for agent_id in ids) == list(range(n)):
                ordered = sorted(ordered, key=lambda item: int(item["agent_id"]))

        vectors = []
        for index, agent in enumerate(ordered):
            if not isinstance(agent, dict) or "st_embedding" not in agent:
                raise ValueError(f"CASPIAN agent {index} is missing st_embedding")
            vector = np.asarray(agent["st_embedding"], dtype=np.float64).reshape(-1)
            if vector.size == 0:
                raise ValueError(f"CASPIAN agent {index} has an empty st_embedding")
            vectors.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))

        embeddings = np.asarray(vectors, dtype=np.float64)
        if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
            raise ValueError("CASPIAN embeddings must be finite two-dimensional vectors")
        return embeddings

    def _get_state(self, trace_id, adjacency):
        key = self._key(trace_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = self._new_state(adjacency)
                self._states[key] = state
            elif state["n"] != adjacency.shape[0]:
                raise ValueError("CASPIAN trace topology changed its agent count")
            return key, state

    def _communication_influence(self, state, embeddings, active_adjacency):
        epsilon = self.epsilon
        current = self._normalise_rows(embeddings, epsilon)
        history = state["target_history"]
        if history is None:
            history = np.zeros_like(current)

        # Communication-only late-interaction CTE approximation.  Appendix C
        # specifies a Gaussian-copula conditional dependence built from the
        # covariance blocks of the (source, target, history) system.  With a
        # per-turn vector adaptation this is the rank-domain partial
        # correlation rho(u_i, v_j | h_j) = (rho_uv - rho_uh rho_vh) /
        # sqrt((1-rho_uh^2)(1-rho_vh^2)), whose Gaussian conditional mutual
        # information is -0.5 log(1 - rho^2) and is clipped to be nonnegative.
        source_copula = self._rank_normalise_rows(current, epsilon)
        history_copula = self._rank_normalise_rows(history, epsilon)
        rho_uv = source_copula @ source_copula.T
        rho_uh = source_copula @ history_copula.T
        rho_vh = np.einsum("jj->j", rho_uh)
        denominator = np.sqrt(
            np.maximum(1.0 - rho_uh * rho_uh, epsilon)
            * np.maximum(1.0 - rho_vh[None, :] * rho_vh[None, :], epsilon)
        )
        partial = (rho_uv - rho_uh * rho_vh[None, :]) / denominator
        partial = np.clip(partial, 0.0, 1.0)
        conditional_mi = -0.5 * np.log(
            np.maximum(1.0 - partial * partial, epsilon)
        )
        residual_norm = np.linalg.norm(current - history, axis=1)
        novelty = np.minimum(1.0, residual_norm)
        instantaneous = conditional_mi * novelty[None, :]

        structural = state["structural"]
        active_source_target = active_adjacency.T
        feasible = (structural > 0) & (active_source_target > 0)
        instantaneous *= feasible

        previous = state["influence"]
        decay = self.influence_ema_decay
        influence = decay * previous + (1.0 - decay) * instantaneous
        influence *= feasible

        state["target_history"] = (
            self.target_ema_decay * history
            + (1.0 - self.target_ema_decay) * current
        )
        return np.nan_to_num(influence, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalised_influence(self, influence):
        outgoing = influence.sum(axis=1)
        incoming = influence.sum(axis=0)
        denominator = np.sqrt(np.outer(outgoing, incoming)) + self.epsilon
        normalised = influence / denominator
        return np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)

    def _spectral_values(self, normalised):
        if normalised.shape[0] == 0:
            return 0.0, 0.0
        values = np.linalg.svd(normalised, compute_uv=False)
        lambda1 = float(values[0]) if values.size else 0.0
        lambda2 = float(values[1]) if values.size > 1 else 0.0
        return lambda1, lambda2

    def _graph_diameter(self, structural):
        n = structural.shape[0]
        distances = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(distances, 0.0)
        distances[structural > 0] = 1.0
        for middle in range(n):
            distances = np.minimum(
                distances,
                distances[:, middle, None] + distances[None, middle, :],
            )
        finite = distances[np.isfinite(distances) & (distances > 0)]
        return int(np.max(finite)) if finite.size else 0

    def _weak_link(self, state, normalised):
        structural = state["structural"] > 0
        n = structural.shape[0]
        if n < 2 or not np.any(structural):
            return False, 0.0, 0.0

        direct = np.where(structural, np.maximum(normalised, 0.0), 0.0)
        diameter = self._graph_diameter(state["structural"])
        if diameter < 1:
            return False, 0.0, 0.0

        best = direct.copy()
        exact = direct.copy()
        for _ in range(2, diameter + 1):
            next_exact = np.zeros_like(exact)
            for middle in range(n):
                next_exact = np.maximum(
                    next_exact,
                    np.minimum(exact[:, middle, None], direct[middle, None, :]),
                )
            exact = next_exact
            best = np.maximum(best, exact)

        bottleneck = float(np.max(best)) if best.size else 0.0
        edge_weights = direct[structural]
        total = float(np.sum(edge_weights))
        energy_scale = float(np.sum(edge_weights * edge_weights) / (total + self.epsilon))
        feasible = total > self.epsilon and bottleneck + self.epsilon >= energy_scale
        return feasible, bottleneck, energy_scale

    @staticmethod
    def _standardise(values, epsilon):
        """Zero-mean/unit-variance scaling that keeps ties from saturating."""
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        spread = float(np.std(values))
        if spread <= epsilon:
            return np.zeros_like(values)
        return (values - float(np.mean(values))) / spread

    def _node_scores(self, raw, normalised):
        """Single-turn attribution score for each agent (Eqs. (11)-(13)).

        The paper's attribution statistics are (i) the outgoing influence
        (origin), (ii) the amplifier ratio outgoing/incoming, and (iii) the
        bridge product outgoing*incoming.  A standardised sum keeps the score
        continuous, which is required for a well-defined AUROC.
        """
        outgoing = normalised.sum(axis=1)
        incoming = normalised.sum(axis=0)
        amplifier = outgoing / (incoming + self.epsilon)
        raw_outgoing = raw.sum(axis=1)
        raw_incoming = raw.sum(axis=0)
        bridge = raw_outgoing * raw_incoming

        weights = self.config.component_weights
        score = (
            float(weights.get("outgoing", 0.0))
            * self._standardise(outgoing, self.epsilon)
            + float(weights.get("amplifier", 0.0))
            * self._standardise(amplifier, self.epsilon)
            + float(weights.get("bridge", 0.0))
            * self._standardise(bridge, self.epsilon)
        )
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def _score_agents(self, state, raw, normalised, embeddings, debate_round):
        """Extension hook: per-agent anomaly score for the current turn.

        The default implementation is the single-turn attribution score.  Model
        variants override this to add further CASPIAN-faithful evidence without
        touching the spectral detection or attribution code.
        """
        return self._node_scores(raw, normalised)

    def _enumerate_paths(self, state, weights):
        structural = state["structural"] > 0
        n = structural.shape[0]
        diameter = self._graph_diameter(state["structural"])
        if n < 2 or diameter < 1:
            return []

        paths = []

        def visit(source, node, path):
            if len(path) - 1 >= diameter:
                return
            for target in range(n):
                if not structural[node, target] or target in path:
                    continue
                next_path = path + [target]
                edge_values = [weights[path[i], path[i + 1]] for i in range(len(path) - 1)]
                edge_values.append(weights[node, target])
                paths.append((float(min(edge_values)), tuple(next_path)))
                visit(source, target, next_path)

        for source in range(n):
            visit(source, source, [source])
        paths.sort(key=lambda item: (-item[0], item[1]))
        return paths

    def _attribute(self, state, records):
        if not records:
            return None
        onset_normalised = records[0]["normalised"]
        origin = int(np.argmax(onset_normalised.sum(axis=1)))

        amplifier_totals = np.zeros(state["n"], dtype=np.float64)
        bridge_totals = np.zeros(state["n"], dtype=np.float64)
        max_normalised = np.zeros_like(onset_normalised)
        for record in records:
            normalised = record["normalised"]
            raw = record["raw"]
            amplifier_totals += normalised.sum(axis=1) / (
                normalised.sum(axis=0) + self.epsilon
            )
            bridge_totals += raw.sum(axis=1) * raw.sum(axis=0)
            max_normalised = np.maximum(max_normalised, normalised)

        amplifier = int(np.argmax(amplifier_totals))
        bridge = int(np.argmax(bridge_totals))
        spines = [
            {"path": list(path), "bottleneck": score, "channel": "communication"}
            for score, path in self._enumerate_paths(state, max_normalised)[: self.spine_top_k]
        ]
        return {
            "origin": origin,
            "amplifier": amplifier,
            "bridge": bridge,
            "spines": spines,
            "channel": "communication",
        }

    def _flags_for_scores(self, scores, state):
        """Flag the top-``top_k`` attributed agents on every turn.

        GAMMAF consumes flags each round to isolate agents.  The paper only
        defines a flag set at cascade confirmation, which would leave the first
        turn (and every turn with no confirmed cascade) unflagged.  Reporting
        the current attribution ranking every turn gives the framework the
        round-1 flagging it requires while keeping the score semantics.
        """
        n = scores.size
        flags = np.zeros(n, dtype=int)
        if n < 2:
            return flags
        count = min(max(1, int(self.config.top_k)), n - 1)
        if count > 0:
            flags[np.argsort(-scores, kind="stable")[:count]] = 1
        return flags

    def predict(self, debate_round, adj_matrix, trace_id=None):
        """Return framework-compatible per-agent flags and anomaly scores."""
        adjacency = self._validate_adjacency(adj_matrix)
        key, state = self._get_state(trace_id, adjacency)
        embeddings = self._extract_embeddings(debate_round, state["n"])
        if state["embedding_dim"] is None:
            state["embedding_dim"] = embeddings.shape[1]
            state["target_history"] = np.zeros_like(embeddings)
        elif state["embedding_dim"] != embeddings.shape[1]:
            raise ValueError("CASPIAN embedding dimension changed within a trace")

        with self._lock:
            raw = self._communication_influence(state, embeddings, adjacency)
            normalised = self._normalised_influence(raw)
            lambda1, lambda2 = self._spectral_values(normalised)
            energy = lambda1 + lambda2
            ratio = lambda2 / (lambda1 + self.epsilon)
            gap = 1.0 - ratio

            previous_energy = state["previous_energy"]
            previous_lambda1 = state["previous_lambda1"]
            previous_ratio = state["previous_ratio"]
            previous_gap = state["previous_gap"]
            warm = previous_energy is not None

            amplification = (
                energy / (previous_energy + self.epsilon) if warm else 0.0
            )
            gap_contraction = previous_gap - gap if warm else 0.0
            phase_magnitude = (
                abs(ratio - previous_ratio) / (previous_ratio + self.epsilon)
                if warm
                else 0.0
            )
            phase_shift = bool(warm and phase_magnitude > gap_contraction)
            watch = bool(
                warm
                and amplification > 1.0
                and gap_contraction > 0.0
                and lambda1 > previous_lambda1
            )
            weak_link, bottleneck, energy_scale = self._weak_link(state, normalised)
            cross_channel = False
            transition = phase_shift or cross_channel

            record = {
                "raw": raw.copy(),
                "normalised": normalised.copy(),
                "watch": watch,
                "phase_shift": phase_shift,
                "cross_channel": cross_channel,
            }

            if not state["cascade_emitted"]:
                if watch and state["watch_start"] is None:
                    state["watch_start"] = state["step"]
                    state["watch_window"] = min(
                        self.max_persistence_window,
                        max(1, int(math.ceil(1.0 / (gap + self.epsilon)))),
                    )
                    state["watch_records"] = []

                if state["watch_start"] is not None:
                    if not watch:
                        # The paper's online description discards a candidate
                        # when Watch drops before confirmation.
                        state["watch_start"] = None
                        state["watch_window"] = None
                        state["watch_records"] = []
                    else:
                        state["watch_records"].append(record)
                        # Algorithm 1 evaluates the instant rule only at the
                        # Watch onset turn (t == tw); later turns rely on the
                        # multi-turn confirmation rule.
                        onset_turn = state["watch_start"] == state["step"]
                        instant = bool(watch and onset_turn and transition and weak_link)
                        interval_complete = len(state["watch_records"]) >= state["watch_window"]
                        watch_count = sum(
                            int(item["watch"]) for item in state["watch_records"]
                        )
                        majority_watch = watch_count * 2 >= state["watch_window"]
                        transition_seen = any(
                            item["phase_shift"] or item["cross_channel"]
                            for item in state["watch_records"]
                        )
                        multi_turn = bool(
                            interval_complete and majority_watch and transition_seen
                        )
                        if instant or multi_turn:
                            state["cascade_emitted"] = True
                            state["watch_records"] = state["watch_records"][-state["watch_window"] :]
                            attribution = self._attribute(state, state["watch_records"])
                            self.last_attribution = attribution
                            if attribution is not None:
                                state["cascade_agents"] = np.zeros(state["n"], dtype=int)
                                for role in ("origin", "amplifier", "bridge"):
                                    state["cascade_agents"][attribution[role]] = 1

            scores = self._score_agents(state, raw, normalised, embeddings, debate_round)
            flags = self._flags_for_scores(scores, state)
            state["previous_energy"] = energy
            state["previous_lambda1"] = lambda1
            state["previous_ratio"] = ratio
            state["previous_gap"] = gap
            state["influence"] = raw
            state["step"] += 1

            # Keep the last computed signals inspectable without changing the
            # framework's two-value prediction contract.
            state["last_signals"] = {
                "lambda1": lambda1,
                "lambda2": lambda2,
                "energy": energy,
                "amplification": amplification,
                "coupling_ratio": ratio,
                "normalised_gap": gap,
                "gap_contraction": gap_contraction,
                "phase_magnitude": phase_magnitude,
                "watch": watch,
                "phase_shift": phase_shift,
                "cross_channel": cross_channel,
                "weak_link": weak_link,
                "weak_link_bottleneck": bottleneck,
                "weak_link_energy_scale": energy_scale,
                "cascade": state["cascade_emitted"],
                "trace_id": key,
            }
            return flags, np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


class CASPIANDetector(_CASPIANDetector):
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

        decay = min(max(float(self.config.persistence_decay), 0.0), 0.95)
        cluster_ema = state.get("cluster_ema")
        if cluster_ema is None or np.shape(cluster_ema) != (n,):
            cluster_ema = cluster.copy()
        else:
            cluster_ema = decay * cluster_ema + (1.0 - decay) * cluster
        state["cluster_ema"] = cluster_ema

        base_weight = float(self.config.base_weight)
        reciprocity_weight = float(self.config.reciprocity_weight)
        cluster_weight = float(self.config.cluster_weight)
        persistence_weight = float(self.config.persistence_weight)
        pair_weight = float(self.config.pair_weight)
        deviation_weight = float(self.config.deviation_weight)

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


class Master:
    """MainEvaluation adapter for training-free CASPIAN."""

    def __init__(self, config_path):
        self.args = load_defense_model_config(config_path)

    def _run(self, train_pkl_path=None):
        # CASPIAN is explicitly online and does not fit an offline detector.
        detector = CASPIANDetector(self.args)
        log_info("Initialized CASPIAN V5 (coordination-first fusion + minority pair).")
        log_done("CASPIAN V5 ready.")
        return {}, detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASPIAN V5 detector")
    parser.add_argument("--config", required=True, help="Path to CASPIAN YAML config")
    parsed = parser.parse_args()
    Master(parsed.config)._run()
