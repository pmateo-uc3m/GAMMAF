import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASPIAN = load_module(ROOT / "defense-models" / "CASPIAN.py", "caspian_test")
EVALUATION = load_module(ROOT / "EvaluationDebateLoop.py", "evaluation_test")


class CasiPianCompatibilityTests(unittest.TestCase):
    def make_detector(self):
        return CASPIAN.CASPIANDetector(
            SimpleNamespace(
                epsilon=1.0e-8,
                target_ema_decay=0.8,
                influence_ema_decay=0.8,
                max_persistence_window=8,
                spine_top_k=3,
                top_k=1,
            )
        )

    @staticmethod
    def round_data(values):
        return [
            {"agent_id": index, "st_embedding": np.asarray(value, dtype=np.float64)}
            for index, value in enumerate(values)
        ]

    def test_temporal_state_isolated_between_trace_ids(self):
        adjacency = [[0, 0], [1, 0]]
        first = self.round_data([[1.0, 0.0], [0.0, 1.0]])
        second = self.round_data([[0.0, 1.0], [1.0, 0.0]])

        detector = self.make_detector()
        detector.begin_trace(("chain", 0), adjacency)
        detector.predict(first, adjacency, trace_id=("chain", 0))
        detector.predict(second, adjacency, trace_id=("chain", 0))
        detector.end_trace(("chain", 0))
        isolated_flags, isolated_scores = detector.predict(
            first, adjacency, trace_id=("chain", 1)
        )

        fresh = self.make_detector()
        fresh_flags, fresh_scores = fresh.predict(first, adjacency, trace_id="fresh")
        np.testing.assert_array_equal(isolated_flags, fresh_flags)
        np.testing.assert_allclose(isolated_scores, fresh_scores)

    def test_edge_cases_return_finite_framework_shapes(self):
        detector = self.make_detector()
        flags, scores = detector.predict(
            [{"agent_id": 0, "st_embedding": np.zeros(3)}],
            [[0]],
            trace_id="single",
        )
        self.assertEqual(flags.shape, (1,))
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(int(flags.sum()), 0)

    def test_optional_trace_dispatch_preserves_legacy_models(self):
        orchestration = object.__new__(EVALUATION.LiveDebateOrchestration)
        orchestration._model_predict_lock = threading.Lock()

        class Legacy:
            def predict(self, round_data, adjacency):
                return [0], [0.0]

        class Temporal:
            def __init__(self):
                self.started = []
                self.received = []

            def begin_trace(self, trace_id, adjacency):
                self.started.append(trace_id)

            def predict(self, round_data, adjacency, trace_id=None):
                self.received.append(trace_id)
                return [0], [0.0]

        legacy_result = orchestration._predict_defense_model(
            Legacy(), [], [[0]], trace_id=("topology", 0)
        )
        temporal = Temporal()
        temporal_result = orchestration._predict_defense_model(
            temporal, [], [[0]], trace_id=("topology", 1)
        )

        self.assertEqual(legacy_result, ([0], [0.0]))
        self.assertEqual(temporal_result, ([0], [0.0]))
        self.assertEqual(temporal.started, [("topology", 1)])
        self.assertEqual(temporal.received, [("topology", 1)])

    def test_runner_assigns_and_cleans_trace_scope(self):
        orchestration = object.__new__(EVALUATION.LiveDebateOrchestration)
        orchestration.config = SimpleNamespace(
            new_random_each_question=False,
            num_questions=1,
            n_questions_on_random_topo=1,
            max_concurrent_inference=1,
            num_agents=2,
        )
        orchestration.answer_seed = 0
        orchestration._model_predict_lock = threading.Lock()

        class Temporal:
            def __init__(self):
                self.started = []
                self.received = []
                self.ended = []

            def begin_trace(self, trace_id, adjacency):
                self.started.append(trace_id)

            def predict(self, round_data, adjacency, trace_id=None):
                self.received.append(trace_id)
                return [0, 0], [0.0, 0.0]

            def end_trace(self, trace_id):
                self.ended.append(trace_id)

        model = Temporal()

        def fake_debate(self, defense_model, question, question_groundtruth,
                        choices, adjacency_matrix, trace_id=None, **kwargs):
            self._predict_defense_model(
                defense_model, [], adjacency_matrix, trace_id=trace_id
            )
            return {"ground_truth": question_groundtruth}

        orchestration.debate_question = MethodType(fake_debate, orchestration)
        traces = orchestration.run_debate_with_defense(
            [{"question": "q", "answer": "A"}],
            model,
            {"chain": [[0, 1], [1, 0]]},
        )

        self.assertEqual(model.started, [("chain", 0)])
        self.assertEqual(model.received, [("chain", 0)])
        self.assertEqual(model.ended, [("chain", 0)])
        self.assertEqual(traces["chain"][0]["ground_truth"], "A")


if __name__ == "__main__":
    unittest.main()
