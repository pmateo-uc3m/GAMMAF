"""Guardian evaluation adapter.

The normal evaluation loop intentionally remains unchanged.  This adapter
loads it and overrides only debate execution so each independent question has
its own ordered embedding history.
"""

import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_root = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("EvaluationDebateLoop_original", _root / "EvaluationDebateLoop.py")
_original = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _original
_spec.loader.exec_module(_original)
for _name in dir(_original):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_original, _name)


class LiveDebateOrchestration(_original.LiveDebateOrchestration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep embedding work in a bounded pool separate from the debate pool.
        # Round N+1 still waits on its own round N, while independent debates
        # can use different processor workers concurrently.
        processor_workers = max(
            1,
            int(getattr(self.config, "max_concurrent_inference", 1))
            // max(1, int(getattr(self.config, "num_agents", 1))),
        )
        self._guardian_processor_pool = ThreadPoolExecutor(
            max_workers=processor_workers,
            thread_name_prefix="guardian-round-processor",
        )
        self._guardian_processor_workers = processor_workers
        self._guardian_processor_init_lock = threading.Lock()

    def _process_on_guardian_worker(self, responses):
        # RoundProcessor keeps resources in thread-local storage. Initialize
        # each worker's resources one at a time to avoid concurrent model/cache
        # construction, then allow the expensive inference calls to overlap.
        get_resources = getattr(self.text_processor, "_get_local_resources", None)
        if get_resources is not None:
            with self._guardian_processor_init_lock:
                get_resources()
        return self.text_processor.process_round(responses)

    def _process_guardian_round(self, responses, question_index=None, round_number=None):
        if self._guardian_processor_pool is None:
            raise RuntimeError("Guardian RoundProcessor pool has already been shut down")
        future = self._guardian_processor_pool.submit(self._process_on_guardian_worker, responses)
        try:
            return future.result()
        except BaseException as exc:
            future.cancel()
            context = f"question={question_index}, round={round_number}"
            raise RuntimeError(f"RoundProcessor failed for {context}: {exc}") from exc

    def shutdown_guardian_processors(self, wait=True, cancel_pending=False):
        """Stop processor workers without affecting already collected traces."""
        pool = getattr(self, "_guardian_processor_pool", None)
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=cancel_pending)
            self._guardian_processor_pool = None

    def debate_question(self, defense_model, question, question_groundtruth, choices,
                        adjacency_matrix, mal_answer="", question_index=None,
                        question_format_data=None):
        if hasattr(defense_model, "reset"):
            defense_model.reset()
        if not hasattr(defense_model, "config"):
            from types import SimpleNamespace
            defense_model.config = SimpleNamespace()
        if not hasattr(defense_model.config, "top_k"):
            defense_model.config.top_k = getattr(self.config, "top_k_defense", 1)
        agents = self.generate_agents(question_index=question_index)
        flags = [0] * len(agents)
        flags_ground_truth = [agent.is_malicious for agent in agents]
        answer_rng = np.random.default_rng(self.answer_seed + (question_index or 0))
        if not mal_answer and sum(flags_ground_truth) > 0:
            mal_answer = answer_rng.choice(["A", "B", "C", "D"]).item()

        history = []
        debate_trace = []
        last_round_responses = self.generate_round_1_concurrent(
            question, choices, agents, mal_answer=mal_answer,
            question_format_data=question_format_data, round_num=1)
        history.append(self._process_guardian_round(last_round_responses, question_index, 1))
        static_adjacency = copy.deepcopy(adjacency_matrix)
        static_mode = getattr(self.config, "static_adjacency_mode", False)
        predict_adj = static_adjacency if static_mode else adjacency_matrix
        with self._model_predict_lock:
            flags, anomaly_scores = defense_model.predict(history[-1], predict_adj, temporal_rounds=list(history))
        adjacency_matrix = modify_adjacency(flags, adjacency_matrix)
        debate_trace.append({"round": 1, "responses": [{"agent_id": r["agent_id"], "answer": r["answer"]} for r in last_round_responses],
                             "flags": flags, "AUROC": roc_auc_score(flags_ground_truth, anomaly_scores),
                             "anomaly_scores": anomaly_scores})

        consensus = False
        for i in range(1, self.config.max_rounds):
            unflagged = [r for r, flag in zip(last_round_responses, flags) if flag == 0]
            if self.config.check_consensus_only_unflagged:
                if self.check_consensus(unflagged):
                    consensus = True
                    break
            elif self.check_consensus(last_round_responses):
                consensus = True
                break
            last_round_responses = self.generate_debate_round_concurrent(
                adjacency_matrix, question, choices, last_round_responses, agents, round=i,
                mal_answer=mal_answer, question_format_data=question_format_data)
            history.append(self._process_guardian_round(last_round_responses, question_index, i + 1))
            predict_adj = static_adjacency if static_mode else adjacency_matrix
            with self._model_predict_lock:
                flags, anomaly_scores = defense_model.predict(history[-1], predict_adj, temporal_rounds=list(history))
            adjacency_matrix = modify_adjacency(flags, adjacency_matrix)
            debate_trace.append({"round": i, "responses": [{"agent_id": r["agent_id"], "answer": r["answer"]} for r in last_round_responses],
                                 "flags": flags, "AUROC": roc_auc_score(flags_ground_truth, anomaly_scores),
                                 "anomaly_scores": anomaly_scores})

        remaining = [r for r, flag in zip(last_round_responses, flags) if flag == 0]
        if not remaining:
            raise RuntimeError("Guardian flagged every agent; cannot select a final answer")
        return {"ground_truth": question_groundtruth, "question": question,
                "final_answer": self.get_answer(remaining),
                "is_correct": self.check_answer(last_round_responses, question_groundtruth),
                "consensus": consensus, "rounds": len(debate_trace),
                "debate_trace": debate_trace, "flags_ground_truth": flags_ground_truth}
