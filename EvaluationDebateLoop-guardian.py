"""Guardian evaluation adapter.

The normal evaluation loop intentionally remains unchanged.  This adapter
loads it and overrides only debate execution so each independent question has
its own ordered embedding history.
"""

import importlib.util
import sys
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
        history.append(self.text_processor.process_round(last_round_responses))
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
            history.append(self.text_processor.process_round(last_round_responses))
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
