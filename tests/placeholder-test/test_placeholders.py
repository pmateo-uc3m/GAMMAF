"""Test artifact -- validates the three new prompt placeholders end to end.

This is NOT a production script.  It runs two small debates (3 agents, one
fixed directed topology, 2 rounds, 1 malicious agent) through the placeholder-
aware ``-complete`` pipelines and records every ``format_data`` dictionary and
rendered prompt:

    1. Evaluation loop  -- ``EvaluationDebateLoop-complete.py`` with a scripted
       defense model that flags the true malicious agents after round 1.
    2. Generation loop  -- ``DebateDataGenerationLoop-complete.py`` (no defense;
       ``flags_string`` is therefore always empty).

Prompts come from
``tests/placeholder-test/prompts/prompts-placeholder-test.json`` and contain
``{topology_string}`` / ``{malicious_agents_string}`` / ``{flags_string}`` in
the system, first-round and debate prompts.  Evidence is written to
``tests/placeholder-test/rendered-evidence.json``.

Run from the GAMMAF directory:
    python tests/placeholder-test/test_placeholders.py
"""

import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from DebateAgent import DebateAgent  # noqa: E402
from DebateConfigLoader import DebateConfig  # noqa: E402
from DatasetManager import ResponseFormat  # noqa: E402
from Utils import AttrDict  # noqa: E402

PROMPTS_FILE = HERE / "prompts" / "prompts-placeholder-test.json"
EVIDENCE_FILE = HERE / "rendered-evidence.json"

# Small directed topology: adj[receiver][sender] == 1.
ADJACENCY = [
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
]
NUM_AGENTS = 3
NUM_MALICIOUS = 1
MAX_ROUNDS = 2
MALICIOUS_SEED = 7

QUESTION = {
    "question": "Which planet is known as the Red Planet?",
    "choices": "A. Mars\nB. Venus\nC. Jupiter\nD. Saturn",
    "answer": "A",
}


def load_module(name, filename):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Stubs (kept intentionally tiny: this test validates prompt plumbing, not the
# LLM's reasoning; the LLM calls themselves are real)
# ---------------------------------------------------------------------------

class StubDataloader:
    SUPPORTS_TOOL_CALLS = False
    TAG = "TEST"

    def __init__(self, prompts):
        self.prompts = prompts

    def get_prompts(self):
        return self.prompts

    def parse_model_output(self, message):
        text = message.content or ""
        message_match = re.search(
            r"<message>\s*:?\s*(.*?)(?=<answer>|\Z)", text, re.DOTALL | re.IGNORECASE
        )
        answer_match = re.search(r"<answer>\s*:?\s*([A-E])", text, re.IGNORECASE)
        message_text = message_match.group(1).strip() if message_match else text
        answer = answer_match.group(1).upper() if answer_match else "A"
        return ResponseFormat(message=message_text or text, answer=answer)

    def is_answer_correct(self, round_responses, correct_answer):
        if not round_responses:
            return False
        return round_responses[0]["answer"].upper() == str(correct_answer).upper()


class StubTextProcessor:
    def process_round(self, round_responses):
        return [str(resp.get("message", "")) for resp in round_responses]


class ScriptedDefenseModel:
    """Deterministic stand-in for a trained defense model."""

    def __init__(self, flagged_indexes):
        self.flagged_indexes = set(flagged_indexes)
        self.config = SimpleNamespace(
            top_k=len(self.flagged_indexes) or 1, threshold=0.5
        )

    def predict(self, debate_embeddings, adjacency_matrix):
        size = len(adjacency_matrix)
        flags = [1 if i in self.flagged_indexes else 0 for i in range(size)]
        anomaly_scores = [1.0 if flag else 0.0 for flag in flags]
        return flags, anomaly_scores


class RecordingAgent(DebateAgent):
    """DebateAgent subclass that records format_data and renders the prompts."""

    records = []

    def _record(self, phase, format_data):
        # Mirror production: first_round_generate renders system+first-round,
        # debate_round_generate renders the debate prompt only.
        templates = {
            "first_round": (
                ("system_prompt", self.system_prompt),
                ("first_round_prompt", self.first_round_prompt),
            ),
            "debate_round": (
                ("debate_prompt", self.debate_prompt),
            ),
        }[phase]
        rendered = {}
        for name, template in templates:
            try:
                rendered[name] = template.format(**format_data)
            except Exception as exc:  # missing-key check
                rendered[name] = f"<FORMAT ERROR: {exc!r}>"
        self.records.append({
            "phase": phase,
            "round_num": format_data.get("round_num"),
            "agent_id": self.agent_id,
            "is_malicious": self.is_malicious,
            "topology_string": format_data.get("topology_string"),
            "malicious_agents_string": format_data.get("malicious_agents_string"),
            "flags_string": format_data.get("flags_string"),
            "rendered": rendered,
        })

    def first_round_generate(self, format_data):
        self._record("first_round", format_data)
        return super().first_round_generate(format_data)

    def debate_round_generate(self, format_data):
        self._record("debate_round", format_data)
        return super().debate_round_generate(format_data)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def parse_topology_string(text):
    parsed = {}
    pattern = (
        r"Agent (\d+): receives messages from (\[[^\]]*\]); "
        r"sends messages to (\[[^\]]*\])"
    )
    for match in re.finditer(pattern, text or ""):
        parsed[int(match.group(1))] = (
            json.loads(match.group(2)),
            json.loads(match.group(3)),
        )
    return parsed


def expected_topology(adjacency):
    size = len(adjacency)
    return {
        i: (
            [j for j in range(size) if j != i and adjacency[i][j] == 1],
            [j for j in range(size) if j != i and adjacency[j][i] == 1],
        )
        for i in range(size)
    }


def index_list_string(indexes):
    return "[" + ", ".join(str(i) for i in sorted(indexes)) + "]"


def check_records(label, records, expected_malicious, expected_flags_by_phase,
                  topology_by_phase):
    errors = []
    if not records:
        errors.append(f"{label}: no records captured")
        return errors

    for record in records:
        where = f"{label} {record['phase']} agent {record['agent_id']}"
        rendered_blob = json.dumps(record["rendered"])
        if "<FORMAT ERROR" in rendered_blob:
            errors.append(f"{where}: {rendered_blob}")
            continue
        for placeholder in ("topology_string", "malicious_agents_string", "flags_string"):
            if "{" + placeholder + "}" in rendered_blob:
                errors.append(f"{where}: {{{placeholder}}} was not substituted")

        if record["malicious_agents_string"] != index_list_string(expected_malicious):
            errors.append(
                f"{where}: malicious_agents_string="
                f"{record['malicious_agents_string']!r} expected "
                f"{index_list_string(expected_malicious)!r}"
            )

        expected_flags = expected_flags_by_phase[record["phase"]]
        if record["flags_string"] != index_list_string(expected_flags):
            errors.append(
                f"{where}: flags_string={record['flags_string']!r} expected "
                f"{index_list_string(expected_flags)!r}"
            )

        parsed = parse_topology_string(record["topology_string"])
        expected = expected_topology(topology_by_phase[record["phase"]])
        if parsed != expected:
            errors.append(
                f"{where}: topology_string does not match the "
                f"{record['phase']} adjacency\n  parsed={parsed}\n  expected={expected}"
            )
    return errors


def check_additivity():
    """Production prompts render identically with and without the new keys.

    Uses the real (read-only) production prompt files and a format_data dict
    holding exactly the keys the original loops provided.
    """
    errors = []
    base_data = {
        "agent_id": 0,
        "question": "Which planet is known as the Red Planet?",
        "choices": "A. Mars\nB. Venus\nC. Jupiter\nD. Saturn",
        "neighbors_messages": "Agent 1\nResponse: B\nArgument: Venus is hotter.\n",
        "round_num": 2,
        "wrong_answer": "B",
        "malicious_indexes": [1],
        "topology": ADJACENCY,
    }
    extended_data = dict(
        base_data,
        topology_string="Network topology for this step: ...",
        malicious_agents_string="[1]",
        flags_string="[0]",
    )
    for name in ("prompts_gsm8k.json", "prompts_blindguard.json", "prompts_msmarco.json"):
        prompts = json.loads((ROOT / "prompts" / name).read_text(encoding="utf-8"))
        for prompt_name, template in prompts.items():
            if not isinstance(template, str):
                continue
            try:
                before = template.format(**base_data)
                after = template.format(**extended_data)
            except KeyError as exc:
                # Prompts requiring dataset-specific keys are outside this check.
                print(f"  (skip {name}:{prompt_name}: missing original key {exc})")
                continue
            if before != after:
                errors.append(
                    f"additivity: {name}:{prompt_name} changed when the new keys were added"
                )
    return errors


def build_live_config():
    return AttrDict({
        "timeout": 120,
        "llm_max_retries": 2,
        "num_agents": NUM_AGENTS,
        "num_malicious_agents": NUM_MALICIOUS,
        "malicious_seed": MALICIOUS_SEED,
        "python_seed": 11,
        "numpy_seed": 11,
        "answer_seed": 11,
        "max_rounds": MAX_ROUNDS,
        "consensus_threshold": 1.0,
        "no_consensus_check": True,
        "check_consensus_only_unflagged": True,
        "top_k_defense": 1,
        "no_defense_baseline": False,
        "max_concurrent_inference": NUM_AGENTS,
        "num_questions": 1,
        "n_questions_on_random_topo": 1,
        "new_random_each_question": False,
        "topologies_seed": 5,
        "density_range_for_random_topo": [0.3, 0.7],
    })


def main():
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded test prompts: {PROMPTS_FILE}")

    # The evaluation loop picks malicious agents with random.Random(seed + qidx).
    eval_malicious = sorted(
        random.Random(MALICIOUS_SEED).sample(range(NUM_AGENTS), NUM_MALICIOUS)
    )
    print(f"Evaluation-loop malicious agent indexes: {eval_malicious}")

    edl = load_module("EvaluationDebateLoop_complete", "EvaluationDebateLoop-complete.py")

    # --- 1. Evaluation debate (with scripted defense) ----------------------
    RecordingAgent.records = []
    orchestrator = edl.LiveDebateOrchestration(
        build_live_config(),
        dataloader=StubDataloader(prompts),
        text_processor=StubTextProcessor(),
        dataset_tag="TEST",
        loader_tag="TEST",
    )
    orchestrator._agent_class = RecordingAgent
    defense_model = ScriptedDefenseModel(eval_malicious)
    edl_traces = orchestrator.run_debate_with_defense(
        [dict(QUESTION)], defense_model, {"test-topology": ADJACENCY}
    )
    eval_records = list(RecordingAgent.records)
    print(f"Evaluation debate finished: {len(eval_records)} agent turns recorded")

    eval_flags = [1 if i in eval_malicious else 0 for i in range(NUM_AGENTS)]
    eval_round2_adjacency = edl.modify_adjacency(eval_flags, ADJACENCY)

    # --- 2. Generation debate (no defense) ---------------------------------
    dgdl = edl._DGDL
    RecordingAgent.records = []
    gen_config = DebateConfig(
        timeout=120,
        is_random_topology=False,
        max_rounds=MAX_ROUNDS,
        number_of_agents=NUM_AGENTS,
        number_malicious_agents=NUM_MALICIOUS,
        consensus_threshold=1.0,
        topology=ADJACENCY,
        llm_max_retries=2,
        malicious_randomization_seed=MALICIOUS_SEED,
        parallel_questions=1,
        verbose=False,
        num_questions=1,
        questions_random_seed=11,
        dataset_tag="TEST",
    )
    gen_orchestrator = dgdl.DebateOrchestration(gen_config)
    gen_orchestrator.prompts = prompts
    gen_orchestrator.dataloader = StubDataloader(prompts)
    gen_orchestrator._agent_class = RecordingAgent
    _, gen_malicious, _ = gen_orchestrator.debate_question(
        QUESTION["question"],
        QUESTION["choices"],
        mal_answer="B",
        question_index=0,
    )
    gen_records = list(RecordingAgent.records)
    print(f"Generation debate finished: {len(gen_records)} agent turns recorded")

    # --- 3. Checks ----------------------------------------------------------
    errors = []
    errors += check_records(
        "evaluation",
        eval_records,
        eval_malicious,
        {"first_round": [], "debate_round": eval_malicious},
        {"first_round": ADJACENCY, "debate_round": eval_round2_adjacency},
    )
    errors += check_records(
        "generation",
        gen_records,
        gen_malicious,
        {"first_round": [], "debate_round": []},
        {"first_round": ADJACENCY, "debate_round": ADJACENCY},
    )
    additivity_errors = check_additivity()
    errors += additivity_errors

    evidence = {
        "test_file": str(Path(__file__).resolve()),
        "prompts_file": str(PROMPTS_FILE),
        "adjacency": ADJACENCY,
        "malicious_seed": MALICIOUS_SEED,
        "evaluation_loop": {
            "malicious_indexes": eval_malicious,
            "round1_adjacency": ADJACENCY,
            "round2_adjacency": eval_round2_adjacency,
            "records": eval_records,
        },
        "generation_loop": {
            "malicious_indexes": gen_malicious,
            "records": gen_records,
        },
        "additivity_check_errors": additivity_errors,
        "check_errors": errors,
    }
    EVIDENCE_FILE.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Evidence written: {EVIDENCE_FILE}")

    if errors:
        print("\nPLACEHOLDER VALIDATION: FAIL")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nPLACEHOLDER VALIDATION: PASS")
    print("\n--- Example: evaluation round 1 (agent 0, flags empty) ---")
    sample = next(r for r in eval_records if r["agent_id"] == 0 and r["phase"] == "first_round")
    print(sample["rendered"]["first_round_prompt"])
    print("\n--- Example: evaluation round 2 (agent 0, flags populated) ---")
    sample = next(r for r in eval_records if r["agent_id"] == 0 and r["phase"] == "debate_round")
    print(sample["rendered"]["debate_prompt"])


if __name__ == "__main__":
    main()
