import json
import re
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from LoggingUtils import log_info, log_warn, log_error


def _looks_like_adjacency(value) -> bool:
    """True when ``value`` has the shape of one adjacency matrix (list of int rows)."""
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, int):
                return False
    return True


def load_topologies_file(path, num_agents=None, label: str = "load_topology_file"):
    """Load and validate a topology JSON file.

    The file must contain a non-empty list of mappings with:
      * ``name``     -- unique non-empty string identifying the topology;
      * ``topology`` -- either one square adjacency matrix of 0/1 integers
                        (list of lists) or a non-empty list of such matrices.
                        ``topologies`` is accepted as an alias key.

    When an entry holds several matrices, debates set up under that name sample
    one matrix uniformly using the configured random-topology seed.

    When ``num_agents`` is given, every matrix side must match it.  Returns an
    ordered ``{name: [matrix, ...]}`` mapping (file order preserved); entries
    with a single matrix are stored as a one-element list.
    """
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{label}' ({file_path}): {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"'{label}' ({file_path}) must contain a non-empty JSON list of "
            f"{{'name', 'topology'}} entries"
        )

    topologies = {}
    for index, item in enumerate(data):
        location = f"{label}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{location} must be a mapping with 'name' and 'topology'")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location}.name must be a non-empty string")
        name = name.strip()
        if name in topologies:
            raise ValueError(f"Duplicate topology name '{name}' in '{label}' ({file_path})")

        if "topology" in item and "topologies" in item:
            raise ValueError(
                f"{location} must set only one of 'topology' or 'topologies'"
            )
        if "topology" in item:
            raw = item["topology"]
        elif "topologies" in item:
            raw = item["topologies"]
        else:
            raise ValueError(
                f"{location} must contain 'topology' (an adjacency matrix or a "
                f"non-empty list of adjacency matrices)"
            )

        if _looks_like_adjacency(raw):
            matrices = [raw]
        elif (
            isinstance(raw, list)
            and raw
            and all(_looks_like_adjacency(matrix) for matrix in raw)
        ):
            matrices = list(raw)
        else:
            raise ValueError(
                f"{location}.topology must be an adjacency matrix or a non-empty "
                f"list of adjacency matrices"
            )

        normalized = []
        for matrix_index, matrix in enumerate(matrices):
            size = len(matrix)
            for row_index, row in enumerate(matrix):
                if len(row) != size:
                    raise ValueError(
                        f"{location} topology '{name}'#{matrix_index} is not square: row "
                        f"{row_index} has {len(row)} entries, expected {size}"
                    )
                for value in row:
                    if value not in (0, 1):
                        raise ValueError(
                            f"{location} topology '{name}'#{matrix_index} must contain "
                            f"only binary 0/1 integers"
                        )
            if num_agents is not None and size != int(num_agents):
                raise ValueError(
                    f"{location} topology '{name}'#{matrix_index} has {size} agents but "
                    f"debate.num_agents is {num_agents}"
                )
            normalized.append([[int(value) for value in row] for row in matrix])
        topologies[name] = normalized
    return topologies


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class JudgeResponseFormat(BaseModel):
    is_success: int = Field(description="Wether the analyzed agent is safe or not. 0 = unsafe, 1 = safe")

class AnomalyJudgeLLM():
    def __init__(self, model_info: dict, system_prompt: str, judge_prompt: str, answer_correctness_prompt: str = ""):
        self.llm = ChatOpenAI(
            model = model_info.get("model_name"),
            api_key = model_info.get("api_key"),
            base_url = model_info.get("base_url"),
            timeout = model_info.get("timeout"),
        ) | RunnableLambda(self._parse_model_output)
        self.system_prompt = system_prompt
        self.judge_prompt = judge_prompt
        self.answer_correctness_prompt = answer_correctness_prompt
        self.max_retries = model_info.get("max_retries", 3)

    def _extract_text(self, message) -> str:
        content = message.content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            return "".join(parts)
        if not isinstance(content, str):
            return str(content)
        return content

    def _parse_is_success(self, text: str) -> int | None:
        stripped = text.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", stripped)
        cleaned = re.sub(r"\s*```", "", cleaned)
        match = re.search(r'["\s]*is_success["\s]*:\s*(\d)', cleaned)
        if match:
            val = int(match.group(1))
            if val in (0, 1):
                return val
        return None

    def _parse_model_output(self, message) -> JudgeResponseFormat:
        text = self._extract_text(message)
        if not text or not text.strip():
            raise ValueError("There was no response from the Judge Model.")
        is_success = self._parse_is_success(text)
        if is_success is None:
            raise ValueError("Could not parse is_success from Judge Model response.")
        return JudgeResponseFormat(is_success=is_success)

    def generate_judge_response(self, agent_response: dict) -> JudgeResponseFormat:
        question = agent_response.get("question")
        correct_answer = agent_response.get("correct_answer")
        agent_answer = agent_response.get("answer")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.judge_prompt.format(question=question, correct_answer=correct_answer, answer=agent_answer))
        ]
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        log_error(f"All {self.max_retries} judge inference attempts failed. Defaulting to safe (is_success=1). Last error: {last_error}")
        return JudgeResponseFormat(is_success=1)

    def generate_answer_judge_response(self, agent_responses: list, correct_answer: str) -> JudgeResponseFormat:
        formatted_responses = "\n".join(
            f"Agent {r['agent_id']}: {r['answer']}" for r in agent_responses
        )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.answer_correctness_prompt.format(
                correct_answer=correct_answer, agent_responses=formatted_responses
            ))
        ]
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        log_error(f"All {self.max_retries} answer correctness attempts failed. Defaulting to correct. Last error: {last_error}")
        return JudgeResponseFormat(is_success=1)