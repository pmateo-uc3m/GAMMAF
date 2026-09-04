from datasets import load_dataset
from typing import List
import numpy as np
from langchain_core.messages import AIMessage
import re
import inspect
import os

from DebateAgent import ResponseFormat


def extract_reason_answer(text: str):
    # Allow both <tag>: value and <tag> value, and any order between tags.
    reason_match = re.search(r'<reason>\s*:?\s*(.*?)(?=<answer>\s*:?\s*|\Z)', text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>\s*:?\s*(.*?)(?=<reason>\s*:?\s*|\Z)', text, re.DOTALL | re.IGNORECASE)

    reason = reason_match.group(1).strip() if reason_match else text
    answer = answer_match.group(1).strip() if answer_match else ""
    return reason, answer


def _select_evaluation_indexes(available_indexes, num_questions, rng):
    """
    Sample *num_questions* indexes without replacement from *available_indexes*.

    Raises a clear error when the exclusion of training/HPS indexes leaves fewer
    available tasks than requested, so an excluded task is never silently
    selected to satisfy the sample size.  The sampling itself is unchanged.
    """
    if len(available_indexes) < num_questions:
        raise ValueError(
            f"Not enough available tasks for evaluation: requested "
            f"{num_questions} question(s), but only {len(available_indexes)} "
            f"remain after excluding the configured (training/HPS) indexes."
        )
    return rng.choice(available_indexes, size=num_questions, replace=False)


def make_loader_kwargs(loader_cls, config=None, **base):
    """Build kwargs for a questions loader.

    Threads ``ma_dataset_path`` from ``config`` into the loader arguments when
    the loader class supports it (currently ``MSMARCOLoader``).  Loaders are
    selected by tag and may be loaded as separate module instances, so the
    capability is detected via the constructor signature rather than class
    identity.
    """
    kwargs = dict(base)
    if config is not None and "dataset_path" in inspect.signature(
        loader_cls.__init__
    ).parameters:
        ma_path = getattr(config, "ma_dataset_path", None)
        if ma_path:
            kwargs["dataset_path"] = ma_path
    return kwargs

class MMLULoader:
    TAG = "MMLU"
    PROMPTS_FILE = "prompts/prompts_blindguard.json"
    def __init__(self, num_questions: int = 25, random_seed: int = 23, indexes = []):

        self.num_questions = num_questions
        self.random_seed = random_seed
        self.indexes = indexes
        self.dataset = load_dataset("cais/mmlu", "all", split="all")
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()


    def get_prompts(self):
        import json
        with open(self.PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def load_questions(self):
        questions = []
        for item in self.dataset:
            question_text = item['question']
            choices = item['choices']
            answer = chr(ord('A') + item['answer'])
            questions.append({
                'question': question_text,
                'choices': choices,
                'answer': answer
            })
        rng = np.random.default_rng(self.random_seed)
        available_indexes = [
            i for i in range(len(questions)) if i not in self.indexes
        ]
        selected_indexes = _select_evaluation_indexes(
            available_indexes, self.num_questions, rng
        )
        
        self.indexes = selected_indexes
        
        return [questions[i] for i in selected_indexes]

    def format_questions(self) -> List[dict]:
        formatted_questions = []
        for i, q in enumerate(self.questions):
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])])
            formatted_questions.append({
                'question_index': i,
                'question': q['question'],
                'choices': choices_str,
                'answer': q['answer']
            })
        return formatted_questions
    
    def get_questions(self) -> List[dict]:
        return self.questions
    
    def get_formatted_questions(self) -> List[dict]:
        return self.formatted_questions
    
    def parse_model_output(self, message: AIMessage) -> ResponseFormat:
        text = message.content
        if not text:
            raise ValueError("Empty response from model")

        reason, answer = extract_reason_answer(text)
        
        # Fallback: if answer is empty, maybe the model just outputted the answer letter?
        if not answer and len(text) < 10 and text.strip().upper() in ['A', 'B', 'C', 'D', 'E']:
            answer = text.strip().upper()
            reason = "No reasoning provided."

        return ResponseFormat(reason=reason, answer=answer)
    
    def is_answer_correct(self, round_responses: list, correct_answer) -> bool:
        counts = {}
        for r in round_responses:
            ans = r["answer"]
            counts[ans] = counts.get(ans, 0) + 1
        majority_answer = max(counts, key=counts.get)
        return majority_answer.upper() == correct_answer.upper()

    def agent_is_safe(self, response_data: dict) -> bool:
        return response_data["response"]['answer'].upper() == response_data["correct_answer"].upper()

class CSQALoader(MMLULoader):
    TAG = "CSQA"
    def __init__(self, num_questions: int = 25, random_seed: int = 23, indexes = []):
        self.num_questions = num_questions
        self.random_seed = random_seed
        self.indexes = indexes
        self.dataset = load_dataset("tau/commonsense_qa", split="train")
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()
        
    def load_questions(self):
        questions = []
        for item in self.dataset:
            question_text = item['question']
            choices = item['choices']
            answer = item['answerKey']
            questions.append({
                'question': question_text,
                'choices': choices,
                'answer': answer
            })
        rng = np.random.default_rng(self.random_seed)
        available_indexes = [
                    i for i in range(len(questions)) if i not in self.indexes
                ]
        selected_indexes = _select_evaluation_indexes(
            available_indexes, self.num_questions, rng
        )
                
        self.indexes = selected_indexes
        
        return [questions[i] for i in selected_indexes]
    
    def format_questions(self) -> List[dict]:
        formatted_questions = []
        for i, q in enumerate(self.questions):
            choices_str = "\n".join([f"{q['choices']['label'][j]}. {q['choices']['text'][j]}" for j in range(0,5)])
            formatted_questions.append({
                'question_index': i,
                'question': q['question'],
                'choices': choices_str,
                'answer': q['answer']
            })
        return formatted_questions
    
    def parse_model_output(self, message: AIMessage) -> ResponseFormat:
        text = message.content
        if not text:
            # Log this case?
            # print(f"[DEBUG] Received empty content from model. Full message: {message}")
            # Return empty ResponseFormat or raise to retry. 
            # Raising matches existing behavior of erroring out but now with clear message.
            raise ValueError("Empty response from model")
            
        reason, answer = extract_reason_answer(text)
        
        # Fallback: if answer is empty, maybe the model just outputted the answer letter?
        if not answer and len(text) < 10 and text.strip().upper() in ['A', 'B', 'C', 'D', 'E']:
            answer = text.strip().upper()
            reason = "No reasoning provided."

        return ResponseFormat(reason=reason, answer=answer)
    
class GSM8KLoader(MMLULoader):
    TAG = "GSM8K"
    PROMPTS_FILE = "prompts/prompts_gsm8k.json"
    def __init__(self, num_questions: int = 25, random_seed: int = 23, indexes = []):
        self.num_questions = num_questions
        self.random_seed = random_seed
        self.indexes = indexes
        self.dataset = load_dataset("openai/gsm8k", 'main', split="train")
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()
        
    def load_questions(self):
        import re
        questions = []
        for item in self.dataset:
            question_text = item['question']
            choices = None
            answer = re.search(r'#### (.*)', item['answer']).group(1).strip()
            questions.append({
                'question': question_text,
                'choices': choices,
                'answer': answer
            })
        rng = np.random.default_rng(self.random_seed)
        available_indexes = [
            i for i in range(len(questions)) if i not in self.indexes
        ]
        selected_indexes = _select_evaluation_indexes(
            available_indexes, self.num_questions, rng
        )
        
        self.indexes = selected_indexes
        
        return [questions[i] for i in selected_indexes]
    
    def format_questions(self) -> List[dict]:
        formatted_questions = []
        for i, q in enumerate(self.questions):
            formatted_questions.append({
                'question_index': i,
                'question': q['question'],
                'choices': "N/A",
                'answer': q['answer']
            })
        return formatted_questions
    
    def extract_number(self, response_str):
        match = re.search(r'-?\d+\.?\d*', str(response_str))
        cleaned = match.group(0) if match else response_str
        return cleaned
    
    def parse_model_output(self, message: AIMessage) -> ResponseFormat:
        text = message.content
        if not text:
            # Log this case?
            # print(f"[DEBUG] Received empty content from model. Full message: {message}")
            # Return empty ResponseFormat or raise to retry. 
            # Raising matches existing behavior of erroring out but now with clear message.
            raise ValueError("Empty response from model")
            
        reason, answer = extract_reason_answer(text)
        
        # Fallback: if answer is empty, maybe the model just outputted the answer letter?
        if not answer and len(text) < 10 and text.strip().upper() in ['A', 'B', 'C', 'D', 'E']:
            answer = text.strip().upper()
            reason = "No reasoning provided."
            
        answer = self.extract_number(answer)
        return ResponseFormat(reason=reason, answer=answer)
    
    def is_answer_correct(self, round_responses: list, correct_answer) -> bool:
        counts = {}
        for r in round_responses:
            ans = r["answer"]
            counts[ans] = counts.get(ans, 0) + 1
        majority_answer = max(counts, key=counts.get)
        return majority_answer == correct_answer

    def agent_is_safe(self, response_data: dict) -> bool:
        return response_data["response"]['answer'] == response_data["correct_answer"]
    
class MMLUProLoader(MMLULoader):
    TAG = "MMLUPro"
    def __init__(self, num_questions: int = 25, random_seed: int = 23, indexes = []):
        self.num_questions = num_questions
        self.random_seed = random_seed
        self.indexes = indexes  
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()
        
    def load_questions(self):
        questions = []
        for item in self.dataset:
            question_text = item['question']
            choices = item['options']
            answer = item['answer']
            questions.append({
                'question': question_text,
                'choices': choices,
                'answer': answer
            })
        rng = np.random.default_rng(self.random_seed)
        available_indexes = [
            i for i in range(len(questions)) if i not in self.indexes
        ]
        selected_indexes = _select_evaluation_indexes(
            available_indexes, self.num_questions, rng
        )
        self.indexes = selected_indexes
        return [questions[i] for i in selected_indexes]

    def format_questions(self) -> List[dict]:
        formatted_questions = []
        for i, q in enumerate(self.questions):
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])])
            formatted_questions.append({
                'question_index': i,
                'question': q['question'],
                'choices': choices_str,
                'answer': q['answer']
            })
        return formatted_questions


class MSMARCOLoader(MMLULoader):
    TAG = "MA"
    PROMPTS_FILE = "prompts/prompts_msmarco.json"

    # Old-format dataset produced by the legacy generator (dict keyed by id).
    DEFAULT_DATASET_PATH = "MA/msmarco.json"
    # New-format benchmark produced by MA/Task_generation/main.py (JSON array).
    NEW_DATASET_PATH = "MA/Task_generation/output/msmarco_contaminated_benchmark.json"

    def __init__(self, num_questions: int = 25, random_seed: int = 23, indexes = [], dataset_path: str | None = None):
        from Utils import AnomalyJudgeLLM
        import json
        self.num_questions = num_questions
        self.random_seed = random_seed
        self.indexes = indexes
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.dataset = self._load_json()
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()
        judge_prompts = json.load(open("MA/judge_prompts.json", "r"))
        self.judge = AnomalyJudgeLLM(
            model_info=json.load(open("MA/judge_model_info.json", "r")),
            system_prompt=judge_prompts["SYSTEM_PROMPT"],
            judge_prompt=judge_prompts["JUDGE_PROMPT"],
            answer_correctness_prompt=judge_prompts.get("ANSWER_CORRECTNESS_PROMPT", "")
        )

    def _resolve_dataset_path(self, dataset_path: str | None) -> str:
        """Resolve the dataset JSON path.

        Precedence: explicit ``dataset_path`` argument, ``MA_DATASET_PATH``
        environment variable, then the legacy/new default paths (the first one
        that exists wins, so both formats keep working).
        """
        if dataset_path:
            return dataset_path
        env_path = os.getenv("MA_DATASET_PATH")
        if env_path:
            return env_path
        for candidate in (self.DEFAULT_DATASET_PATH, self.NEW_DATASET_PATH):
            if os.path.exists(candidate):
                return candidate
        return self.DEFAULT_DATASET_PATH

    def _load_json(self):
        import json
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return data
        raise ValueError("Expected MSMARCO dataset to be a JSON object or an array")

    def _normalize_entry(self, key, item):
        """Normalize one raw dataset entry to the internal question schema.

        Supports the legacy format (dict keyed by query id with ``question`` /
        ``correct answer`` / ``incorrect answer`` / ``adv_texts``) and the new
        Task_generation format (``query`` / ``answers`` / ``safe_passages`` /
        ``adv_passages``).  Returns ``None`` for malformed entries.
        """
        if not isinstance(item, dict):
            return None

        if "question" in item:
            question_text = item.get("question")
            correct_answer = item.get("correct answer")
            incorrect_answer = item.get("incorrect answer")
            adv_texts = item.get("adv_texts")
            if question_text is None or correct_answer is None:
                return None
            return {
                "query_id": item.get("id", key),
                "question": question_text,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "adv_texts": adv_texts if isinstance(adv_texts, list) else [],
                "safe_texts": [],
            }

        if "query" in item:
            query_text = item.get("query")
            answers = item.get("answers")
            adv_passages = item.get("adv_passages")
            if query_text is None or answers is None:
                return None
            if isinstance(answers, list) and answers:
                correct_answer = ", ".join(str(a) for a in answers)
            else:
                correct_answer = str(answers)
            if not correct_answer.strip():
                return None
            return {
                "query_id": item.get("query_id", key),
                "question": query_text,
                "correct_answer": correct_answer,
                "incorrect_answer": "",
                "adv_texts": adv_passages if isinstance(adv_passages, list) else [],
                "safe_texts": item.get("safe_passages") if isinstance(item.get("safe_passages"), list) else [],
            }

        return None

    def load_questions(self):
        questions = []
        if isinstance(self.dataset, dict):
            raw_items = self.dataset.items()
        else:
            raw_items = ((str(i), item) for i, item in enumerate(self.dataset))
        for key, item in raw_items:
            entry = self._normalize_entry(key, item)
            if entry is None:
                continue
            questions.append(entry)
        if not questions:
            raise ValueError("No valid questions found in MSMARCO dataset")
        available_indexes = [
            i for i in range(len(questions)) if i not in self.indexes
        ]
        rng = np.random.default_rng(self.random_seed)
        n = min(self.num_questions, len(questions))
        selected_indexes = _select_evaluation_indexes(available_indexes, n, rng)
        self.indexes = selected_indexes
        return [questions[i] for i in selected_indexes]

    def format_questions(self) -> List[dict]:
        formatted = []
        for i, q in enumerate(self.questions):
            formatted.append({
                "question_index": i,
                "question": q["question"],
                "adv_texts": q["adv_texts"],
                "safe_texts": q["safe_texts"],
                "correct_answer": q["correct_answer"],
                "incorrect_answer": q["incorrect_answer"],
                "query_id": q.get("query_id"),
            })
        return formatted

    def parse_model_output(self, message: AIMessage) -> ResponseFormat:
        text = message.content
        if not text:
            raise ValueError("Empty response from model")
        reason, answer = extract_reason_answer(text)
        return ResponseFormat(reason=reason, answer=answer)

    def is_answer_correct(self, round_responses: list, correct_answer) -> bool:
        try:
            result = self.judge.generate_answer_judge_response(round_responses, correct_answer)
            return bool(result.is_success)
        except Exception as e:
            from LoggingUtils import log_warn
            log_warn(f"Answer correctness judge failed, defaulting to correct: {e}")
            return True

    def agent_is_safe(self, response_data: dict) -> int:
        try:
            agent_response = {
                "question": response_data.get("question", ""),
                "correct_answer": response_data.get("correct_answer", ""),
                "answer": response_data["response"]["answer"],
            }
            agent_evaluation = self.judge.generate_judge_response(agent_response)
            return int(agent_evaluation.is_success)
        except Exception as e:
            from LoggingUtils import log_warn
            log_warn(f"agent_is_safe judge call failed, defaulting to safe=1: {e}")
            return 1