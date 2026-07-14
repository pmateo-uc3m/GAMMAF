from datasets import load_dataset
from typing import List
import numpy as np
from langchain_core.messages import AIMessage
import re

from DebateAgent import ResponseFormat


def extract_reason_answer(text: str):
    # Allow both <tag>: value and <tag> value, and any order between tags.
    reason_match = re.search(r'<reason>\s*:?\s*(.*?)(?=<answer>\s*:?\s*|\Z)', text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>\s*:?\s*(.*?)(?=<reason>\s*:?\s*|\Z)', text, re.DOTALL | re.IGNORECASE)

    reason = reason_match.group(1).strip() if reason_match else text
    answer = answer_match.group(1).strip() if answer_match else ""
    return reason, answer

class MMLULoader:
    TAG = "MMLU"
    PROMPTS_FILE = "prompts/prompts_blindguard.json"
    def __init__(self, num_questions: int = 25, random_seed: int = 23):

        self.num_questions = num_questions
        self.random_seed = random_seed
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
            answer = item['answer']
            questions.append({
                'question': question_text,
                'choices': choices,
                'answer': answer
            })
        return np.random.default_rng(self.random_seed).choice(questions, size=self.num_questions, replace=False)
    
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
    
    def is_answer_correct(self, model_answer, correct_answer) -> bool:
        return model_answer.upper() == correct_answer.upper()

    def agent_is_safe(self, response_data: dict) -> bool:
        return response_data["response"]['answer'].upper() == response_data["correct_answer"].upper()

class CSQALoader(MMLULoader):
    TAG = "CSQA"
    def __init__(self, num_questions: int = 25, random_seed: int = 23):
        self.num_questions = num_questions
        self.random_seed = random_seed
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
        return np.random.default_rng(self.random_seed).choice(questions, size=self.num_questions, replace=False)
    
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
    def __init__(self, num_questions: int = 25, random_seed: int = 23):
        self.num_questions = num_questions
        self.random_seed = random_seed
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
        return np.random.default_rng(self.random_seed).choice(questions, size=self.num_questions, replace=False)
    
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
    
    def is_answer_correct(self, model_answer, correct_answer) -> bool:
        return model_answer == correct_answer

    def agent_is_safe(self, response_data: dict) -> bool:
        return response_data["response"]['answer'] == response_data["correct_answer"]
    
class MMLUProLoader(MMLULoader):
    TAG = "MMLUPro"
    def __init__(self, num_questions: int = 25, random_seed: int = 23):
        self.num_questions = num_questions
        self.random_seed = random_seed
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
        return np.random.default_rng(self.random_seed).choice(questions, size=self.num_questions, replace=False)
    
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

    def __init__(self, num_questions: int = 25, random_seed: int = 23):
        from Utils import AnomalyJudgeLLM
        import json
        self.num_questions = num_questions
        self.random_seed = random_seed
        self.dataset = self._load_json()
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()
        judge_prompts = json.load(open("MA/judge_prompts.json", "r"))
        self.judge = AnomalyJudgeLLM(
            model_info=json.load(open("MA/judge_model_info.json", "r")),
            system_prompt=judge_prompts["SYSTEM_PROMPT"],
            judge_prompt=judge_prompts["JUDGE_PROMPT"]
        )

    def _load_json(self):
        import json
        with open("MA/msmarco.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Expected MSMARCO dataset to be a JSON object")
        return data

    def load_questions(self):
        questions = []
        for key, item in self.dataset.items():
            if not isinstance(item, dict):
                continue
            question_text = item.get("question")
            correct_answer = item.get("correct answer")
            incorrect_answer = item.get("incorrect answer")
            adv_texts = item.get("adv_texts")
            if question_text is None or correct_answer is None:
                continue
            questions.append({
                "question": question_text,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "adv_texts": adv_texts if isinstance(adv_texts, list) else [],
            })
        if not questions:
            raise ValueError("No valid questions found in MSMARCO dataset")
        rng = np.random.default_rng(self.random_seed)
        n = min(self.num_questions, len(questions))
        return list(rng.choice(questions, size=n, replace=False))

    def format_questions(self) -> List[dict]:
        formatted = []
        for i, q in enumerate(self.questions):
            formatted.append({
                "question_index": i,
                "question": q["question"],
                "adv_texts": q["adv_texts"],
                "correct_answer": q["correct_answer"],
                "incorrect_answer": q["incorrect_answer"],
            })
        return formatted

    def parse_model_output(self, message: AIMessage) -> ResponseFormat:
        text = message.content
        if not text:
            raise ValueError("Empty response from model")
        reason, answer = extract_reason_answer(text)
        return ResponseFormat(reason=reason, answer=answer)

    def is_answer_correct(self, model_answer, correct_answer) -> bool:
        return model_answer.strip().lower() == correct_answer.strip().lower()

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