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
    def __init__(self, num_questions: int = 25, random_seed: int = 23):

        self.num_questions = num_questions
        self.random_seed = random_seed
        self.dataset = load_dataset("cais/mmlu", "all", split="all")
        self.questions = self.load_questions()
        self.formatted_questions = self.format_questions()
        
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
    
    def validate_answer(self, model_answer, correct_answer) -> bool:
        return chr(correct_answer + 65) == model_answer.upper()
    
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
    
    def validate_answer(self, model_answer, correct_answer) -> bool:
        return model_answer.upper() == correct_answer.upper()
    
class GSM8KLoader(MMLULoader):
    TAG = "GSM8K"
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
    
    def validate_answer(self, model_answer, correct_answer) -> bool:
        return model_answer == correct_answer
    
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
    
    def validate_answer(self, model_answer, correct_answer) -> bool:
        return model_answer.upper() == correct_answer.upper()