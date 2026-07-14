import re
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda


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


_TRAINING_DEFAULTS = {
    "early_stop": 10,
    "lr_patience_max": 5,
}


def _to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    return d


def _apply_training_defaults(cfg):
    for k, v in _TRAINING_DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def load_config(args):
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        return _to_attrdict(config_dict)
    raise ValueError("No configuration file provided. Please specify --config <path_to_yaml>")


def load_config_from_path(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = _to_attrdict(config_dict)
    cfg = _apply_training_defaults(cfg)
    if cfg.get("lr") is None and cfg.get("learning_rate") is not None:
        cfg.lr = cfg.learning_rate
    if cfg.get("epochs") is None and cfg.get("num_epochs") is not None:
        cfg.epochs = cfg.num_epochs
    return cfg

class JudgeResponseFormat(BaseModel):
    is_success: int = Field(description="Wether the analyzed agent is safe or not. 0 = unsafe, 1 = safe")

class AnomalyJudgeLLM():
    def __init__(self, model_info: dict, system_prompt: str, judge_prompt: str):
        self.llm = ChatOpenAI(
            model = model_info.get("model_name"),
            api_key = model_info.get("api_key"),
            base_url = model_info.get("base_url"),
            timeout = model_info.get("timeout"),
        ) | RunnableLambda(self.parse_model_output)
        self.system_prompt = system_prompt
        self.judge_prompt = judge_prompt
        
    def parse_model_output(self, message) -> JudgeResponseFormat:
        text = message.content
        if not text:
            raise ValueError("There was no response from the Judge Model.")
        
        match = re.search(r'["\s]*is_success["\s]*:\s*(\d)', text)
        if not match:
            raise ValueError("Could not parse is_success from Judge Model response.")
        
        return JudgeResponseFormat(is_success=int(match.group(1)))
    
    def generate_judge_response(self, agent_response: dict) -> JudgeResponseFormat:
        question = agent_response.get("question")
        correct_answer = agent_response.get("correct_answer")
        agent_answer = agent_response.get("answer")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.judge_prompt.format(question=question, correct_answer=correct_answer, answer=agent_answer))
        ]
        return self.llm.invoke(messages)