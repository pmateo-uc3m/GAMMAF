import time
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from LoggingUtils import log_warn, log_error

class ResponseFormat(BaseModel):
    reason: str = Field(description="The logical argument or premise")
    answer: str = Field(description="The detailed response or conclusion")

class DebateAgent:
    def __init__(self, 
                 agent_id: int, 
                 model,
                 system_prompt: str, 
                 first_round_prompt: str,
                 debate_prompt: str,
                 max_retries: int =3,
                 is_malicious: bool = False,
                 ):
        
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt
        self.first_round_prompt = first_round_prompt
        self.debate_prompt = debate_prompt
        self.messages: List[BaseMessage] = []
        self.current_response: Optional[ResponseFormat] = None
        self.max_retries = max_retries
        self.is_malicious = is_malicious
        
    def first_round_generate(self, format_data: dict):
        prompt = self.first_round_prompt.format(**format_data)
        sys_prompt = self.system_prompt.format(**format_data)
        self.messages.append(SystemMessage(content=sys_prompt))
        self.messages.append(HumanMessage(content=prompt))
        last_exception = None
        for i in range(self.max_retries):
            try:
                self.current_response = self.model.invoke(self.messages)
                formatted_response = f"<answer>: {self.current_response.answer} \n<reason>: {self.current_response.reason}"
                self.messages.append(AIMessage(content=formatted_response))
                return self.current_response
            except Exception as e:
                last_exception = e
                log_warn(f"Agent {self.agent_id} first_round attempt {i + 1}/{self.max_retries} failed: {e}")
                if i < self.max_retries - 1:
                    time.sleep(1)
        log_error(f"Agent {self.agent_id} all {self.max_retries} first_round attempts failed.")
        raise last_exception
    
    def debate_round_generate(self, format_data: dict):
        prompt = self.debate_prompt.format(**format_data)
        self.messages.append(HumanMessage(content=prompt))
        last_exception = None
        for i in range(self.max_retries):
            try:
                self.current_response = self.model.invoke(self.messages)
                formatted_response = f"<answer>: {self.current_response.answer} \n<reason>: {self.current_response.reason}"
                self.messages.append(AIMessage(content=formatted_response))
                return self.current_response
            except Exception as e:
                last_exception = e
                log_warn(f"Agent {self.agent_id} debate_round attempt {i + 1}/{self.max_retries} failed: {e}")
                if i < self.max_retries - 1:
                    time.sleep(1)
        log_error(f"Agent {self.agent_id} all {self.max_retries} debate_round attempts failed.")
        raise last_exception
    
    def get_current_response(self) -> Optional[ResponseFormat]:
        return self.current_response
    
    def get_formatted_response(self) -> Optional[str]:
        if self.current_response:
            return f"Agent {self.agent_id} Response: \n<answer>: {self.current_response.answer} \n<reason>: {self.current_response.reason}"
        return None