from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

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
                # print("[DEBUG] Response from agent:\n\n", self.current_response)
                formatted_response = f"<answer>: {self.current_response.answer} \n<reason>: {self.current_response.reason}"
                self.messages.append(AIMessage(content=formatted_response))
                return self.current_response
            except Exception as e:
                last_exception = e
                is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
                if not is_timeout:
                    raise e  # For non-timeout errors, we want to raise immediately instead of retrying
        raise last_exception
    
    def debate_round_generate(self, format_data: dict):
        prompt = self.debate_prompt.format(**format_data)
        self.messages.append(HumanMessage(content=prompt))
        last_exception = None
        for i in range(self.max_retries):
            try:
                self.current_response = self.model.invoke(self.messages)
                # print("[DEBUG] Response from agent:\n\n", self.current_response)
                formatted_response = f"<answer>: {self.current_response.answer} \n<reason>: {self.current_response.reason}"
                self.messages.append(AIMessage(content=formatted_response))
                return self.current_response
            except Exception as e:
                last_exception = e
                is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
                if not is_timeout:
                    raise e
        raise last_exception
    
    def get_current_response(self) -> Optional[ResponseFormat]:
        return self.current_response
    
    def get_formatted_response(self) -> Optional[str]:
        if self.current_response:
            return f"Agent {self.agent_id} Response: \n<answer>: {self.current_response.answer} \n<reason>: {self.current_response.reason}"
        return None