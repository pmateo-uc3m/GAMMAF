import time
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from DatasetManager import TAResponseFormat
from LoggingUtils import log_error


class TAAgent:
    """Tool-call aware debate agent used by InjecAgent-style datasets.

    The first-round flow mirrors ``DebateAgent`` (system prompt + first-round
    prompt as the predefined setup messages), then adds a tool-call round-trip:
    whenever the model returns a tool call with empty content, the tool-call
    message is retained, a synthetic successful ``ToolMessage`` is appended,
    and the agent is re-invoked. The final content becomes ``message`` (what is
    passed to neighbours and the defense model), the called tool becomes
    ``answer``/``called_tool``, and ``<tool_call>: ..., <message>: ...`` is the
    trace entry recorded in the debate trace.
    """

    def __init__(self,
                 agent_id: int,
                 model,
                 system_prompt: str,
                 first_round_prompt: str,
                 debate_prompt: str,
                 max_retries: int = 3,
                 is_malicious: bool = False,
                 max_tool_call_follow_ups: int = 3,
                 ):
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt
        self.first_round_prompt = first_round_prompt
        self.debate_prompt = debate_prompt
        self.messages: List[BaseMessage] = []
        self.current_response: Optional[Any] = None
        self.max_retries = max_retries
        self.is_malicious = is_malicious
        self.max_tool_call_follow_ups = max_tool_call_follow_ups
        self._tools_bound = False

    def _prepare_format_data(self, format_data: dict) -> dict:
        format_data = dict(format_data)
        if self.is_malicious:
            format_data["tool_response"] = format_data.get("anomalous_response", "")
        else:
            format_data["tool_response"] = format_data.get("benign_response", "")
        return format_data

    def _ensure_tools_bound(self, format_data: dict) -> None:
        tools = format_data.get("tools_description")
        if tools and not self._tools_bound:
            self.model = self.model.bind_tools(tools)
            self._tools_bound = True

    def _invoke_with_tool_round_trip(self):
        response = self.model.invoke(self.messages)
        tool_calls_made = 0
        called_tool = ""
        final_appended = False
        while getattr(response, "tool_calls", None) and not str(response.content or "").strip():
            self.messages.append(response)
            final_appended = True
            tool_calls_made += 1
            tc = response.tool_calls[0]
            if isinstance(tc, dict):
                tool_name = tc.get("name", "")
                tool_call_id = tc.get("id", "")
            else:
                tool_name = getattr(tc, "name", "")
                tool_call_id = getattr(tc, "id", "")
            if not called_tool:
                called_tool = tool_name
            self.messages.append(ToolMessage(
                content="Tool call executed successfully.",
                tool_call_id=tool_call_id or f"call_{tool_calls_made}",
            ))
            if tool_calls_made >= self.max_tool_call_follow_ups:
                break
            response = self.model.invoke(self.messages)
            final_appended = False
        if not final_appended:
            self.messages.append(response)
        content = str(response.content or "").strip()
        return content, called_tool, tool_calls_made

    def _generate(self, format_data: dict):
        self._ensure_tools_bound(format_data)
        last_exception = None
        for i in range(self.max_retries):
            snapshot = len(self.messages)
            try:
                content, called_tool, _ = self._invoke_with_tool_round_trip()
                if called_tool:
                    trace = f"<tool_call>: {called_tool}, <message>: {content}"
                else:
                    trace = f"<message>: {content}"
                self.current_response = TAResponseFormat(
                    message=content,
                    answer=called_tool,
                    called_tool=called_tool,
                    trace=trace,
                )
                return self.current_response
            except Exception as e:
                last_exception = e
                del self.messages[snapshot:]
                if i == self.max_retries - 1:
                    log_error(f"Agent {self.agent_id} all {self.max_retries} attempts failed: {e}")
                else:
                    time.sleep(1)
        raise last_exception

    def first_round_generate(self, format_data: dict):
        format_data = self._prepare_format_data(format_data)
        sys_prompt = self.system_prompt.format(**format_data)
        prompt = self.first_round_prompt.format(**format_data)
        self.messages.append(SystemMessage(content=sys_prompt))
        self.messages.append(HumanMessage(content=prompt))
        return self._generate(format_data)

    def debate_round_generate(self, format_data: dict):
        format_data = self._prepare_format_data(format_data)
        prompt = self.debate_prompt.format(**format_data)
        self.messages.append(HumanMessage(content=prompt))
        return self._generate(format_data)

    def get_current_response(self) -> Optional[Any]:
        return self.current_response

    def get_formatted_response(self) -> Optional[str]:
        if self.current_response:
            return f"Agent {self.agent_id} Response: \n{self.current_response.to_message_content()}"
        return None


# Backwards-compatible alias: this file previously defined ``DebateAgent``.
DebateAgent = TAAgent