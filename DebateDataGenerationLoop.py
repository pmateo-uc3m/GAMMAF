from DebateAgent import DebateAgent
from DebateConfigLoader import DebateConfig
from typing import List
from langchain_openai import ChatOpenAI
from DebateAgent import ResponseFormat, DebateAgent
from dotenv import load_dotenv
from pydantic import SecretStr
import random
import os
import datetime
from collections import Counter

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import json
import numpy as np

import re
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
import DatasetManager
import inspect

load_dotenv()  # Load environment variables from .env file

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def generate_random_topologies(num_agents: int, density: float, rng):
    max_edges = num_agents * (num_agents - 1)
    target_edges = int(density * max_edges)

    if target_edges < num_agents - 1:
        raise ValueError("Density too low for connectivity")

    adj = np.zeros((num_agents, num_agents), dtype=int)

    # Step 1: directed chain (weak connectivity)
    perm = rng.permutation(num_agents)
    for i in range(num_agents - 1):
        adj[perm[i], perm[i + 1]] = 1

    # Step 2: add random edges
    edges = np.argwhere(adj == 0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    rng.shuffle(edges)
    for u, v in edges:
        if adj.sum() >= target_edges:
            break
        adj[u, v] = 1

    return adj.tolist()

# Just an auxiliary function to deal with GSM8K answers containing non-numeric text
def extract_number(response_str):
    # print(f"[DEBUG] Extracting number from response: {response_str}")
    match = re.search(r'-?\d+\.?\d*', str(response_str))
    cleaned = match.group(0) if match else response_str
    # print(f"[DEBUG] Extracted number: {cleaned}")
    return cleaned

def parse_model_output(message: AIMessage) -> ResponseFormat:
    text = message.content
    if not text:
         # Log this case?
        #  print(f"\n[DEBUG] Received empty content from model. Full message: {message}")
         # Return empty ResponseFormat or raise to retry. 
         # Raising matches existing behavior of erroring out but now with clear message.
         raise ValueError("Empty response from model")
         
    # Regex for XML-like format requested in prompts
    reason_match = re.search(r'<reason>:\s*(.*?)(?=\n<answer>:|<answer>:|\Z)', text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    
    reason = reason_match.group(1).strip() if reason_match else text
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Fallback: if answer is empty, maybe the model just outputted the answer letter?
    if not answer and len(text) < 10 and text.strip().upper() in ['A', 'B', 'C', 'D', 'E']:
         answer = text.strip().upper()
         reason = "No reasoning provided."

    return ResponseFormat(reason=reason, answer=answer)

class DebateOrchestration:
    def __init__(self, config: DebateConfig):
        self.config = config
        self.topology = config.topology
        self.random_flag = config.is_random_topology
        with open(config.prompts_file, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        
        model_name = _require_env("MODEL_NAME")
        base_url = _require_env("BASE_URL")
        api_key = SecretStr(_require_env("API_KEY"))

        self.base_llm = ChatOpenAI(
            model = model_name,
            api_key = api_key,
            base_url = base_url,
            timeout = config.timeout,
        )
        self.llm = self.base_llm | RunnableLambda(parse_model_output)
        
    def generate_agents(self, question_index: int = None) -> List[DebateAgent]:
        agents : List[DebateAgent] = []
        mal_idx = np.random.default_rng(
            self.config.malicious_randomization_seed + question_index if question_index is not None else self.config.malicious_randomization_seed
            ).choice(list(range(self.config.number_of_agents)), size=self.config.number_malicious_agents, replace=False)
        for i in range(self.config.number_of_agents):
            if i in mal_idx:
                # Here I must add more control over the agents parameters from the config file (max_retries...)
                agents.append(DebateAgent(
                    agent_id=i,
                    model=self.llm,
                    system_prompt = self.prompts["SYSTEM_PROMPT_MALICIOUS"],
                    first_round_prompt = self.prompts["FIRST_ROUND_PROMPT_MALICIOUS"],
                    debate_prompt = self.prompts["DEBATE_PROMPT_MALICIOUS"],
                    is_malicious=True,
                ))
            else:
                agents.append(DebateAgent(
                    agent_id=i,
                    model=self.llm,
                    system_prompt = self.prompts["SYSTEM_PROMPT"],
                    first_round_prompt = self.prompts["FIRST_ROUND_PROMPT"],
                    debate_prompt = self.prompts["DEBATE_PROMPT"],
                    is_malicious=False,
                ))
                
        return agents
        
    # What will happen if we scale the number of agents so not all can run concurrently?
    def generate_round_1_concurrent(self, question: str, choices: str, agents: List[DebateAgent], mal_answer: str = ""):
                
        def single_agent_round_1(agent: DebateAgent):
            format_data={
                "agent_id" : agent.agent_id,
                "question" : question,
                "choices" : choices,
            }
            if mal_answer:
                format_data['wrong_answer'] = str(mal_answer)

            response = agent.first_round_generate(format_data=format_data)
            
            if self.dataset_name == "GSM8K":
                response.answer = extract_number(response.answer)
            
            return {
                "agent_id" : agent.agent_id,
                "is_malicious" : agent.is_malicious,
                "answer" : response.answer.upper(),
                "reason" : response.reason,
            }
            
        round_responses = []
        
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            agent_tasks = {
                executor.submit(single_agent_round_1, agent) : agent
                for agent in agents
            }
            
            for completed_task in as_completed(agent_tasks):
                agent = agent_tasks[completed_task]
                try:
                    result = completed_task.result()
                    round_responses.append(result)
                    
                except Exception as e:
                    raise RuntimeError(f"agent_{agent.agent_id}_round_1_failed: {e}") from e
        
        # Once the generation of the first round for all agents finishes:
        round_responses.sort(key=lambda x: x['agent_id'])
        
        return round_responses
    
    def generate_debate_round_concurrent(self, question, choices, previous_round_responses, agents: List[DebateAgent], round, topology, mal_answer: str = ""):
        
        def single_agent_round_debate(agent: DebateAgent, topology = topology):
            # Ensure topology is a list of lists
            topology = topology
            
            neighbors = [
                (j, previous_round_responses[j]) for j in range(len(previous_round_responses)) 
                if (topology[agent.agent_id][j] == 1 and j != agent.agent_id)
            ]
            
            format_neighbors = "\n".join(
                f"Agent {m[0]}\nResponse: {m[1]['answer']}\nArgument: {m[1]['reason']}\n" 
                for m in neighbors
            )
            format_data={
                "agent_id" : agent.agent_id,
                "question" : question,
                "choices" : choices,
                "neighbors_messages" : format_neighbors,
                "round_num" : round,
            }
            if mal_answer:
                format_data['wrong_answer'] = str(mal_answer)

            response = agent.debate_round_generate(format_data=format_data)
            if not isinstance(response.answer, str):
                print(f"[DEBUG] Agent {agent.agent_id} Round {round} - Response is not a string: type={type(response.answer)}, value={response.answer}")
                response.answer = str(response.answer)
                
            if self.dataset_name == "GSM8K":
                response.answer = extract_number(response.answer)    
            return {
                "agent_id" : agent.agent_id,
                "is_malicious" : agent.is_malicious,
                "answer" : response.answer.upper(),
                "reason" : response.reason,
            }
            
        round_responses = []
        
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            agent_tasks = {
                executor.submit(single_agent_round_debate, agent) : agent
                for agent in agents
            }
            
            for completed_task in as_completed(agent_tasks):
                agent = agent_tasks[completed_task]
                try:
                    result = completed_task.result()
                    round_responses.append(result)
                    
                except Exception as e:
                    raise RuntimeError(f"agent_{agent.agent_id}_round_{round}_failed: {e}") from e
        
        # Once the generation of the first round for all agents finishes:
        round_responses.sort(key=lambda x: x['agent_id'])
        
        return round_responses
    
    def check_consensus(self, round_responses) -> bool:
        response_counts = {}
        for resp in round_responses:
            answer = resp['answer']
            if answer not in response_counts:
                response_counts[answer] = 0
            response_counts[answer] += 1
        
        total_agents = len(round_responses)
        for count in response_counts.values():
            if count / total_agents >= self.config.consensus_threshold:
                return True
        return False
    
    def get_answer(self, round_responses) -> str:
        response_counts = {}
        for resp in round_responses:
            answer = resp['answer']
            if answer not in response_counts:
                response_counts[answer] = 0
            response_counts[answer] += 1
        
        sorted_responses = sorted(response_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_responses[0][0]
    
    def check_answer(self, round_responses, correct_answer) -> bool:
        final_answer = self.get_answer(round_responses)
        
        # We manage the case of each dataset separately so as to not have conflicts with the answer format
        try:
            return self.dataloader.validate_answer(final_answer, correct_answer)
        except Exception as e:
            print(f"Error comparing answers: final_answer={final_answer}, correct_answer={correct_answer}, error={e}")
            return False
    
    def debate_question(self, question:str, choices:str, pbar=None, mal_answer: str = "", question_index: int = None):
        agents = self.generate_agents(question_index=question_index)
        all_rounds_responses = []
        
        # Track malicious agent indexes
        malicious_indexes = [agent.agent_id for agent in agents if agent.is_malicious]
        
        if self.random_flag:
            rng = np.random.default_rng(self.config.random_topology_data["seed"] + (question_index if question_index is not None else 0))
            density = rng.uniform(self.config.random_topology_data["density interval"][0], self.config.random_topology_data["density interval"][1])
            generated_topology = generate_random_topologies(
                num_agents=self.config.number_of_agents,
                density=density,
                rng=rng
            )
            
        topology = self.topology if not self.random_flag else generated_topology
        # If mal_answer is empty and we have malicious agents, generate a random wrong answer
        # This is needed because malicious agent prompts require {wrong_answer} key
        if not mal_answer and malicious_indexes:
            # Generate a random wrong answer (any choice except the correct one, which is unknown here)
            # So we just pick a random choice A-D
            mal_answer = random.choice(["A", "B", "C", "D"])
        
        round_1_responses = self.generate_round_1_concurrent(question, choices, agents, mal_answer=mal_answer)
        if pbar:
            pbar.update(1)
        all_rounds_responses.append(round_1_responses)
        
        # This is commented out to force always more than one round (otherwise there is no message passing)
        # if self.check_consensus(round_1_responses):
        #     return all_rounds_responses, malicious_indexes, topology
        
        for i in range(2, self.config.max_rounds+1):
            round_i_responses = self.generate_debate_round_concurrent(
                question,
                choices,
                all_rounds_responses[-1],
                agents,
                round=i,
                topology=topology,
                mal_answer=mal_answer
            )
            if pbar:
                pbar.update(1)
            all_rounds_responses.append(round_i_responses)
            if self.check_consensus(round_i_responses):
                return all_rounds_responses, malicious_indexes, topology
        return all_rounds_responses, malicious_indexes, topology
    
    def run_debate(self, questions: List[dict], progress_bars: dict, master_pbar, malicious_consensus = False):
    
        results = [None] * len(questions)
        lock = threading.Lock()
        failure_counts = Counter()
        failure_examples = []
        
        def process_single_question(index: int, question_data: dict):
            question_text = question_data['question']
            choices_text = question_data['choices']
            mal_answer = ""
            if malicious_consensus:
                wrong_answer_idx = np.random.default_rng(self.config.questions_random_seed).choice([i for i in range(0,4) if i!=question_data['answer']])
                mal_answer = chr(wrong_answer_idx + 65)
            
            pbar = progress_bars.get(index) if progress_bars else None
            
            debate_result, malicious_indexes, topology = self.debate_question(question_text, choices_text, pbar=pbar, mal_answer=mal_answer, question_index=index)
            expanded_result = {
                "question": question_text,
                "choices": choices_text,
                "debate_rounds": debate_result,
                "malicious_agent_indexes": malicious_indexes,
                "topology": topology,
                "consensus_reached": self.check_consensus(debate_result[-1]),
                "final_answer": self.get_answer(debate_result[-1]),
                "correct_answer": question_data['answer'],
                "is_correct": self.check_answer(debate_result[-1], question_data['answer']),
            }
            
            num_rounds = len(debate_result)
            with lock:
                if pbar:
                    pbar.set_description(f"Q{index+1}: {num_rounds} round(s) - Finished")
                    pbar.n = pbar.total
                    pbar.refresh()
                
                if master_pbar:
                    master_pbar.update(1)
                
            return index, expanded_result
        
        executor = ThreadPoolExecutor(max_workers=self.config.parallel_questions)
        future_to_index = {
            executor.submit(process_single_question, idx, q_data): idx
            for idx, q_data in enumerate(questions)
        }
        
        try:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    msg = str(e).strip() or e.__class__.__name__
                    failure_counts[msg] += 1
                    if len(failure_examples) < 8:
                        failure_examples.append((idx, msg))
                    tqdm.write(f"[WARN] Q{idx+1} failed: {msg}")
        except KeyboardInterrupt:
            print("\nCancelling all pending tasks immediately...")
            # Cancel all pending futures
            for future in future_to_index.keys():
                future.cancel()
            # Force shutdown without waiting
            executor._shutdown = True
            executor.shutdown(wait=False)

        if failure_counts:
            tqdm.write("\n[INFO] Question failure summary:")
            for reason, count in failure_counts.most_common():
                tqdm.write(f"  - {count}x {reason}")
            tqdm.write("[INFO] Example failed questions:")
            for idx, msg in failure_examples:
                tqdm.write(f"  - Q{idx+1}: {msg}")
        
        return results
    
    def run_evaluation(self):
        
        dataset_classes = {
            cls.TAG.upper(): cls
            for name, cls in inspect.getmembers(DatasetManager, inspect.isclass)
            if hasattr(cls, "TAG")
        }
        
        dataset_name = self.config.dataset_tag.upper()
        self.dataset_name = dataset_name
        
        if dataset_name not in dataset_classes:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        loader_cls = dataset_classes[dataset_name]

        
        loader_kwargs = {
            "num_questions": getattr(self.config, "num_questions", None),
            "random_seed": getattr(self.config, "questions_random_seed", None),
        }
            
        self.dataloader = loader_cls(**loader_kwargs)

        # Use parser from the selected dataloader.
        self.llm = self.base_llm | RunnableLambda(self.dataloader.parse_model_output)
        
        questions = self.dataloader.get_formatted_questions()
        
        master_pbar = tqdm(
            total=len(questions),
            desc="Overall Progress",
            position=0,
            ncols=80,
            leave=True
        )
        
        progress_bars = {}
        if self.config.verbose:
            for idx in range(len(questions)):
                progress_bars[idx] = tqdm(
                    total=self.config.max_rounds,
                    desc=f"Q{idx+1}",
                    position=idx+1,
                    ncols=80,
                    leave=True
                )

        start_time = datetime.datetime.now()
        interrupted = False
        
        try:
            results = self.run_debate(questions, progress_bars, master_pbar=master_pbar, malicious_consensus=True)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C). Saving partial results...")
            interrupted = True
            results = None
        finally:
            # Close all progress bars
            master_pbar.close()
            if self.config.verbose:
                for pbar in progress_bars.values():
                    pbar.close()
                    
        return results, interrupted