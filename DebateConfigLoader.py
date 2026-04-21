from dataclasses import dataclass, field, asdict
from typing import List, TypedDict

AdjMatrix = List[List[int]]

RandomTopologyData = TypedDict(
"RandomTopologyData",
{
"seed": int,
"density interval": tuple[float, float],
},
)

@dataclass
class DebateConfig:
    timeout: float = 0
    is_random_topology: bool = False  # Flag to indicate if the topology is random (for per-question randomization)
    random_topology_data: RandomTopologyData = field(
                    default_factory=lambda: {
                    "seed": 24,
                    "density interval": (0.3, 0.7),
                    }
    )
    
    # Debate configuration
    max_rounds: int = 3
    number_of_agents: int = 5
    number_malicious_agents: int = 2
    consensus_threshold: float = 1
    topology: List[List[int]] = field(default_factory=lambda: [
        [0,1,1,1,1],
        [1,0,1,1,1],
        [1,1,0,1,1],
        [1,1,1,0,1],
        [1,1,1,1,0]
    ])
    prompts_file: str = "prompts1.json"
    malicious_randomization_seed: int = 42  # Seed for per-question malicious agent randomization
    
    # Execution configuration
    parallel_questions: int = 50
    parallel_agents: bool = True
    save_logs_json: bool = True
    save_logs_dir: str = "debate_logs"
    verbose: bool = False
    
    num_questions: int = 500
    questions_random_seed: int = 28
    dataset_tag: str = "MMLU"
    
    def __post_init__(self):
        if self.number_malicious_agents > self.number_of_agents:
            raise ValueError("Number of malicious agents cannot exceed total number of agents.")
        # Pending to implement here more validation
        
    def to_dict(self):
        """Convert config to dictionary automatically"""
        d = asdict(self)
        d['topology'] = str(d["topology"])
        return d