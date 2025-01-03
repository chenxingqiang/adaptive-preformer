from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    """Model configuration"""
    input_dim: int = 64
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 1000
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    data_paths: List[str]
    output_dir: str
    device: str = 'cuda'
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f) 