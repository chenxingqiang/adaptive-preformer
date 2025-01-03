import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Model configuration
    model: dict = None
    
    # Training configuration
    training: dict = None
    
    # Data configuration
    data: dict = None
    
    # Preprocessing configuration
    preprocessing: dict = None
    
    # Quality assessment configuration
    quality: dict = None
    
    # Optimization configuration
    optimization: dict = None
    
    @classmethod
    def from_yaml(cls, path):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
        
    def save(self, path):
        """Save configuration to file"""
        path = Path(path)
        
        # Save as YAML
        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(asdict(self), f, indent=2)
            
        # Also save as JSON for easier parsing
        with open(path / 'config.json', 'w') as f:
            json.dump(asdict(self), f, indent=2)
            
class ConfigManager:
    """Manager for experiment configurations"""
    def __init__(self, base_config_path):
        self.base_config = ExperimentConfig.from_yaml(base_config_path)
        self.experiment_configs = {}
        
    def create_experiment_variant(self, name, **kwargs):
        """Create a variant of the base configuration"""
        config = ExperimentConfig.from_yaml(self.base_config)
        
        # Update configuration with new parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                current = getattr(config, key)
                if isinstance(current, dict):
                    current.update(value)
                else:
                    setattr(config, key, value)
                    
        self.experiment_configs[name] = config
        return config
        
    def save_experiment_configs(self, save_dir):
        """Save all experiment configurations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save base config
        self.base_config.save(save_dir / 'base')
        
        # Save variants
        for name, config in self.experiment_configs.items():
            config.save(save_dir / name)
            
    def load_experiment_config(self, name):
        """Load a specific experiment configuration"""
        if name in self.experiment_configs:
            return self.experiment_configs[name]
        else:
            raise ValueError(f"No configuration found for experiment: {name}")
            
    def get_all_configs(self):
        """Get all experiment configurations"""
        return {
            'base': self.base_config,
            **self.experiment_configs
        } 