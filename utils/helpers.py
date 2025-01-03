import torch
import numpy as np
import random
import logging
from pathlib import Path
import json
import yaml

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    
def load_config(config_path):
    """Load configuration from file"""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path) as f:
            config = json.load(f)
    elif config_path.suffix in ['.yml', '.yaml']:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
    return config
    
def save_experiment_results(results, save_dir):
    """Save experiment results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame(results['metrics'])
    metrics_df.to_csv(save_dir / 'metrics.csv')
    
    # Save model predictions
    torch.save(results['predictions'], save_dir / 'predictions.pt')
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(results['config'], f, indent=2)
        
    # Save plots
    for name, fig in results['figures'].items():
        fig.savefig(save_dir / f'{name}.png')
        plt.close(fig) 