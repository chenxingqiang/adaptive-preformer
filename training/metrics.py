import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingMetrics:
    """Training metrics tracker"""
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'quality_scores': [],
            'preprocessing_params': [],
            'computation_time': []
        }
        
        self.epoch_metrics = {
            'loss_components': [],
            'quality_distribution': [],
            'parameter_changes': []
        }
        
    def update(self, loss_components):
        """Update batch metrics"""
        self.epoch_metrics['loss_components'].append(loss_components)
        
    def log_epoch(self, train_loss, val_loss):
        """Log epoch-level metrics"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        
        # Compute epoch statistics
        self.compute_epoch_stats()
        
        # Reset epoch metrics
        self.epoch_metrics = {k: [] for k in self.epoch_metrics}
        
    def compute_epoch_stats(self):
        """Compute statistics for the epoch"""
        loss_components = pd.DataFrame(self.epoch_metrics['loss_components'])
        
        self.metrics['loss_components'].append({
            'mean': loss_components.mean(),
            'std': loss_components.std()
        })
        
    def plot_training_curves(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0,0].plot(self.metrics['train_loss'], label='Train')
        axes[0,0].plot(self.metrics['val_loss'], label='Validation')
        axes[0,0].set_title('Loss Curves')
        axes[0,0].legend()
        
        # Quality scores
        sns.violinplot(data=self.metrics['quality_scores'], ax=axes[0,1])
        axes[0,1].set_title('Quality Score Distribution')
        
        # Loss components
        loss_df = pd.DataFrame(self.metrics['loss_components'])
        sns.heatmap(loss_df.T, ax=axes[1,0], cmap='viridis')
        axes[1,0].set_title('Loss Components')
        
        # Preprocessing parameters
        param_df = pd.DataFrame(self.metrics['preprocessing_params'])
        sns.lineplot(data=param_df, ax=axes[1,1])
        axes[1,1].set_title('Preprocessing Parameters')
        
        plt.tight_layout()
        return fig
        
    def get_summary(self):
        """Get training summary"""
        return {
            'final_train_loss': self.metrics['train_loss'][-1],
            'final_val_loss': self.metrics['val_loss'][-1],
            'best_val_loss': min(self.metrics['val_loss']),
            'total_epochs': len(self.metrics['train_loss']),
            'quality_stats': {
                'mean': np.mean(self.metrics['quality_scores']),
                'std': np.std(self.metrics['quality_scores'])
            }
        }
        
    def save(self, path):
        """Save metrics to file"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as CSV
        pd.DataFrame(self.metrics).to_csv(path / 'metrics.csv')
        
        # Save plots
        self.plot_training_curves().savefig(path / 'training_curves.png') 