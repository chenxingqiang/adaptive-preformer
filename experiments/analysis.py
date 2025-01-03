import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import json

class ExperimentAnalyzer:
    """Analyzes and visualizes experiment results"""
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.results = self.load_results()
        
    def load_results(self):
        """Load experiment results"""
        results = {
            'metrics': pd.read_csv(self.experiment_dir / 'metrics.csv'),
            'config': json.load(open(self.experiment_dir / 'config.json')),
            'checkpoints': list(self.experiment_dir.glob('checkpoint_*.pt'))
        }
        return results
        
    def plot_training_curves(self):
        """Plot training and validation curves"""
        metrics = self.results['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        sns.lineplot(data=metrics, x='epoch', y='train_loss', ax=axes[0,0], label='Train')
        sns.lineplot(data=metrics, x='epoch', y='val_loss', ax=axes[0,0], label='Val')
        axes[0,0].set_title('Loss Curves')
        
        # Accuracy curves
        sns.lineplot(data=metrics, x='epoch', y='train_acc', ax=axes[0,1], label='Train')
        sns.lineplot(data=metrics, x='epoch', y='val_acc', ax=axes[0,1], label='Val')
        axes[0,1].set_title('Accuracy Curves')
        
        # Quality scores
        sns.lineplot(data=metrics, x='epoch', y='quality_mean', ax=axes[1,0])
        axes[1,0].set_title('Average Quality Scores')
        
        # Efficiency metrics
        sns.lineplot(data=metrics, x='epoch', y='compute_time_ms', ax=axes[1,1])
        axes[1,1].set_title('Computation Time')
        
        plt.tight_layout()
        return fig
        
    def analyze_feature_space(self, model, dataloader):
        """Analyze learned feature space"""
        features = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for batch, target in dataloader:
                batch_features = model.extract_features(batch)
                features.append(batch_features.cpu())
                labels.append(target.cpu())
                
        features = torch.cat(features)
        labels = torch.cat(labels)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Learned Features')
        
        return plt.gcf()
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'config': self.results['config'],
            'final_metrics': self.results['metrics'].iloc[-1].to_dict(),
            'best_checkpoint': max(self.results['checkpoints'], key=lambda x: int(x.stem.split('_')[-1])),
            'training_curves': self.plot_training_curves(),
            'feature_analysis': None  # Will be filled if model is provided
        }
        
        return report 