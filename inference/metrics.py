import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class InferenceMetrics:
    """Metrics tracker for inference"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.batch_times = []
        self.quality_scores = []
        
    def update(self, batch_metrics):
        """Update metrics with batch results"""
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
            
    def get_summary(self):
        """Get summary of inference metrics"""
        summary = {}
        
        # Compute timing statistics
        times = np.array(self.metrics['inference_time'])
        summary['timing'] = {
            'mean_time': times.mean(),
            'std_time': times.std(),
            'median_time': np.median(times),
            'p95_time': np.percentile(times, 95),
            'throughput': 1.0 / times.mean()
        }
        
        # Quality statistics
        quality = np.array(self.metrics['quality_score'])
        summary['quality'] = {
            'mean_quality': quality.mean(),
            'std_quality': quality.std(),
            'min_quality': quality.min(),
            'max_quality': quality.max()
        }
        
        return summary
        
    def plot_metrics(self):
        """Plot inference metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Inference time distribution
        sns.histplot(
            self.metrics['inference_time'],
            ax=axes[0,0],
            bins=50
        )
        axes[0,0].set_title('Inference Time Distribution')
        axes[0,0].set_xlabel('Time (s)')
        
        # Quality score distribution
        sns.histplot(
            self.metrics['quality_score'],
            ax=axes[0,1],
            bins=50
        )
        axes[0,1].set_title('Quality Score Distribution')
        
        # Inference time series
        axes[1,0].plot(self.metrics['inference_time'])
        axes[1,0].set_title('Inference Time Series')
        axes[1,0].set_xlabel('Batch')
        axes[1,0].set_ylabel('Time (s)')
        
        # Quality score series
        axes[1,1].plot(self.metrics['quality_score'])
        axes[1,1].set_title('Quality Score Series')
        axes[1,1].set_xlabel('Batch')
        
        plt.tight_layout()
        return fig
        
    def save_metrics(self, save_path):
        """Save metrics to file"""
        save_path = Path(save_path)
        
        # Save metrics as CSV
        pd.DataFrame(self.metrics).to_csv(
            save_path / 'inference_metrics.csv'
        )
        
        # Save plots
        self.plot_metrics().savefig(
            save_path / 'inference_metrics.png'
        ) 