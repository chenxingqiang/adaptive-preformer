import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
import time

class PerformanceMetrics:
    """Comprehensive performance metrics calculator"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.compute_times = []
        self.memory_usage = []
        self.quality_scores = []
        
    def update(self, outputs, targets, compute_time, memory):
        self.predictions.append(outputs['logits'].detach().cpu())
        self.targets.append(targets.cpu())
        self.compute_times.append(compute_time)
        self.memory_usage.append(memory)
        self.quality_scores.append(outputs['quality_scores'].detach().cpu())
        
    def compute(self):
        # Concatenate all batches
        preds = torch.cat(self.predictions)
        targets = torch.cat(self.targets)
        quality = torch.cat(self.quality_scores)
        
        # Basic metrics
        accuracy = (preds.argmax(1) == targets).float().mean().item()
        auc = roc_auc_score(targets, F.softmax(preds, dim=1)[:, 1])
        
        # Efficiency metrics
        avg_compute_time = np.mean(self.compute_times)
        avg_memory = np.mean(self.memory_usage)
        
        # Quality metrics
        avg_quality = quality.mean().item()
        quality_std = quality.std().item()
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'compute_time_ms': avg_compute_time * 1000,
            'memory_mb': avg_memory / (1024 * 1024),
            'quality_mean': avg_quality,
            'quality_std': quality_std
        }
    
