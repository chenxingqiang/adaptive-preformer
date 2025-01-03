import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr
import logging
from tqdm import tqdm

class ModelEvaluator:
    """Comprehensive model evaluation tools"""
    def __init__(self, model, device, metrics_tracker=None):
        self.model = model
        self.device = device
        self.metrics_tracker = metrics_tracker
        
    def evaluate_model(self, test_loader, criteria=None):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_features = []
        all_quality_scores = []
        
        with torch.no_grad():
            for batch, targets in tqdm(test_loader, desc="Evaluating"):
                # Forward pass
                outputs = self.model(batch.to(self.device))
                
                # Collect results
                all_predictions.append(outputs['logits'].cpu())
                all_targets.append(targets.cpu())
                all_features.append(outputs['features'].cpu())
                all_quality_scores.append(outputs['quality_scores'].cpu())
                
        # Concatenate results
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        features = torch.cat(all_features)
        quality_scores = torch.cat(all_quality_scores)
        
        # Compute metrics
        results = {
            'classification_metrics': self.compute_classification_metrics(predictions, targets),
            'feature_metrics': self.compute_feature_metrics(features),
            'quality_metrics': self.compute_quality_metrics(quality_scores),
            'efficiency_metrics': self.compute_efficiency_metrics(test_loader)
        }
        
        return results
        
    def compute_classification_metrics(self, predictions, targets):
        """Compute classification metrics"""
        pred_labels = predictions.argmax(dim=1)
        conf_matrix = confusion_matrix(targets, pred_labels)
        class_report = classification_report(targets, pred_labels, output_dict=True)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'accuracy': (pred_labels == targets).float().mean().item()
        }
        
    def compute_feature_metrics(self, features):
        """Compute feature quality metrics"""
        # Feature correlation
        corr_matrix = np.corrcoef(features.numpy().T)
        feature_redundancy = np.mean(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
        
        # Feature sparsity
        feature_sparsity = (features.abs() < 1e-5).float().mean().item()
        
        return {
            'feature_redundancy': feature_redundancy,
            'feature_sparsity': feature_sparsity
        }
        
    def compute_quality_metrics(self, quality_scores):
        """Compute quality assessment metrics"""
        temporal_consistency = torch.abs(
            quality_scores[:, 1:] - quality_scores[:, :-1]
        ).mean().item()
        
        quality_stats = {
            'mean': quality_scores.mean().item(),
            'std': quality_scores.std().item(),
            'temporal_consistency': temporal_consistency
        }
        
        return quality_stats
        
    def compute_efficiency_metrics(self, data_loader):
        """Compute model efficiency metrics"""
        batch = next(iter(data_loader))[0].to(self.device)
        
        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = self.model(batch)
        max_memory = torch.cuda.max_memory_allocated()
        
        # Compute time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_event.record()
            _ = self.model(batch)
            end_event.record()
            
        torch.cuda.synchronize()
        compute_time = start_event.elapsed_time(end_event)
        
        return {
            'memory_mb': max_memory / (1024 * 1024),
            'compute_time_ms': compute_time,
            'samples_per_second': batch.shape[0] / (compute_time / 1000)
        } 