import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss for joint optimization"""
    def __init__(self):
        super().__init__()
        
        # Task-specific losses
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Quality assessment loss
        self.quality_loss = QualityConsistencyLoss()
        
        # Efficiency losses
        self.complexity_loss = ComplexityLoss()
        self.memory_loss = MemoryEfficiencyLoss()
        
        # Learnable loss weights
        self.loss_weights = nn.Parameter(torch.ones(6))
        
    def forward(self, outputs, targets, model_info):
        losses = {
            'reconstruction': self.reconstruction_loss(outputs['reconstructed'], targets),
            'prediction': self.prediction_loss(outputs['predicted'], targets),
            'contrastive': self.contrastive_loss(outputs['features']),
            'quality': self.quality_loss(outputs['quality_scores']),
            'complexity': self.complexity_loss(model_info['compute_stats']),
            'memory': self.memory_loss(model_info['memory_stats'])
        }
        
        # Dynamic loss weighting
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = sum(w * l for w, l in zip(weights, losses.values()))
        
        return total_loss, losses

class QualityConsistencyLoss(nn.Module):
    """Loss for maintaining consistency in quality assessment"""
    def __init__(self):
        super().__init__()
        
    def forward(self, quality_scores):
        temporal_consistency = torch.abs(
            quality_scores[:, 1:] - quality_scores[:, :-1]
        ).mean()
        
        return temporal_consistency

class ComplexityLoss(nn.Module):
    """Loss for controlling computational complexity"""
    def __init__(self, target_flops=1e9):
        super().__init__()
        self.target_flops = target_flops
        
    def forward(self, compute_stats):
        return F.relu(compute_stats['flops'] - self.target_flops) 