import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityAssessor(nn.Module):
    """Hierarchical signal quality assessment"""
    def __init__(self, config):
        super().__init__()
        
        # Local quality assessment
        self.local_assessor = LocalQualityNet(config)
        
        # Global quality assessment
        self.global_assessor = GlobalQualityNet(config)
        
        # Feature fusion
        self.fusion = QualityFusion(config)
        
    def forward(self, x):
        # Local quality assessment
        local_quality = self.local_assessor(x)
        
        # Global quality assessment
        global_quality = self.global_assessor(x)
        
        # Fuse quality scores
        quality_score = self.fusion(local_quality, global_quality)
        
        return quality_score
        
class LocalQualityNet(nn.Module):
    """Local signal quality assessment"""
    def __init__(self, config):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv(x)
        
class GlobalQualityNet(nn.Module):
    """Global signal quality assessment"""
    def __init__(self, config):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.global_pool(x).squeeze(-1)
        return self.mlp(x)
        
class QualityFusion(nn.Module):
    """Quality score fusion"""
    def __init__(self, config):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, local_quality, global_quality):
        # Combine quality scores with attention
        scores = torch.stack([local_quality, global_quality], dim=-1)
        weights = self.attention(scores)
        
        return (scores * weights).sum(dim=-1) 