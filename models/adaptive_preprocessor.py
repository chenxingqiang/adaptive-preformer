import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePreprocessor(nn.Module):
    """Learnable preprocessing pipeline"""
    def __init__(self, input_dim):
        super().__init__()
        
        # Quality assessment
        self.quality_assessor = QualityAssessor(input_dim)
        
        # Learnable filter bank
        self.filter_bank = nn.ModuleList([
            LearnableFilter(input_dim) 
            for _ in range(4)  # Multiple filter types
        ])
        
        # Adaptive feature extraction
        self.feature_extractor = AdaptiveFeatureExtractor(input_dim)
        
        # Preprocessing strategy
        self.strategy_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.filter_bank))
        )
        
    def forward(self, x):
        # Assess signal quality
        quality_scores = self.quality_assessor(x)
        
        # Determine preprocessing strategy
        strategy_weights = F.softmax(self.strategy_net(x.mean(1)), dim=-1)
        
        # Apply filters adaptively
        filtered = torch.stack([
            f(x) * w for f, w in zip(self.filter_bank, strategy_weights.T)
        ]).sum(0)
        
        # Extract features
        features = self.feature_extractor(filtered, quality_scores)
        
        return {
            'processed': filtered,
            'features': features,
            'quality_scores': quality_scores,
            'strategy_weights': strategy_weights
        }