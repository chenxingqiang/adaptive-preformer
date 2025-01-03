import torch
import torch.nn as nn

class HierarchicalQualityAssessor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.local_assessor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        local_quality = self.local_assessor(x).sigmoid()
        return {
            'local_quality': local_quality,
            'global_quality': local_quality.mean()
        } 