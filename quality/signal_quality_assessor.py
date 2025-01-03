import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import numpy as np

class SignalQualityAssessor(nn.Module):
    """Hierarchical signal quality assessment"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 局部质量评估
        self.local_assessor = LocalQualityNet(
            input_dim=config.input_dim,
            window_size=config.quality.local_window_size
        )
        
        # 全局质量评估
        self.global_assessor = GlobalQualityNet(
            input_dim=config.input_dim,
            hidden_dim=config.quality.hidden_dim
        )
        
        # 噪声检测器
        self.noise_detector = NoiseDetector(
            input_dim=config.input_dim
        )
        
        # 质量分数融合
        self.quality_fusion = QualityScoreFusion(
            input_dim=3  # local, global, noise scores
        )
        
    def forward(self, x):
        """
        Args:
            x: Input signal (B, L, D)
        Returns:
            quality_score: Overall quality score
            quality_info: Detailed quality information
        """
        # 局部质量评估
        local_quality = self.local_assessor(x)
        
        # 全局质量评估
        global_quality = self.global_assessor(x)
        
        # 噪声检测
        noise_score = self.noise_detector(x)
        
        # 融合质量分数
        quality_scores = torch.stack([
            local_quality,
            global_quality,
            noise_score
        ], dim=-1)
        
        overall_quality = self.quality_fusion(quality_scores)
        
        return overall_quality, {
            'local_quality': local_quality,
            'global_quality': global_quality,
            'noise_score': noise_score
        }

class NoiseDetector(nn.Module):
    """Signal noise level detection"""
    def __init__(self, input_dim):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.noise_estimator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 转换维度 (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        
        # 特征提取
        features = self.conv_layers(x).squeeze(-1)
        
        # 噪声估计
        noise_score = self.noise_estimator(features)
        
        return noise_score

class QualityScoreFusion(nn.Module):
    """Fusion of different quality scores"""
    def __init__(self, input_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, quality_scores):
        """
        Args:
            quality_scores: Different quality scores (B, N_scores)
        Returns:
            Fused quality score (B, 1)
        """
        # 计算注意力权重
        weights = self.attention(quality_scores)
        
        # 加权融合
        fused_score = (quality_scores * weights).sum(dim=-1, keepdim=True)
        
        return fused_score   

class QualityScoreFusion(nn.Module):
    """Fusion of different quality scores"""
    def __init__(self, input_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, quality_scores):
        """
        Args:
            quality_scores: Different quality scores (B, N_scores)
        Returns:
            Fused quality score (B, 1)
        """
        # 计算注意力权重
        weights = self.attention(quality_scores)
        
        # 加权融合
        fused_score = (quality_scores * weights).sum(dim=-1, keepdim=True)
        
        return fused_score 