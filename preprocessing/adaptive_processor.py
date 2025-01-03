import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class SignalQualityAssessor(nn.Module):
    """信号质量评估模块"""
    def __init__(self,
                 d_model: int,
                 n_features: int = 8):
        super().__init__()
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, n_features, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 质量评分
        self.quality_scorer = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 提取特征
        features = self.feature_extractor(x.unsqueeze(1))
        features = features.squeeze(-1)
        
        # 计算质量分数
        quality_score = self.quality_scorer(features)
        
        return quality_score, features

class NoiseDetector(nn.Module):
    """噪声检测模块"""
    def __init__(self,
                 d_model: int,
                 window_size: int = 64):
        super().__init__()
        self.window_size = window_size
        
        # 局部特征提取
        self.local_conv = nn.Conv1d(1, d_model, kernel_size=window_size, stride=window_size//2)
        
        # 噪声检测器
        self.detector = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取局部特征
        local_features = self.local_conv(x.unsqueeze(1))
        
        # 检测噪声
        noise_mask = self.detector(local_features)
        
        # 上采样到原始大小
        noise_mask = F.interpolate(noise_mask, size=x.size(-1), mode='linear')
        
        return noise_mask.squeeze(1)

class AdaptiveFilterBank(nn.Module):
    """自适应滤波器组"""
    def __init__(self,
                 n_filters: int = 8,
                 filter_length: int = 31):
        super().__init__()
        self.n_filters = n_filters
        self.filter_length = filter_length
        
        # 可学习的滤波器参数
        self.filter_params = nn.Parameter(torch.randn(n_filters, 1, filter_length))
        
        # 滤波器选择网络
        self.filter_selector = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, n_filters),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        # 应用滤波器组
        x_pad = F.pad(x.unsqueeze(1), ((self.filter_length-1)//2, (self.filter_length-1)//2))
        filtered = F.conv1d(x_pad, self.filter_params)  # [B, n_filters, L]
        
        # 基于特征选择滤波器
        weights = self.filter_selector(features)  # [B, n_filters]
        
        # 加权组合
        output = torch.sum(filtered * weights.view(B, -1, 1), dim=1)
        
        return output

class AdaptivePreprocessor(nn.Module):
    """自适应预处理器"""
    def __init__(self,
                 d_model: int = 256,
                 n_filters: int = 8):
        super().__init__()
        
        # 质量评估
        self.quality_assessor = SignalQualityAssessor(d_model)
        
        # 噪声检测
        self.noise_detector = NoiseDetector(d_model)
        
        # 自适应滤波
        self.filter_bank = AdaptiveFilterBank(n_filters)
        
        # 预处理策略网络
        self.strategy_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),  # 3个预处理操作的权重
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 质量评估
        quality_score, features = self.quality_assessor(x)
        
        # 确定预处理策略
        strategy_weights = self.strategy_net(features)
        
        # 噪声检测
        noise_mask = self.noise_detector(x)
        
        # 自适应滤波
        filtered = self.filter_bank(x, features)
        
        # 组合处理结果
        output = (
            strategy_weights[:, 0:1] * x +
            strategy_weights[:, 1:2] * filtered +
            strategy_weights[:, 2:3] * (x * (1 - noise_mask))
        )
        
        # 返回处理后的信号和中间结果
        return output, {
            'quality_score': quality_score,
            'noise_mask': noise_mask,
            'strategy_weights': strategy_weights
        } 