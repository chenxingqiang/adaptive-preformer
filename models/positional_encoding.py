import torch
import torch.nn as nn
import math

class ContinuousPositionalEncoding(nn.Module):
    """Continuous positional encoding for irregular sampling intervals"""
    def __init__(self, d_model, max_len=10000, learnable=True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.learnable = learnable
        
        # 基础傅里叶特征
        self.fourier_features = FourierFeatures(d_model // 2)
        
        # 可学习的尺度因子
        self.scale_factor = nn.Parameter(torch.ones(1))
        
        # 可学习的相位偏移
        if learnable:
            self.phase_shift = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x, timestamps=None):
        """
        Args:
            x: Input tensor (B, L, D)
            timestamps: Optional timestamp tensor (B, L)
        """
        if timestamps is None:
            timestamps = torch.arange(x.size(1), device=x.device)
            
        # 计算连续傅里叶特征
        fourier_encodings = self.fourier_features(timestamps)
        
        # 应用尺度因子和相位偏移
        pe = self.scale_factor * fourier_encodings
        if self.learnable:
            pe = pe + self.phase_shift
            
        return pe

class FourierFeatures(nn.Module):
    """Fourier feature generation for continuous positions"""
    def __init__(self, output_dim, input_scale=1.0):
        super().__init__()
        self.output_dim = output_dim
        self.input_scale = input_scale
        
        # 生成频率矩阵
        self.freq_bands = nn.Parameter(
            torch.randn(output_dim) * input_scale,
            requires_grad=True
        )
        
    def forward(self, x):
        """
        Args:
            x: Input positions (B, L)
        Returns:
            Fourier features (B, L, 2*output_dim)
        """
        # 计算投影
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.freq_bands
        
        # 生成正弦和余弦特征
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)
        
        # 拼接特征
        return torch.cat([sin_features, cos_features], dim=-1) 