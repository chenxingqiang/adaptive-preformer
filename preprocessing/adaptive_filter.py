import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal

class AdaptiveFilterBank(nn.Module):
    """Learnable filter bank for signal preprocessing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 可学习的滤波器参数
        self.filter_params = nn.ParameterDict({
            'low_cutoff': nn.Parameter(torch.tensor([0.1])),
            'high_cutoff': nn.Parameter(torch.tensor([50.0])),
            'filter_order': nn.Parameter(torch.tensor([4.0])),
            'ripple_db': nn.Parameter(torch.tensor([1.0]))
        })
        
        # 自适应阈值
        self.adaptive_threshold = nn.Parameter(torch.tensor([0.5]))
        
        # 质量感知调制
        self.quality_modulation = QualityModulation(
            input_dim=config.input_dim
        )
        
    def forward(self, x, quality_score=None):
        """
        Args:
            x: Input signal (B, L, D)
            quality_score: Optional quality score (B, 1)
        Returns:
            Filtered signal
        """
        # 获取当前滤波器参数
        current_params = self._get_filter_params(quality_score)
        
        # 应用滤波器
        x_filtered = self._apply_filter(x, current_params)
        
        # 质量感知调制
        if quality_score is not None:
            x_filtered = self.quality_modulation(x_filtered, quality_score)
            
        return x_filtered
        
    def _get_filter_params(self, quality_score=None):
        """Get adaptive filter parameters based on quality"""
        params = {}
        
        # 基础参数
        params['low_cutoff'] = torch.sigmoid(self.filter_params['low_cutoff']) * 50
        params['high_cutoff'] = params['low_cutoff'] + \
                               torch.sigmoid(self.filter_params['high_cutoff']) * 50
        params['order'] = torch.clamp(self.filter_params['filter_order'], 2, 8)
        params['ripple'] = torch.sigmoid(self.filter_params['ripple_db']) * 3
        
        # 根据质量分数调整参数
        if quality_score is not None:
            params['low_cutoff'] *= (1 + 0.5 * (1 - quality_score))
            params['high_cutoff'] *= (1 - 0.3 * (1 - quality_score))
            params['order'] += 2 * (1 - quality_score)
            
        return params
        
    def _apply_filter(self, x, params):
        """Apply filtering with given parameters"""
        # 转换为频域
        x_fft = torch.fft.rfft(x, dim=1)
        
        # 构建滤波器响应
        freqs = torch.fft.rfftfreq(x.size(1), d=1.0/self.config.sampling_rate)
        filter_response = self._get_filter_response(freqs, params)
        
        # 应用滤波器
        x_filtered_fft = x_fft * filter_response.unsqueeze(0).unsqueeze(-1)
        
        # 转回时域
        x_filtered = torch.fft.irfft(x_filtered_fft, n=x.size(1), dim=1)
        
        return x_filtered
        
    def _get_filter_response(self, freqs, params):
        """Generate filter frequency response"""
        # 构建巴特沃斯滤波器响应
        low_pass = 1 / (1 + (freqs/params['low_cutoff'])**(2*params['order']))
        high_pass = 1 / (1 + (params['high_cutoff']/freqs)**(2*params['order']))
        
        return low_pass * high_pass

class QualityModulation(nn.Module):
    """Quality-aware signal modulation"""
    def __init__(self, input_dim):
        super().__init__()
        
        self.modulation_net = nn.Sequential(
            nn.Linear(input_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, quality_score):
        """
        Args:
            x: Input signal (B, L, D)
            quality_score: Quality score (B, 1)
        """
        # 生成调制系数
        quality_expanded = quality_score.expand(-1, x.size(1), -1)
        modulation_input = torch.cat([x, quality_expanded], dim=-1)
        modulation_coef = self.modulation_net(modulation_input)
        
        # 应用调制
        return x * modulation_coef 