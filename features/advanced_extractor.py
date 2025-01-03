import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.fft import fft, rfft
import pywt

class TimeFeatureExtractor(nn.Module):
    """时域特征提取器"""
    def __init__(self,
                 d_model: int,
                 window_sizes: List[int] = [32, 64, 128]):
        super().__init__()
        self.d_model = d_model
        self.window_sizes = window_sizes
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, d_model // len(window_sizes), 
                         kernel_size=size, padding=size//2),
                nn.BatchNorm1d(d_model // len(window_sizes)),
                nn.ReLU()
            )
            for size in window_sizes
        ])
        
        # 统计特征提取
        self.stat_proj = nn.Linear(4, d_model // len(window_sizes))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        features = []
        
        # 卷积特征
        for conv in self.conv_layers:
            feat = conv(x.unsqueeze(1))
            features.append(feat)
            
        # 统计特征
        stats = torch.stack([
            x.mean(dim=1),
            x.std(dim=1),
            x.max(dim=1)[0],
            x.min(dim=1)[0]
        ], dim=1)
        stat_features = self.stat_proj(stats).unsqueeze(-1).expand(-1, -1, L)
        features.append(stat_features)
        
        return torch.cat(features, dim=1)

class FrequencyFeatureExtractor(nn.Module):
    """频域特征提取器"""
    def __init__(self,
                 d_model: int,
                 n_freq_bands: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_freq_bands = n_freq_bands
        
        # 频带分析器
        self.band_analyzer = nn.Sequential(
            nn.Linear(n_freq_bands, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 相位分析器
        self.phase_analyzer = nn.Sequential(
            nn.Linear(n_freq_bands, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算FFT
        fft_features = rfft(x, dim=-1)
        magnitudes = torch.abs(fft_features)
        phases = torch.angle(fft_features)
        
        # 分频带
        freq_bands = torch.split(magnitudes, magnitudes.size(-1) // self.n_freq_bands, dim=-1)
        phase_bands = torch.split(phases, phases.size(-1) // self.n_freq_bands, dim=-1)
        
        # 提取频带特征
        band_features = self.band_analyzer(torch.stack([f.mean(dim=-1) for f in freq_bands], dim=-1))
        phase_features = self.phase_analyzer(torch.stack([p.mean(dim=-1) for p in phase_bands], dim=-1))
        
        return torch.cat([band_features, phase_features], dim=-1)

class WaveletFeatureExtractor(nn.Module):
    """小波特征提取器"""
    def __init__(self,
                 d_model: int,
                 wavelet: str = 'db4',
                 levels: int = 4):
        super().__init__()
        self.d_model = d_model
        self.wavelet = wavelet
        self.levels = levels
        
        # 小波系数分析器
        self.coeff_analyzer = nn.Sequential(
            nn.Linear(levels * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        features = []
        
        # 对每个样本进行小波变换
        for i in range(B):
            coeffs = pywt.wavedec(x[i].cpu().numpy(), self.wavelet, level=self.levels)
            
            # 提取每个尺度的特征
            level_features = []
            for coeff in coeffs:
                level_features.extend([
                    np.mean(np.abs(coeff)),
                    np.std(coeff)
                ])
            
            features.append(level_features)
            
        # 转换回tensor并分析
        features = torch.tensor(features, device=x.device)
        return self.coeff_analyzer(features).unsqueeze(-1).expand(-1, -1, L)

class AdvancedFeatureExtractor(nn.Module):
    """高级特征提取器"""
    def __init__(self,
                 d_model: int = 256):
        super().__init__()
        
        # 各类特征提取器
        self.time_extractor = TimeFeatureExtractor(d_model // 3)
        self.freq_extractor = FrequencyFeatureExtractor(d_model // 3)
        self.wavelet_extractor = WaveletFeatureExtractor(d_model // 3)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取各类特征
        time_features = self.time_extractor(x)
        freq_features = self.freq_extractor(x)
        wavelet_features = self.wavelet_extractor(x)
        
        # 特征融合
        combined = torch.cat([
            time_features,
            freq_features,
            wavelet_features
        ], dim=1)
        
        return self.fusion(combined.transpose(1, 2)).transpose(1, 2) 