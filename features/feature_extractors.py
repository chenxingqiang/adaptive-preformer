import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import pywt

class FeatureExtractorBase(nn.Module):
    """Base class for feature extractors"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def compute_importance(self, features, labels=None):
        """Compute feature importance scores"""
        raise NotImplementedError
        
class TimeFeatureExtractor(FeatureExtractorBase):
    """Time domain feature extraction"""
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        
        self.feature_nets = nn.ModuleDict({
            'statistical': StatisticalNet(input_dim),
            'temporal': TemporalNet(input_dim),
            'morphological': MorphologicalNet(input_dim)
        })
        
        # 特征融合
        total_features = sum(net.output_dim for net in self.feature_nets.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # 提取各类特征
        features = []
        for net in self.feature_nets.values():
            feat = net(x)
            features.append(feat)
            
        # 特征融合
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)
        
class FrequencyFeatureExtractor(FeatureExtractorBase):
    """Frequency domain feature extraction"""
    def __init__(self, input_dim, output_dim, sampling_rate=250):
        super().__init__(input_dim, output_dim)
        self.sampling_rate = sampling_rate
        
        self.spectral_nets = nn.ModuleDict({
            'power': SpectralPowerNet(input_dim),
            'phase': SpectralPhaseNet(input_dim),
            'coherence': SpectralCoherenceNet(input_dim)
        })
        
        total_features = sum(net.output_dim for net in self.spectral_nets.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # 计算频谱
        x_fft = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(x.size(1), d=1.0/self.sampling_rate)
        
        # 提取频域特征
        features = []
        for net in self.spectral_nets.values():
            feat = net(x_fft, freqs)
            features.append(feat)
            
        # 特征融合
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)

class WaveletFeatureExtractor(FeatureExtractorBase):
    """Wavelet-based feature extraction"""
    def __init__(self, input_dim, output_dim, wavelet='db4', levels=5):
        super().__init__(input_dim, output_dim)
        self.wavelet = wavelet
        self.levels = levels
        
        # 小波特征网络
        self.wavelet_nets = nn.ModuleDict({
            f'level_{i}': WaveletLevelNet(input_dim) 
            for i in range(levels)
        })
        
        total_features = sum(net.output_dim for net in self.wavelet_nets.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # 小波分解
        coeffs = self._wavelet_transform(x)
        
        # 提取各尺度特征
        features = []
        for i, (net_name, net) in enumerate(self.wavelet_nets.items()):
            feat = net(coeffs[i])
            features.append(feat)
            
        # 特征融合
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)
        
    def _wavelet_transform(self, x):
        """Perform wavelet transform"""
        # 转换为numpy进行小波变换
        x_np = x.cpu().numpy()
        coeffs_list = []
        
        for i in range(x_np.shape[0]):  # 批处理
            coeffs = pywt.wavedec(x_np[i], self.wavelet, level=self.levels)
            coeffs_list.append(coeffs)
            
        # 转回PyTorch tensor
        return [torch.tensor(np.array([c[i] for c in coeffs_list]), 
                           device=x.device)
                for i in range(len(coeffs_list[0]))]

class StatisticalFeatureExtractor(FeatureExtractorBase):
    """Statistical feature extraction"""
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        
        self.feature_nets = nn.ModuleDict({
            'moments': StatisticalMomentsNet(input_dim),
            'quantiles': QuantileNet(input_dim),
            'crossings': CrossingStatsNet(input_dim)
        })
        
        total_features = sum(net.output_dim for net in self.feature_nets.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        features = []
        for net in self.feature_nets.values():
            feat = net(x)
            features.append(feat)
            
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)
        
    def compute_importance(self, features, labels=None):
        """Compute statistical feature importance"""
        if labels is None:
            # 使用方差作为重要性度量
            return torch.var(features, dim=0)
        else:
            # 使用互信息作为重要性度量
            return self._mutual_information(features, labels)
            
    def _mutual_information(self, features, labels):
        """Compute mutual information between features and labels"""
        # 实现互信息计算
        return torch.tensor([
            self._estimate_mi(features[:, i], labels)
            for i in range(features.size(1))
        ]) 