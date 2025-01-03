import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from scipy import signal
from sklearn.metrics import roc_auc_score
import librosa

class QualityEvaluator:
    """信号质量评估器"""
    def __init__(self):
        self.metrics = {
            'snr': self._compute_snr,
            'entropy': self._compute_entropy,
            'clarity': self._compute_clarity,
            'stability': self._compute_stability
        }
        
    def __call__(self, processed: torch.Tensor, original: torch.Tensor) -> Dict[str, float]:
        """评估信号质量"""
        results = {}
        for name, metric_fn in self.metrics.items():
            results[f'quality_{name}'] = metric_fn(processed, original)
        return results
        
    def _compute_snr(self, processed: torch.Tensor, original: torch.Tensor) -> float:
        """计算信噪比"""
        noise = processed - original
        signal_power = torch.mean(original ** 2, dim=-1)
        noise_power = torch.mean(noise ** 2, dim=-1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return float(torch.mean(snr).item())
        
    def _compute_entropy(self, processed: torch.Tensor, original: torch.Tensor) -> float:
        """计算信号熵"""
        hist = torch.histogram(processed, bins=50)[0]
        prob = hist / torch.sum(hist)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-8))
        return float(entropy.item())
        
    def _compute_clarity(self, processed: torch.Tensor, original: torch.Tensor) -> float:
        """计算信号清晰度"""
        # 使用频谱对比度作为清晰度指标
        spec = torch.stft(processed, n_fft=256, return_complex=True)
        mag_spec = torch.abs(spec)
        clarity = torch.mean(torch.max(mag_spec, dim=-1)[0] / (torch.mean(mag_spec, dim=-1) + 1e-8))
        return float(clarity.item())
        
    def _compute_stability(self, processed: torch.Tensor, original: torch.Tensor) -> float:
        """计算信号稳定性"""
        diff = torch.diff(processed, dim=-1)
        stability = 1.0 / (torch.std(diff) + 1e-8)
        return float(stability.item())

class FeatureAnalyzer:
    """特征分析器"""
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        
    def __call__(self, features: torch.Tensor) -> Dict[str, float]:
        """分析特征质量"""
        results = {}
        
        # 特征相关性
        results['feature_correlation'] = self._compute_feature_correlation(features)
        
        # 特征重要性
        results['feature_importance'] = self._compute_feature_importance(features)
        
        # 特征冗余度
        results['feature_redundancy'] = self._compute_feature_redundancy(features)
        
        # 特征稳定性
        results['feature_stability'] = self._compute_feature_stability(features)
        
        return results
        
    def _compute_feature_correlation(self, features: torch.Tensor) -> float:
        """计算特征间相关性"""
        features_flat = features.view(features.size(0), -1)
        corr_matrix = torch.corrcoef(features_flat.T)
        return float(torch.mean(torch.abs(corr_matrix - torch.eye(corr_matrix.size(0), device=corr_matrix.device))).item())
        
    def _compute_feature_importance(self, features: torch.Tensor) -> float:
        """计算特征重要性"""
        variance = torch.var(features, dim=0)
        importance = torch.mean(torch.sort(variance, descending=True)[0][:self.n_components])
        return float(importance.item())
        
    def _compute_feature_redundancy(self, features: torch.Tensor) -> float:
        """计算特征冗余度"""
        # 使用PCA估计冗余度
        features_flat = features.view(features.size(0), -1)
        U, S, V = torch.svd(features_flat)
        redundancy = 1.0 - (torch.sum(S[:self.n_components]) / torch.sum(S))
        return float(redundancy.item())
        
    def _compute_feature_stability(self, features: torch.Tensor) -> float:
        """计算特征稳定性"""
        stability = 1.0 / (torch.std(features, dim=0).mean() + 1e-8)
        return float(stability.item())

class PerformanceEvaluator:
    """性能评估器"""
    def __init__(self):
        self.metrics = {
            'latency': self._measure_latency,
            'memory': self._measure_memory,
            'throughput': self._measure_throughput,
            'efficiency': self._measure_efficiency
        }
        
    def __call__(self, 
                 model: nn.Module,
                 preprocessor: nn.Module,
                 feature_extractor: nn.Module) -> Dict[str, float]:
        """评估模型性能"""
        results = {}
        for name, metric_fn in self.metrics.items():
            results[f'performance_{name}'] = metric_fn(model, preprocessor, feature_extractor)
        return results
        
    def _measure_latency(self, 
                        model: nn.Module,
                        preprocessor: nn.Module,
                        feature_extractor: nn.Module) -> float:
        """测量处理延迟"""
        batch_size = 32
        seq_length = 1000
        device = next(model.parameters()).device
        
        # 生成测试数据
        x = torch.randn(batch_size, seq_length, device=device)
        
        # 测量时间
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            processed, _ = preprocessor(x)
            features = feature_extractor(processed)
            _ = model(features)
        end_time.record()
        
        torch.cuda.synchronize()
        return start_time.elapsed_time(end_time)
        
    def _measure_memory(self,
                       model: nn.Module,
                       preprocessor: nn.Module,
                       feature_extractor: nn.Module) -> float:
        """测量内存使用"""
        torch.cuda.reset_peak_memory_stats()
        
        batch_size = 32
        seq_length = 1000
        device = next(model.parameters()).device
        
        # 运行推理
        x = torch.randn(batch_size, seq_length, device=device)
        with torch.no_grad():
            processed, _ = preprocessor(x)
            features = feature_extractor(processed)
            _ = model(features)
            
        return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
    def _measure_throughput(self,
                          model: nn.Module,
                          preprocessor: nn.Module,
                          feature_extractor: nn.Module) -> float:
        """测量吞吐量"""
        batch_size = 32
        seq_length = 1000
        n_iterations = 100
        device = next(model.parameters()).device
        
        # 预热
        x = torch.randn(batch_size, seq_length, device=device)
        with torch.no_grad():
            for _ in range(10):
                processed, _ = preprocessor(x)
                features = feature_extractor(processed)
                _ = model(features)
                
        # 测量吞吐量
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            for _ in range(n_iterations):
                processed, _ = preprocessor(x)
                features = feature_extractor(processed)
                _ = model(features)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        return (n_iterations * batch_size) / (elapsed_time / 1000)  # samples/second
        
    def _measure_efficiency(self,
                          model: nn.Module,
                          preprocessor: nn.Module,
                          feature_extractor: nn.Module) -> float:
        """测量计算效率"""
        # 计算FLOPs
        batch_size = 1
        seq_length = 1000
        device = next(model.parameters()).device
        
        x = torch.randn(batch_size, seq_length, device=device)
        flops = 0
        
        def count_flops(module, input, output):
            nonlocal flops
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                flops += np.prod(input[0].shape) * np.prod(module.weight.shape)
                
        # 注册钩子
        hooks = []
        for module in [model, preprocessor, feature_extractor]:
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    hooks.append(m.register_forward_hook(count_flops))
                    
        # 运行推理
        with torch.no_grad():
            processed, _ = preprocessor(x)
            features = feature_extractor(processed)
            _ = model(features)
            
        # 移除钩子
        for hook in hooks:
            hook.remove()
            
        return float(flops) 