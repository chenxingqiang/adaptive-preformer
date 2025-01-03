import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List
import time
import logging
from torch.utils.tensorboard import SummaryWriter

class PerformanceEvaluator:
    """Comprehensive performance evaluation system"""
    def __init__(self, config, model, device='cuda'):
        self.config = config
        self.model = model
        self.device = device
        
        # 指标记录器
        self.metrics = {
            'accuracy': [],
            'f1_score': [],
            'auc_roc': [],
            'latency': [],
            'memory': [],
            'gpu_util': []
        }
        
        # TensorBoard记录器
        self.writer = SummaryWriter('runs/experiment')
        
        # 性能监控器
        self.latency_monitor = LatencyProfiler()
        self.memory_monitor = MemoryProfiler()
        self.gpu_monitor = GPUProfiler() if torch.cuda.is_available() else None
        
    def evaluate(self, dataloader, epoch):
        """评估一个epoch的性能"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 性能监控开始
                self.latency_monitor.start()
                self.memory_monitor.start()
                if self.gpu_monitor:
                    self.gpu_monitor.start()
                
                # 模型推理
                inputs = self._prepare_inputs(batch)
                outputs = self.model(inputs)
                
                # 性能监控结束
                latency = self.latency_monitor.end()
                memory = self.memory_monitor.end()
                gpu_util = self.gpu_monitor.end() if self.gpu_monitor else 0
                
                # 收集预测结果
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch[1].cpu().numpy())
                
                # 记录性能指标
                self.metrics['latency'].append(latency)
                self.metrics['memory'].append(memory)
                self.metrics['gpu_util'].append(gpu_util)
        
        # 计算评估指标
        results = self._compute_metrics(predictions, targets)
        
        # 记录到TensorBoard
        self._log_metrics(results, epoch)
        
        return results
        
    def _compute_metrics(self, predictions, targets):
        """计算评估指标"""
        results = {
            'accuracy': accuracy_score(targets, np.argmax(predictions, axis=1)),
            'f1_score': f1_score(targets, np.argmax(predictions, axis=1), average='weighted'),
            'auc_roc': roc_auc_score(targets, predictions, multi_class='ovr'),
            'avg_latency': np.mean(self.metrics['latency']),
            'p95_latency': np.percentile(self.metrics['latency'], 95),
            'peak_memory': max(self.metrics['memory']),
            'avg_gpu_util': np.mean(self.metrics['gpu_util'])
        }
        return results
        
    def _log_metrics(self, results, epoch):
        """记录指标到TensorBoard"""
        for name, value in results.items():
            self.writer.add_scalar(f'metrics/{name}', value, epoch)
            
    def analyze_efficiency(self):
        """分析计算效率"""
        efficiency_stats = {
            'throughput': len(self.metrics['latency']) / sum(self.metrics['latency']),
            'memory_efficiency': np.mean(self.metrics['memory']) / torch.cuda.get_device_properties(0).total_memory,
            'gpu_efficiency': np.mean(self.metrics['gpu_util']) / 100.0
        }
        return efficiency_stats
        
    def export_results(self, save_path):
        """导出评估结果"""
        import pandas as pd
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'metric': list(self.metrics.keys()),
            'mean': [np.mean(values) for values in self.metrics.values()],
            'std': [np.std(values) for values in self.metrics.values()],
            'min': [np.min(values) for values in self.metrics.values()],
            'max': [np.max(values) for values in self.metrics.values()]
        })
        
        # 保存结果
        results_df.to_csv(save_path, index=False)
        
class LatencyProfiler:
    """延迟分析器"""
    def __init__(self):
        self.start_time = None
        
    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        
    def end(self):
        torch.cuda.synchronize()
        return (time.perf_counter() - self.start_time) * 1000  # 转换为毫秒

class MemoryProfiler:
    """内存分析器"""
    def __init__(self):
        self.start_memory = None
        
    def start(self):
        torch.cuda.synchronize()
        self.start_memory = torch.cuda.memory_allocated()
        
    def end(self):
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        return (end_memory - self.start_memory) / 1024**2  # 转换为MB

class GPUProfiler:
    """GPU利用率分析器"""
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except:
            self.enabled = False
            
    def start(self):
        if not self.enabled:
            return
        self.start_util = self._get_gpu_util()
        
    def end(self):
        if not self.enabled:
            return 0
        end_util = self._get_gpu_util()
        return (self.start_util + end_util) / 2
        
    def _get_gpu_util(self):
        import pynvml
        return pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu 