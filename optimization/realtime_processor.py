import torch
import numpy as np
from collections import deque
import time
from threading import Lock

class CircularBuffer:
    """Thread-safe circular buffer for real-time processing"""
    def __init__(self, max_size, dim):
        self.buffer = torch.zeros(max_size, dim)
        self.max_size = max_size
        self.head = 0
        self.size = 0
        self.lock = Lock()
        
    def push(self, data):
        with self.lock:
            batch_size = data.size(0)
            indices = torch.arange(self.head, self.head + batch_size) % self.max_size
            self.buffer[indices] = data
            self.head = (self.head + batch_size) % self.max_size
            self.size = min(self.size + batch_size, self.max_size)
            
    def get_latest(self, n):
        with self.lock:
            if n > self.size:
                return None
            start_idx = (self.head - n) % self.max_size
            if start_idx < self.head:
                return self.buffer[start_idx:self.head]
            else:
                return torch.cat([
                    self.buffer[start_idx:],
                    self.buffer[:self.head]
                ])

class IncrementalUpdate:
    """Incremental feature and state updates"""
    def __init__(self, model, update_freq=10):
        self.model = model
        self.update_freq = update_freq
        self.cached_states = {}
        self.update_count = 0
        
    def update(self, new_data):
        self.update_count += 1
        
        # 增量特征更新
        new_features = self.model.extract_features(new_data)
        
        # 状态更新
        if self.update_count % self.update_freq == 0:
            self._update_states(new_features)
            
        return new_features
        
    def _update_states(self, new_features):
        # 更新缓存的状态
        for name, state in self.cached_states.items():
            if isinstance(state, torch.Tensor):
                alpha = 0.9  # 衰减因子
                self.cached_states[name] = (
                    alpha * state + (1 - alpha) * new_features
                )

class LatencyMonitor:
    """Monitor and analyze processing latency"""
    def __init__(self, window_size=100):
        self.latencies = deque(maxlen=window_size)
        self.start_time = None
        
    def start(self):
        self.start_time = time.perf_counter()
        
    def stop(self):
        if self.start_time is None:
            return
        latency = (time.perf_counter() - self.start_time) * 1000  # ms
        self.latencies.append(latency)
        self.start_time = None
        
    def get_stats(self):
        latencies = np.array(self.latencies)
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }

class MemoryProfiler:
    """Profile memory usage and efficiency"""
    def __init__(self):
        self.memory_stats = []
        
    def profile(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB
            
            self.memory_stats.append({
                'allocated': allocated,
                'cached': cached,
                'timestamp': time.time()
            })
            
    def get_stats(self):
        if not self.memory_stats:
            return {}
            
        stats = np.array([(s['allocated'], s['cached']) 
                         for s in self.memory_stats])
        
        return {
            'mean_allocated': np.mean(stats[:, 0]),
            'peak_allocated': np.max(stats[:, 0]),
            'mean_cached': np.mean(stats[:, 1]),
            'peak_cached': np.max(stats[:, 1])
        } 