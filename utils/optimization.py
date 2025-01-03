import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
import copy

class ModelOptimizer:
    """Tools for model optimization and compression"""
    def __init__(self, model):
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())
        
    def quantize_model(self, dtype='int8'):
        """Quantize model weights"""
        quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv1d},
            dtype=getattr(torch, dtype)
        )
        return quantized_model
        
    def prune_model(self, amount=0.3):
        """Prune model weights"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                prune.l1_unstructured(module, 'weight', amount=amount)
                prune.remove(module, 'weight')
                
    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.model.eval()
        
        # Fuse batch norm layers
        torch.quantization.fuse_modules(self.model, ['conv', 'bn', 'relu'])
        
        # Optimize memory layout
        self.model = torch.jit.script(self.model)
        
        return self.model
        
    def restore_original(self):
        """Restore model to original state"""
        self.model.load_state_dict(self.original_state)
        
    def benchmark_model(self, input_size, num_runs=100):
        """Benchmark model performance"""
        device = next(self.model.parameters()).device
        x = torch.randn(input_size).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(x)
                
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = self.model(x)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
                
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        } 