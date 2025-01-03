import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import seaborn as sns

class ModelAnalyzer:
    """Analysis tools for model performance and behavior"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def profile_performance(self, input_data):
        """Profile model performance"""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                _ = self.model(input_data)
                
        return prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        )
        
    def analyze_attention_patterns(self, outputs):
        """Analyze attention patterns"""
        attention_maps = outputs['attention_weights']
        
        # Compute attention statistics
        sparsity = (attention_maps < 0.01).float().mean()
        entropy = -(attention_maps * torch.log(attention_maps + 1e-10)).sum(-1).mean()
        
        return {
            'sparsity': sparsity.item(),
            'entropy': entropy.item(),
            'attention_maps': attention_maps.detach().cpu()
        }
        
    def analyze_quality_distribution(self, outputs):
        """Analyze quality score distribution"""
        quality_scores = outputs['quality_scores'].detach().cpu()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(quality_scores.numpy().flatten(), bins=50)
        plt.title('Distribution of Quality Scores')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        
        return plt.gcf()
        
    def compute_efficiency_metrics(self, batch_size=32, seq_length=1000):
        """Compute efficiency metrics"""
        input_size = (batch_size, seq_length, self.model.input_dim)
        input_data = torch.randn(input_size).to(self.device)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            _ = self.model(input_data)
        inference_time = time.time() - start_time
        
        # Measure memory usage
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = self.model(input_data)
        max_memory = torch.cuda.max_memory_allocated()
        
        return {
            'inference_time_ms': inference_time * 1000,
            'memory_usage_mb': max_memory / (1024 * 1024),
            'samples_per_second': batch_size / inference_time
        } 