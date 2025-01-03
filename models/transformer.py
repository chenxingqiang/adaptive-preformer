import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_attention import EfficientLongSequenceAttention
from .positional_encoding import AdaptiveFourierPositionalEncoding

class AdaptivePreFormer(nn.Module):
    """Adaptive Transformer for preprocessing and feature learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = AdaptiveFourierPositionalEncoding(
            config.d_model,
            config.max_seq_length
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Quality assessment
        self.quality_assessor = QualityAssessor(config.d_model)
        
        # Output heads
        self.task_heads = nn.ModuleDict()
        
    def forward(self, x):
        # Initial quality assessment
        quality_scores, quality_features = self.quality_assessor(x)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoder(x, quality_scores)
        x = x + pos_enc
        
        # Process through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, quality_scores)
            attention_weights.append(attn)
            
        return {
            'features': x,
            'quality_scores': quality_scores,
            'attention_weights': attention_weights,
            'quality_features': quality_features
        }
        
    def get_stats(self):
        """Get model statistics for loss computation"""
        return {
            'compute_stats': self.get_compute_stats(),
            'memory_stats': self.get_memory_stats()
        }
        
class TransformerLayer(nn.Module):
    """Single transformer layer with efficient attention"""
    def __init__(self, config):
        super().__init__()
        
        # Attention
        self.attention = EfficientLongSequenceAttention(
            config.d_model,
            config.nhead
        )
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, quality_scores=None):
        # Attention
        attended, attention_weights = self.attention(
            self.norm1(x),
            quality_scores
        )
        x = x + self.dropout(attended)
        
        # Feed forward
        x = x + self.dropout(
            self.feed_forward(self.norm2(x))
        )
        
        return x, attention_weights