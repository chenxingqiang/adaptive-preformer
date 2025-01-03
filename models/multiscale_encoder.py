import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleEncoder(nn.Module):
    """Multi-scale feature encoder with adaptive pooling"""
    def __init__(self, input_dim, hidden_dims=[64, 128, 256], pool_sizes=[4, 8, 16]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.pool_sizes = pool_sizes
        
        # 多尺度编码器
        self.encoders = nn.ModuleList([
            ScaleEncoder(
                input_dim if i == 0 else hidden_dims[i-1],
                hidden_dims[i],
                pool_size
            )
            for i, pool_size in enumerate(pool_sizes)
        ])
        
        # 特征融合
        self.fusion = CrossScaleAttention(
            hidden_dims,
            num_heads=4
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, L, D)
        Returns:
            Multi-scale features and attention weights
        """
        # 多尺度特征提取
        scale_features = []
        for encoder in self.encoders:
            scale_feat = encoder(x)
            scale_features.append(scale_feat)
            
        # 跨尺度特征融合
        fused_features, attention_weights = self.fusion(scale_features)
        
        return fused_features, attention_weights

class ScaleEncoder(nn.Module):
    """Single-scale feature encoder"""
    def __init__(self, input_dim, hidden_dim, pool_size):
        super().__init__()
        self.pool_size = pool_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 自适应池化
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        
    def forward(self, x):
        # 转换维度 (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        
        # 特征提取
        x = self.conv(x)
        
        # 自适应池化
        x = self.pool(x)
        
        # 转回 (B, L, D)
        return x.transpose(1, 2)

class CrossScaleAttention(nn.Module):
    """Cross-scale feature fusion with attention"""
    def __init__(self, hidden_dims, num_heads=4):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        
        # 跨尺度注意力
        self.attention = nn.MultiheadAttention(
            sum(hidden_dims),
            num_heads,
            dropout=0.1
        )
        
        # 输出投影
        self.output_proj = nn.Linear(
            sum(hidden_dims),
            hidden_dims[-1]
        )
        
    def forward(self, scale_features):
        # 拼接多尺度特征
        concat_features = torch.cat(scale_features, dim=-1)
        
        # 应用跨尺度注意力
        attn_output, attn_weights = self.attention(
            concat_features,
            concat_features,
            concat_features
        )
        
        # 输出投影
        output = self.output_proj(attn_output)
        
        return output, attn_weights 