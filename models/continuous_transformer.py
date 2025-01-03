import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ContinuousPositionalEncoding(nn.Module):
    """连续位置编码模块"""
    def __init__(self, 
                 d_model: int,
                 max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 可学习的尺度因子
        self.scale_factor = nn.Parameter(torch.ones(1))
        
        # 傅里叶特征投影
        self.freq_bands = nn.Parameter(torch.randn(d_model // 2))
        
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            timestamps: Optional timestamp tensor of shape [batch_size, seq_len]
        """
        if timestamps is None:
            timestamps = torch.arange(x.size(1), device=x.device)[None, :].float()
            
        # 计算傅里叶特征
        freqs = timestamps[:, :, None] * self.freq_bands[None, None, :]
        fourier_features = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        
        # 应用尺度因子
        pe = self.scale_factor * fourier_features
        
        return x + pe

class EfficientAttention(nn.Module):
    """高效注意力模块"""
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 local_context: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.local_context = local_context
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        H = self.n_heads
        
        # 投影查询、键、值
        q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, -1).transpose(1, 2)
        
        # 局部注意力
        local_attn = self._compute_local_attention(q, k, v, self.local_context)
        
        # 全局注意力(稀疏)
        global_attn = self._compute_global_attention(q, k, v)
        
        # 融合局部和全局注意力
        attn_out = local_attn + global_attn
        
        # 投影输出
        out = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, L, D))
        
        return self.dropout(out)
        
    def _compute_local_attention(self,
                               q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               context_size: int) -> torch.Tensor:
        """计算局部注意力"""
        B, H, L, D = q.shape
        
        # 构建局部注意力掩码
        local_mask = self._get_local_mask(L, context_size, q.device)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores.masked_fill(~local_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        return torch.matmul(attn_weights, v)
        
    def _compute_global_attention(self,
                                q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor) -> torch.Tensor:
        """计算全局注意力(使用LSH实现稀疏注意力)"""
        # TODO: 实现LSH稀疏注意力
        pass

class MultiScaleEncoder(nn.Module):
    """多尺度编码器"""
    def __init__(self,
                 d_model: int,
                 n_scales: int = 3,
                 scale_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        
        # 多尺度特征提取
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=2**i, stride=2**i),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )
            for i in range(n_scales)
        ])
        
        # 特征融合
        self.fusion = FeatureFusion(d_model, n_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取多尺度特征
        features = []
        for extractor in self.extractors:
            features.append(extractor(x.transpose(1, 2)).transpose(1, 2))
            
        # 融合特征
        return self.fusion(features)

class FeatureFusion(nn.Module):
    """特征融合模块"""
    def __init__(self,
                 d_model: int,
                 n_scales: int):
        super().__init__()
        self.d_model = d_model
        
        # 特征投影
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_scales)
        ])
        
        # 注意力融合
        self.attention = nn.MultiheadAttention(d_model, 8)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 投影特征
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # 对齐序列长度
        max_len = max(feat.size(1) for feat in projected)
        aligned = [
            F.pad(feat, (0, 0, 0, max_len - feat.size(1)))
            for feat in projected
        ]
        
        # 堆叠特征
        stacked = torch.stack(aligned, dim=0)
        
        # 注意力融合
        fused, _ = self.attention(stacked, stacked, stacked)
        
        return fused.mean(0)  # 平均池化

class ContinuousTransformer(nn.Module):
    """连续序列Transformer"""
    def __init__(self,
                 d_model: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # 连续位置编码
        self.pos_encoding = ContinuousPositionalEncoding(d_model)
        
        # 多尺度编码器
        self.multi_scale = MultiScaleEncoder(d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # 输出层范数
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 应用位置编码
        x = self.pos_encoding(x, timestamps)
        
        # 多尺度特征提取
        x = self.multi_scale(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, src_mask=mask)
            
        return self.norm(x) 