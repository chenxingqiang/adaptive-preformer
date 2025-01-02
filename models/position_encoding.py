import math
import torch


class QualityAwarePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 基础位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 质量嵌入
        self.quality_embedder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, quality_scores):
        # x: [batch_size, seq_len, d_model]
        # quality_scores: [batch_size, seq_len]
        pe = self.pe[:x.size(1)]
        quality_embedding = self.quality_embedder(quality_scores.unsqueeze(-1))
        
        # 质量感知的位置编码
        return pe * quality_embedding