import torch
import torch.nn as nn
import math

class QualityAwarePositionEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the model
            dropout: Dropout value
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)  # Fixed: using int for max_len
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved and restored)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)