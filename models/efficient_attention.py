class EfficientLongSequenceAttention(nn.Module):
    """Efficient attention mechanism for long sequences"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Local attention
        self.local_attention = LocalSelfAttention(d_model, n_heads, window_size=256)
        
        # Global attention with memory compression
        self.global_memory = nn.Parameter(torch.randn(1, 64, d_model))
        self.memory_attention = MemoryAttention(d_model, n_heads)
        
        # Sparse attention
        self.sparse_attention = SparseSelfAttention(d_model, n_heads, sparsity=0.9)
        
        # Attention fusion
        self.fusion_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, x, mask=None):
        # Local attention
        local_out = self.local_attention(x, mask)
        
        # Global memory attention
        memory_out = self.memory_attention(x, self.global_memory)
        
        # Sparse attention
        sparse_out = self.sparse_attention(x, mask)
        
        # Fusion with learned weights
        weights = F.softmax(self.fusion_weights, dim=0)
        out = (weights[0] * local_out + 
               weights[1] * memory_out + 
               weights[2] * sparse_out)
               
        return out 