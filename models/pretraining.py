from AdaptivePreFormer.models.transformer import AdaptivePreFormer


class AdaptivePreFormerPretraining(AdaptivePreFormer):
    def __init__(self, *args, pretraining_tasks, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add pretraining heads
        self.pretraining_heads = nn.ModuleDict({
            'reconstruction': ReconstructionHead(self.d_model),
            'temporal_prediction': TemporalPredictionHead(self.d_model),
            'contrastive': ContrastiveHead(self.d_model)
        })
    
    def forward_pretraining(self, x, task):
        # Get base features
        features = self.get_features(x)
        
        # Apply task-specific head
        return self.pretraining_heads[task](features) 