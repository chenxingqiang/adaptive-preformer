from AdaptivePreFormer.models.adaptive_preprocessor import LearnablePreprocessor
from .quality_assessor import HierarchicalQualityAssessor


class AdaptivePreFormer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.quality_assessor = HierarchicalQualityAssessor(input_dim)
        self.preprocessor = LearnablePreprocessor()
        
        # 特征提取和编码
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = QualityAwarePositionEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 任务头
        self.task_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # 质量评估
        quality_scores = self.quality_assessor(x)
        
        # 自适应预处理
        x = self.preprocessor(x, quality_scores['local_quality'])
        
        # 特征提取和位置编码
        x = self.input_proj(x)
        pos_encoding = self.pos_encoder(x, quality_scores['local_quality'])
        x = x + pos_encoding
        
        # Transformer编码
        x = self.transformer(x)
        
        # 任务预测
        return self.task_head(x.mean(dim=1))