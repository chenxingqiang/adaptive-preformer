import torch


class HierarchicalQualityAssessor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.local_assessor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )
        
        self.global_assessor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, input_dim, sequence_length]
        local_quality = self.local_assessor(x)  # [batch_size, 1, sequence_length]
        global_quality = self.global_assessor(x)  # [batch_size, 1]
        
        return {
            'local_quality': local_quality.squeeze(1),
            'global_quality': global_quality.squeeze(1)
        }
    

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()
        quality_scores = []
        preprocess_params = []
        
        for data, target in tqdm(val_loader, desc='Evaluating'):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            
            # 收集评估指标
            loss = self.criterion(
                output['predictions'],
                target,
                output['quality_scores'],
                output['preprocess_params']
            )
            acc = (output['predictions'].argmax(dim=1) == target).float().mean()
            
            losses.update(loss.item())
            accuracies.update(acc.item())
            
            # 收集质量分数和预处理参数
            quality_scores.append(output['quality_scores'])
            preprocess_params.append(output['preprocess_params'])
            
        # 分析质量评估和预处理参数
        quality_analysis = self.analyze_quality_scores(torch.cat(quality_scores))
        param_analysis = self.analyze_preprocess_params(torch.cat(preprocess_params))
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'quality_analysis': quality_analysis,
            'param_analysis': param_analysis
        }
        
    def analyze_quality_scores(self, scores):
        return {
            'mean': scores.mean().item(),
            'std': scores.std().item(),
            'min': scores.min().item(),
            'max': scores.max().item()
        }
        
    def analyze_preprocess_params(self, params):
        return {
            'sparsity': (params.abs() < 1e-3).float().mean().item(),
            'magnitude': params.abs().mean().item()
        }