import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import AverageMeter


class TrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 100
        self.weight_decay = 0.01
        self.steps_per_epoch = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gradient_clip = 1.0


class AdaptivePreFormerLoss(nn.Module):
    def __init__(self, quality_weight=0.1, param_weight=0.01):
        super().__init__()
        self.quality_weight = quality_weight
        self.param_weight = param_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, quality_scores, preprocess_params):
        # 主任务损失
        task_loss = self.ce_loss(predictions, targets)
        
        # 质量评估损失
        quality_loss = torch.mean(torch.abs(quality_scores))
        
        # 参数正则化损失
        param_loss = torch.mean(torch.abs(preprocess_params))
        
        return task_loss + self.quality_weight * quality_loss + self.param_weight * param_loss


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        
        # 优化器设置
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=config.steps_per_epoch
        )
        
        self.criterion = AdaptivePreFormerLoss()
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Compute loss using logits
            loss = self.criterion(
                output['logits'],  # Use logits here
                target,
                output['quality_scores'],
                output['preprocess_params']
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            
            # Compute accuracy
            acc = (output['predictions'] == target).float().mean()
            
            # Update statistics
            losses.update(loss.item())
            accuracies.update(acc.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
            
        return losses.avg, accuracies.avg
    

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