import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

class MultiObjectiveLoss:
    """Multi-objective loss with dynamic weighting"""
    def __init__(self, loss_configs: Dict[str, dict]):
        self.losses = nn.ModuleDict({
            name: self._create_loss(config)
            for name, config in loss_configs.items()
        })
        
        # 动态权重
        self.log_weights = nn.Parameter(
            torch.zeros(len(loss_configs))
        )
        
        # 损失历史记录
        self.loss_history = {name: [] for name in loss_configs}
        
    def forward(self, predictions, targets, aux_data=None):
        """
        计算加权多目标损失
        """
        current_losses = {}
        weights = F.softmax(self.log_weights, dim=0)
        
        # 计算各项损失
        for i, (name, loss_fn) in enumerate(self.losses.items()):
            loss_val = loss_fn(predictions, targets, aux_data)
            current_losses[name] = loss_val
            self.loss_history[name].append(loss_val.item())
            
        # 加权组合
        total_loss = sum(weights[i] * loss 
                        for i, loss in enumerate(current_losses.values()))
                        
        return total_loss, current_losses, weights
        
    def _create_loss(self, config):
        """根据配置创建损失函数"""
        loss_type = config.get('type', 'mse')
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'reconstruction':
            return ReconstructionLoss(**config)
        elif loss_type == 'consistency':
            return ConsistencyLoss(**config)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
class ReconstructionLoss(nn.Module):
    """信号重建损失"""
    def __init__(self, alpha=1.0, use_stft=True):
        super().__init__()
        self.alpha = alpha
        self.use_stft = use_stft
        
    def forward(self, pred, target, aux_data=None):
        # 时域重建损失
        time_loss = F.mse_loss(pred, target)
        
        if self.use_stft:
            # 频域重建损失
            pred_stft = torch.stft(pred, n_fft=256, return_complex=True)
            target_stft = torch.stft(target, n_fft=256, return_complex=True)
            freq_loss = F.mse_loss(pred_stft.abs(), target_stft.abs())
            
            return time_loss + self.alpha * freq_loss
        
        return time_loss

class ConsistencyLoss(nn.Module):
    """特征一致性损失"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred, target, aux_data=None):
        if aux_data is None or 'features' not in aux_data:
            return torch.tensor(0.0, device=pred.device)
            
        feat1, feat2 = aux_data['features']
        
        # 归一化特征
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(feat1, feat2.T) / self.temperature
        
        # 对比损失
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

class MultiObjectiveTrainer:
    """多目标训练器"""
    def __init__(self, 
                 model: nn.Module,
                 loss_fn: MultiObjectiveLoss,
                 optimizer: torch.optim.Optimizer,
                 scheduler=None,
                 device='cuda'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 训练状态跟踪
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = []
        
        for batch in dataloader:
            # 准备数据
            inputs = self._prepare_inputs(batch)
            targets = self._prepare_targets(batch)
            aux_data = self._prepare_aux_data(batch)
            
            # 前向传播
            predictions = self.model(inputs)
            
            # 计算损失
            total_loss, losses, weights = self.loss_fn(
                predictions, targets, aux_data
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
            
        self.epoch += 1
        return np.mean(epoch_losses)
        
    def _prepare_inputs(self, batch):
        """准备输入数据"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return [b.to(self.device) for b in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
            
    def _prepare_targets(self, batch):
        """准备目标数据"""
        if not isinstance(batch, (tuple, list)):
            return None
        return batch[1].to(self.device)
        
    def _prepare_aux_data(self, batch):
        """准备辅助数据"""
        if not isinstance(batch, (tuple, list)) or len(batch) <= 2:
            return None
        return batch[2] 