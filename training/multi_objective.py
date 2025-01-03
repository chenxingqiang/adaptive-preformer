import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from pathlib import Path
import json

class MultiObjectiveTrainer:
    """多目标训练器"""
    def __init__(self,
                 model: nn.Module,
                 preprocessor: nn.Module,
                 feature_extractor: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda',
                 log_dir: Optional[str] = None):
        self.model = model.to(device)
        self.preprocessor = preprocessor.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 损失权重
        self.loss_weights = {
            'task': 1.0,
            'quality': 0.1,
            'efficiency': 0.01
        }
        
        # 评估器
        self.quality_evaluator = QualityEvaluator()
        self.feature_analyzer = FeatureAnalyzer()
        self.performance_evaluator = PerformanceEvaluator()
        
        # 日志设置
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True)
            wandb.init(project="adaptive-preformer", dir=str(self.log_dir))
            
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   epoch: int) -> Dict[str, float]:
        self.model.train()
        self.preprocessor.train()
        self.feature_extractor.train()
        
        total_loss = 0
        metrics = defaultdict(float)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            processed_data, preprocess_info = self.preprocessor(data)
            features = self.feature_extractor(processed_data)
            output = self.model(features)
            
            # 计算多个损失
            losses = self._compute_losses(output, target, preprocess_info, features)
            total_loss = sum(w * l for l, w in zip(losses.values(), self.loss_weights.values()))
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 更新指标
            metrics['total_loss'] += total_loss.item()
            for k, v in losses.items():
                metrics[f'{k}_loss'] += v.item()
                
            # 记录日志
            if batch_idx % 100 == 0:
                self._log_training_step(epoch, batch_idx, len(dataloader), metrics)
                
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {k: v / len(dataloader) for k, v in metrics.items()}
        
    def evaluate(self, 
                dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.preprocessor.eval()
        self.feature_extractor.eval()
        
        metrics = defaultdict(float)
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                processed_data, preprocess_info = self.preprocessor(data)
                features = self.feature_extractor(processed_data)
                output = self.model(features)
                
                # 收集预测结果
                all_outputs.append(output)
                all_targets.append(target)
                
                # 评估质量
                quality_metrics = self.quality_evaluator(processed_data, data)
                metrics.update(quality_metrics)
                
                # 分析特征
                feature_metrics = self.feature_analyzer(features)
                metrics.update(feature_metrics)
                
                # 评估性能
                perf_metrics = self.performance_evaluator(
                    self.model, self.preprocessor, self.feature_extractor
                )
                metrics.update(perf_metrics)
                
        # 计算总体指标
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        metrics.update(self._compute_metrics(all_outputs, all_targets))
        
        return metrics
        
    def _compute_losses(self,
                       output: torch.Tensor,
                       target: torch.Tensor,
                       preprocess_info: Dict[str, torch.Tensor],
                       features: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # 任务损失
        losses['task'] = F.cross_entropy(output, target)
        
        # 质量损失
        losses['quality'] = self._compute_quality_loss(preprocess_info)
        
        # 效率损失
        losses['efficiency'] = self._compute_efficiency_loss(features)
        
        return losses
        
    def _compute_quality_loss(self,
                            preprocess_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 基于预处理信息计算质量损失
        quality_score = preprocess_info['quality_score']
        noise_mask = preprocess_info['noise_mask']
        
        # 鼓励高质量分数
        quality_loss = 1 - quality_score.mean()
        
        # 鼓励合理的噪声检测
        noise_loss = -torch.mean(
            noise_mask * torch.log(noise_mask + 1e-8) +
            (1 - noise_mask) * torch.log(1 - noise_mask + 1e-8)
        )
        
        return quality_loss + 0.1 * noise_loss
        
    def _compute_efficiency_loss(self,
                               features: torch.Tensor) -> torch.Tensor:
        # 计算特征的稀疏性损失
        sparsity_loss = torch.mean(torch.abs(features))
        
        # 计算特征的冗余性损失
        redundancy_loss = self._compute_feature_redundancy(features)
        
        return sparsity_loss + 0.1 * redundancy_loss
        
    @staticmethod
    def _compute_feature_redundancy(features: torch.Tensor) -> torch.Tensor:
        # 计算特征间的相关性
        features_flat = features.view(features.size(0), -1)
        corr_matrix = torch.corrcoef(features_flat.T)
        
        # 惩罚高相关性
        redundancy = torch.mean(torch.abs(corr_matrix - torch.eye(corr_matrix.size(0), device=corr_matrix.device)))
        
        return redundancy
        
    def _log_training_step(self,
                          epoch: int,
                          batch_idx: int,
                          total_batches: int,
                          metrics: Dict[str, float]) -> None:
        if not hasattr(self, 'log_dir'):
            return
            
        # 计算平均指标
        avg_metrics = {k: v / (batch_idx + 1) for k, v in metrics.items()}
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'progress': batch_idx / total_batches,
            **avg_metrics
        })
        
        # 保存到文件
        metrics_file = self.log_dir / f'metrics_epoch_{epoch}.json'
        with open(metrics_file, 'w') as f:
            json.dump(avg_metrics, f, indent=2) 