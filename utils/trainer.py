import torch


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
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 计算损失
            loss = self.criterion(
                output['predictions'], 
                target,
                output['quality_scores'],
                output['preprocess_params']
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            
            # 计算准确率
            acc = (output['predictions'].argmax(dim=1) == target).float().mean()
            
            # 更新统计
            losses.update(loss.item())
            accuracies.update(acc.item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
            
        return losses.avg, accuracies.avg