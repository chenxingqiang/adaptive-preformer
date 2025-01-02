import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from dataset import EEGDataset
from model import AdaptivePreFormer
from trainer import Trainer, TrainingConfig, Evaluator

def main():
    # 配置
    config = TrainingConfig()
    config.batch_size = 32
    config.num_epochs = 100
    config.learning_rate = 1e-4
    
    # 数据集路径
    data_dir = Path("data/PhysioNetP300")
    
    # 加载数据集
    full_dataset = EEGDataset(data_dir, sequence_length=512)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 模型初始化
    model = AdaptivePreFormer(
        input_dim=64,  # EEG通道数
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    # 训练器和评估器
    trainer = Trainer(model, config)
    evaluator = Evaluator(model, config.device)
    
    # 训练循环
    best_acc = 0
    for epoch in range(config.num_epochs):
        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # 评估
        eval_results = evaluator.evaluate(val_loader)
        
        # 保存最佳模型
        if eval_results['accuracy'] > best_acc:
            best_acc = eval_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'best_model.pth')
            
        # 记录结果
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {eval_results["loss"]:.4f}, Val Acc: {eval_results["accuracy"]:.4f}')
        print('Quality Analysis:', eval_results['quality_analysis'])
        print('Parameter Analysis:', eval_results['param_analysis'])

if __name__ == '__main__':
    # 下载并准备数据
    from prepare_data import EEGDataDownloader
    downloader = EEGDataDownloader()
    data_dir = downloader.download_physionet_p300()
    
    # 开始训练
    main()