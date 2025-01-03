import torch
import argparse
import yaml
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from models.positional_encoding import ContinuousPositionalEncoding
from models.multiscale_encoder import MultiScaleEncoder
from optimization.multi_objective import MultiObjectiveTrainer
from evaluation.performance_evaluator import PerformanceEvaluator
from visualization.analysis_visualizer import AnalysisVisualizer
from utils.config_manager import ConfigManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'analyze'])
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory for saving outputs')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = ConfigManager(args.config)
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'run.log'),
            logging.StreamHandler()
        ]
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 初始化模型和组件
    model = initialize_model(config).to(device)
    trainer = initialize_trainer(model, config, device)
    evaluator = PerformanceEvaluator(config, model, device)
    visualizer = AnalysisVisualizer(output_dir / 'visualizations')
    
    # 加载数据
    train_loader, val_loader, test_loader = load_data(config)
    
    if args.mode == 'train':
        train(model, trainer, evaluator, visualizer,
              train_loader, val_loader, config, output_dir)
    elif args.mode == 'test':
        test(model, evaluator, visualizer,
             test_loader, config, output_dir, args.checkpoint)
    else:
        analyze(model, evaluator, visualizer,
               test_loader, config, output_dir, args.checkpoint)

def initialize_model(config):
    """初始化模型及其组件"""
    model = AdaptivePreFormer(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    if config.model.pretrained:
        load_pretrained(model, config.model.pretrained_path)
        
    return model

def initialize_trainer(model, config, device):
    """初始化训练器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs
    )
    
    return MultiObjectiveTrainer(
        model=model,
        loss_fn=config.get_loss_function(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

def train(model, trainer, evaluator, visualizer,
          train_loader, val_loader, config, output_dir):
    """训练流程"""
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
        # 训练一个epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # 验证
        val_results = evaluator.evaluate(val_loader, epoch)
        val_loss = val_results['loss']
        
        # 记录日志
        logging.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'val_loss={val_loss:.4f}')
        
        # 保存检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, trainer, epoch, val_loss,
                          output_dir / 'best_model.pth')
        
        # 可视化当前结果
        visualizer.plot_training_progress(
            train_loss=train_loss,
            val_results=val_results,
            epoch=epoch
        )
        
        # 提前停止检查
        if trainer.should_stop():
            logging.info('Early stopping triggered')
            break

def test(model, evaluator, visualizer,
         test_loader, config, output_dir, checkpoint_path):
    """测试流程"""
    # 加载模型检查点
    load_checkpoint(model, checkpoint_path)
    
    # 运行测试评估
    test_results = evaluator.evaluate(test_loader, epoch='test')
    
    # 保存结果
    evaluator.export_results(output_dir / 'test_results.csv')
    
    # 可视化分析
    visualizer.create_interactive_dashboard(test_results)
    visualizer.plot_quality_analysis(test_results)
    
    logging.info(f'Test results: {test_results}')

def analyze(model, evaluator, visualizer,
           test_loader, config, output_dir, checkpoint_path):
    """分析流程"""
    # 加载模型检查点
    load_checkpoint(model, checkpoint_path)
    
    # 进行详细分析
    analysis_results = evaluator.detailed_analysis(test_loader)
    
    # 生成可视化
    visualizer.visualize_attention(analysis_results['attention_weights'])
    visualizer.visualize_features(analysis_results['features'])
    visualizer.plot_ablation_study(analysis_results['ablation_results'])
    
    # 导出分析报告
    visualizer.export_analysis_report(analysis_results)
    
    logging.info('Analysis completed')

if __name__ == '__main__':
    main() 