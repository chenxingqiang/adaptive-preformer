import torch
import logging
from pathlib import Path
from datetime import datetime
import json
import wandb

class ExperimentManager:
    """Manages experiment execution and logging"""
    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize metrics
        self.metrics = PerformanceMetrics()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb
        self.setup_wandb()
        
        # Model analyzer
        self.analyzer = ModelAnalyzer(model, config.device)
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.log_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        
    def setup_wandb(self):
        """Initialize wandb tracking"""
        wandb.init(
            project="adaptive-preformer",
            config=self.config.__dict__,
            name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
    def run_experiment(self):
        """Execute complete experiment pipeline"""
        logging.info("Starting experiment...")
        
        # Initial model analysis
        self.run_initial_analysis()
        
        # Training loop
        best_metric = float('inf')
        for epoch in range(self.config.training.num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Analysis
            analysis_results = self.analyze_epoch(epoch)
            
            # Log results
            self.log_epoch_results(epoch, train_metrics, val_metrics, analysis_results)
            
            # Save checkpoint
            if val_metrics['loss'] < best_metric:
                best_metric = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics)
                
        # Final analysis
        self.run_final_analysis()
        
    def run_initial_analysis(self):
        """Initial model and data analysis"""
        logging.info("Running initial analysis...")
        
        # Analyze model architecture
        model_stats = self.analyzer.compute_efficiency_metrics()
        logging.info(f"Model efficiency metrics: {model_stats}")
        
        # Analyze data characteristics
        data_stats = self.analyze_data()
        logging.info(f"Data statistics: {data_stats}")
        
        # Quality assessment
        quality_stats = self.analyze_data_quality()
        logging.info(f"Data quality metrics: {quality_stats}")
        
        # Save initial analysis
        self.save_analysis_results({
            'model_stats': model_stats,
            'data_stats': data_stats,
            'quality_stats': quality_stats
        })
        
    def analyze_data(self):
        """Analyze dataset characteristics"""
        stats = {
            'train_size': len(self.train_loader.dataset),
            'val_size': len(self.val_loader.dataset),
            'num_classes': self.model.num_classes,
            'input_dim': self.model.input_dim,
            'seq_length': self.config.model.max_seq_length
        }
        return stats
        
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.log_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}") 
        
    def analyze_data_quality(self):
        """Analyze data quality"""
        quality_scores = []
        artifacts = []
        
        for batch in self.train_loader:
            with torch.no_grad():
                quality_score, components = self.model.quality_assessor(batch)
                quality_scores.append(quality_score.cpu())
                artifacts.append(components.cpu())
                
        return {
            'mean_quality': torch.cat(quality_scores).mean().item(),
            'quality_std': torch.cat(quality_scores).std().item(),
            'artifact_types': self.analyze_artifacts(torch.cat(artifacts))
        } 