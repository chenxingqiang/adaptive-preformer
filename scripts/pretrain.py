import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from models.pretraining_tasks import ReconstructionTask, TemporalPredictionTask, ContrastiveTask
from data.pretraining_dataset import PretrainingDataset
from models.transformer import AdaptivePreFormerPretraining
from data.augmentations import AugmentationPipeline
from utils.training import EarlyStopping, PretrainingValidator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pretraining.log'),
            logging.StreamHandler()
        ]
    )

def pretrain():
    # Setup
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    data_paths = [
        Path("data/EEGBCI"),
        Path("data/TUAB"),
        Path("data/SleepEDF"),
    ]
    
    dataset = PretrainingDataset(data_paths)
    
    # Add validation set
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Add augmentations
    train_dataset.dataset.transform = AugmentationPipeline()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model and tasks
    model = AdaptivePreFormerPretraining(
        input_dim=dataset.data.shape[1],
        d_model=256,
        nhead=8,
        num_layers=6,
        pretraining_tasks={
            'reconstruction': ReconstructionTask(),
            'temporal': TemporalPredictionTask(),
            'contrastive': ContrastiveTask()
        }
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Initialize validator and early stopping
    validator = PretrainingValidator(model, val_loader, device)
    early_stopping = EarlyStopping(patience=7)
    
    # Training loop
    logging.info("Starting pretraining...")
    for epoch in range(100):
        # Training
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)
        
        # Validation
        val_losses = validator.validate()
        avg_val_loss = sum(val_losses.values()) / len(val_losses)
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
            
        # Log metrics
        logging.info(f"Epoch {epoch+1}/100:")
        logging.info(f"Train Loss: {train_loss:.4f}")
        for task, loss in val_losses.items():
            logging.info(f"Val Loss ({task}): {loss:.4f}")
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_losses)
            
    # Save final model
    save_final_model(model)
    logging.info("Pretraining completed!")

if __name__ == '__main__':
    pretrain() 