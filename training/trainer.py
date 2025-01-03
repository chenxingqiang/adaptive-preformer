import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from pathlib import Path

class AdaptiveTrainer:
    """Training manager for AdaptivePreFormer"""
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs
        )
        
        # Initialize losses
        self.criterion = MultiObjectiveLoss()
        
        # Setup metrics tracking
        self.metrics = TrainingMetrics()
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                # Forward pass
                outputs = self.model(data)
                
                # Compute losses
                loss, loss_components = self.criterion(outputs, target, self.model.get_stats())
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                self.metrics.update(loss_components)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
        return total_loss / len(self.train_loader)
        
    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                outputs = self.model(data)
                loss, _ = self.criterion(outputs, target, self.model.get_stats())
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader)
        
    def train(self):
        """Complete training loop"""
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(patience=self.config.training.patience)
        
        for epoch in range(self.config.training.num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.metrics.log_epoch(train_loss, val_loss)
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered")
                break
                
        return self.metrics.get_summary() 