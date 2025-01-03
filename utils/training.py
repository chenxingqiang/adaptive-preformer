import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
class PretrainingValidator:
    """Validation during pretraining"""
    def __init__(self, model, val_loader, device):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        
    def validate(self):
        self.model.eval()
        total_loss = {task: 0.0 for task in self.model.pretraining_heads.keys()}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = batch.to(self.device)
                
                # Compute loss for each task
                for task_name, task in self.model.pretraining_heads.items():
                    inputs, targets = task.generate_targets(batch)
                    predictions = self.model.forward_pretraining(inputs, task_name)
                    loss = task.compute_loss(predictions, targets)
                    total_loss[task_name] += loss.item()
                    
        # Average losses
        avg_loss = {task: loss/len(self.val_loader) for task, loss in total_loss.items()}
        return avg_loss 