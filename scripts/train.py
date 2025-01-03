import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from utils.dataset import EEGDataset
from models.transformer import AdaptivePreFormer

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device and ensure correct types
        data = data.float().to(device)  # Ensure float type for input data
        targets = targets.long().to(device)  # Ensure long type for targets
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs['logits'], targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = outputs['logits'].argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
        
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            # Move data to device and ensure correct types
            data = data.float().to(device)
            targets = targets.long().to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs['logits'], targets)
            
            # Track metrics
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            
    return total_loss / len(val_loader), correct / total

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    datasets = []
    data_root = Path("data")
    
    # Load required datasets
    required_datasets = ["EEGBCI"]
    for ds_name in required_datasets:
        if (data_root / ds_name).exists():
            datasets.append(EEGDataset(data_root / ds_name, required=True))
    
    if not datasets:
        raise RuntimeError("No datasets could be loaded!")
    
    # Get dataset properties
    dataset = datasets[0]  # Use first dataset
    input_dim = dataset.data.shape[1]
    sequence_length = dataset.data.shape[2]
    num_classes = len(torch.unique(dataset.labels))
    
    print(f"\nDataset properties:")
    print(f"Input dimensions: {input_dim}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nSplit sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
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
    
    # Initialize model
    model = AdaptivePreFormer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        num_classes=num_classes,
        max_seq_length=sequence_length
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(100):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
