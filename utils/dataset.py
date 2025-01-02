import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

class EEGDataset(Dataset):
    def __init__(self, root_dir, sequence_length=512, transform=None):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 获取所有数据文件
        self.samples = []
        for label in [0, 1]:
            label_dir = self.root_dir / str(label)
            for file_path in label_dir.glob("*.sub*"):
                self.samples.append((file_path, label))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        x = torch.load(file_path)
        
        # 序列长度处理
        if x.shape[1] > self.sequence_length:
            start = random.randint(0, x.shape[1] - self.sequence_length)
            x = x[:, start:start + self.sequence_length]
        else:
            pad_length = self.sequence_length - x.shape[1]
            x = torch.nn.functional.pad(x, (0, pad_length))
            
        if self.transform:
            x = self.transform(x)
            
        return x, label
    

    class ContinuousSequenceDataset(Dataset):
    def __init__(self, data_path, sequence_length=1000, transform=None):
        super().__init__()
        self.data = torch.load(data_path)
        self.sequence_length = sequence_length
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x, y = self.data[idx]
        
        # 确保序列长度一致
        if x.shape[1] > self.sequence_length:
            start = random.randint(0, x.shape[1] - self.sequence_length)
            x = x[:, start:start + self.sequence_length]
        else:
            # 填充
            pad_length = self.sequence_length - x.shape[1]
            x = F.pad(x, (0, pad_length))
            
        if self.transform:
            x = self.transform(x)
            
        return x, y
    

class TrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 100
        self.warmup_epochs = 10
        self.weight_decay = 0.01
        self.gradient_clip = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count