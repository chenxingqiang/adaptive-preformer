import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

class AverageMeter:
    """Computes and stores the average and current value"""
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

class EEGDataset(Dataset):
    def __init__(self, data_path, required=False):
        """
        Args:
            data_path: Path to dataset directory
            required: If True, raise error when dataset not found; if False, log warning
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            msg = f"Dataset path {data_path} does not exist"
            if required:
                raise FileNotFoundError(msg)
            logging.warning(msg)
            self.data = torch.tensor([])
            self.labels = torch.tensor([])
            return
            
        # 查找.pt文件
        pt_files = list(self.data_path.glob("*.pt"))
        if not pt_files:
            msg = f"No .pt files found in {data_path}"
            if required:
                raise FileNotFoundError(msg) 
            logging.warning(msg)
            self.data = torch.tensor([])
            self.labels = torch.tensor([])
            return
            
        # 加载数据文件
        data_file = pt_files[0]  # 使用第一个找到的.pt文件
        try:
            data_dict = torch.load(data_file, weights_only=True)  # 添加weights_only=True
            self.data = data_dict['data'].float()  # Ensure float type for EEG data
            self.labels = data_dict['labels'].long()  # Ensure long type for labels
            
            # 分析数据集
            self.num_samples = len(self.data)
            self.num_features = self.data[0].shape[0] if len(self.data) > 0 else 0
            self.num_classes = len(torch.unique(self.labels))
            
            print(f"\nDataset loaded from {data_file}:")
            print(f"Number of samples: {self.num_samples}")
            print(f"Number of features: {self.num_features}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Data shape: {self.data.shape}")
            print(f"Labels shape: {self.labels.shape}")
            print(f"Unique labels: {torch.unique(self.labels).tolist()}\n")
            
        except Exception as e:
            msg = f"Error loading data from {data_file}: {str(e)}"
            if required:
                raise RuntimeError(msg)
            logging.warning(msg)
            self.data = torch.tensor([])
            self.labels = torch.tensor([])
            self.num_samples = 0
            self.num_features = 0
            self.num_classes = 0
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
    @property
    def input_dim(self):
        """Return input dimension (number of features)"""
        return self.num_features
        
    @property
    def output_dim(self):
        """Return output dimension (number of classes)"""
        return self.num_classes
        