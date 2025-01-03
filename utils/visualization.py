import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.metrics = {
            'train_loss': [],
            'val_loss': {},
            'learning_rate': []
        }
        
    def update(self, epoch, train_loss, val_losses, lr):
        # Update metrics
        self.metrics['train_loss'].append(train_loss)
        for task, loss in val_losses.items():
            if task not in self.metrics['val_loss']:
                self.metrics['val_loss'][task] = []
            self.metrics['val_loss'][task].append(loss)
        self.metrics['learning_rate'].append(lr)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        for task, loss in val_losses.items():
            self.writer.add_scalar(f'Loss/val_{task}', loss, epoch)
        self.writer.add_scalar('Learning_rate', lr, epoch)
        
    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        for task, losses in self.metrics['val_loss'].items():
            plt.plot(losses, label=f'Val Loss ({task})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        return plt.gcf()
        
    def close(self):
        self.writer.close() 