import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainingTask(nn.Module):
    """Base class for all pretraining tasks"""
    def __init__(self):
        super().__init__()
        
    def compute_loss(self, predictions, targets):
        raise NotImplementedError
        
    def generate_targets(self, x):
        raise NotImplementedError

class ReconstructionTask(PretrainingTask):
    """Signal reconstruction task"""
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        
    def random_masking(self, x):
        """Randomly mask input sequence"""
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        return x_masked, ids_restore
        
    def compute_loss(self, predictions, targets):
        return F.mse_loss(predictions, targets)
        
    def generate_targets(self, x):
        x_masked, restore_ids = self.random_masking(x)
        return x_masked, x, restore_ids

class TemporalPredictionTask(PretrainingTask):
    """Predict future timesteps"""
    def __init__(self, num_future_steps=5):
        super().__init__()
        self.num_future_steps = num_future_steps
        
    def compute_loss(self, predictions, targets):
        return F.mse_loss(predictions, targets)
        
    def generate_targets(self, x):
        # Split sequence into input and target future steps
        input_seq = x[:, :-self.num_future_steps, :]
        target_seq = x[:, -self.num_future_steps:, :]
        return input_seq, target_seq

class ContrastiveTask(PretrainingTask):
    """Contrastive learning between different views"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def generate_views(self, x):
        """Generate two different views of the same sequence"""
        # View 1: Random temporal crop
        view1 = self.temporal_crop(x)
        
        # View 2: Frequency masking
        view2 = self.frequency_mask(x)
        
        return view1, view2
        
    def compute_loss(self, z1, z2):
        """InfoNCE loss"""
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        logits = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(B, device=z1.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss 