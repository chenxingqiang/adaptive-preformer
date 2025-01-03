import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
class DistributedTrainer:
    """Distributed training handler"""
    def __init__(self, rank, world_size, model, train_loader, val_loader):
        self.rank = rank
        self.world_size = world_size
        
        # Setup process group
        setup_distributed(rank, world_size)
        
        # Wrap model
        self.model = DistributedDataParallel(
            model.to(rank),
            device_ids=[rank]
        )
        
        # Setup data loaders
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = val_loader
        
    def train(self, epochs):
        for epoch in range(epochs):
            self.train_sampler.set_epoch(epoch)
            train_loss = self.train_epoch()
            
            if self.rank == 0:
                val_loss = self.validate()
                self.log_metrics(epoch, train_loss, val_loss)
                
    def cleanup(self):
        dist.destroy_process_group() 