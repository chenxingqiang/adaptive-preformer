import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time

from AdaptivePreFormer.inference.metrics import InferenceMetrics
from AdaptivePreFormer.preprocessing.optimizer import PreprocessingOptimizer
from AdaptivePreFormer.quality.assessor import QualityAssessor

class InferenceManager:
    """Manager for model inference"""
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Initialize preprocessing
        self.preprocessor = PreprocessingOptimizer(config.model.input_dim)
        
        # Setup quality assessment
        self.quality_assessor = QualityAssessor(config.model.input_dim)
        
        # Initialize metrics
        self.metrics = InferenceMetrics()
        
    def preprocess_batch(self, batch):
        """Preprocess a batch of data"""
        # Quality assessment
        quality_scores = self.quality_assessor(batch)[0]
        
        # Adaptive preprocessing
        processed_batch, _ = self.preprocessor(batch)
        
        return processed_batch, quality_scores
        
    def run_inference(self, data_loader, return_features=False):
        """Run inference on data"""
        self.model.eval()
        predictions = []
        features = []
        quality_scores = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Running inference"):
                # Preprocess batch
                batch, quality = self.preprocess_batch(batch.to(self.device))
                
                # Model inference
                start_time = time.time()
                outputs = self.model(batch)
                inference_time = time.time() - start_time
                
                # Collect results
                predictions.append(outputs['logits'].cpu())
                if return_features:
                    features.append(outputs['features'].cpu())
                quality_scores.append(quality.cpu())
                
                # Update metrics
                self.metrics.update({
                    'inference_time': inference_time,
                    'quality_score': quality.mean().item()
                })
                
        results = {
            'predictions': torch.cat(predictions),
            'quality_scores': torch.cat(quality_scores),
            'metrics': self.metrics.get_summary()
        }
        
        if return_features:
            results['features'] = torch.cat(features)
            
        return results
        
    def optimize_for_inference(self):
        """Optimize model for inference"""
        # Fuse batch normalization layers
        torch.nn.utils.fusion.fuse_conv_bn_eval(self.model)
        
        # Quantize model
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        return self.model
        
    def export_model(self, save_path):
        """Export model for deployment"""
        save_path = Path(save_path)
        
        # Export to TorchScript
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, save_path / 'model.pt')
        
        # Save preprocessing parameters
        torch.save(
            self.preprocessor.state_dict(),
            save_path / 'preprocessor.pt'
        )
        
        # Save configuration
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        logging.info(f"Model exported to {save_path}") 