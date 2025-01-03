import torch
import onnx
import tensorrt as trt
import numpy as np
from pathlib import Path
import logging

class ModelDeployer:
    """Utilities for model deployment"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def export_onnx(self, save_path, input_shape):
        """Export model to ONNX format"""
        save_path = Path(save_path)
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path / 'model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(save_path / 'model.onnx')
        onnx.checker.check_model(onnx_model)
        
        logging.info(f"Model exported to ONNX: {save_path / 'model.onnx'}")
        
    def optimize_for_deployment(self):
        """Optimize model for deployment"""
        # Freeze batch norm statistics
        self.model.eval()
        
        # Fuse operations
        self.model = torch.quantization.fuse_modules(
            self.model,
            ['conv', 'bn', 'relu']
        )
        
        # Quantize model
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        return self.model
        
    def create_deployment_package(self, save_path):
        """Create deployment package with all necessary files"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Export model
        self.export_onnx(save_path, (1, self.config.sequence_length, self.config.input_dim))
        
        # Save preprocessing parameters
        torch.save(
            self.model.preprocessor.state_dict(),
            save_path / 'preprocessor.pt'
        )
        
        # Save configuration
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        # Create README
        with open(save_path / 'README.md', 'w') as f:
            f.write(self.generate_deployment_docs())
            
        logging.info(f"Deployment package created at {save_path}")
        
    def generate_deployment_docs(self):
        """Generate deployment documentation"""
        docs = f"""
        # Model Deployment Package
        
        ## Model Information
        - Input shape: {self.config.input_dim}
        - Sequence length: {self.config.sequence_length}
        - Model type: AdaptivePreFormer
        
        ## Files
        - model.onnx: ONNX model file
        - preprocessor.pt: Preprocessing parameters
        - config.json: Model configuration
        
        ## Usage
        ```python
        import torch
        import onnxruntime
        
        # Load model
        session = onnxruntime.InferenceSession('model.onnx')
        
        # Load preprocessor
        preprocessor = torch.load('preprocessor.pt')
        
        # Preprocess input
        input_data = preprocess_input(data, preprocessor)
        
        # Run inference
        output = session.run(None, {{'input': input_data}})
        ```
        """
        return docs 