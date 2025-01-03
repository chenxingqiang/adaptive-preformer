import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import networkx as nx

class ModelVisualizer:
    """Advanced visualization tools for model analysis"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def plot_attention_patterns(self, input_data, layer_idx=-1):
        """Visualize attention patterns"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_data)
            
        # Get attention weights from specified layer
        attn_weights = outputs['attention_weights'][layer_idx]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attn_weights[0].cpu(),  # First head of batch
            cmap='viridis',
            xticklabels=50,
            yticklabels=50
        )
        plt.title(f'Attention Pattern (Layer {layer_idx})')
        return plt.gcf()
        
    def visualize_feature_evolution(self, input_sequence):
        """Visualize feature evolution through layers"""
        features = []
        self.model.eval()
        
        def hook_fn(module, input, output):
            features.append(output.detach().cpu())
            
        # Register hooks
        hooks = []
        for layer in self.model.transformer.layers:
            hooks.append(layer.register_forward_hook(hook_fn))
            
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_sequence)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Create interactive visualization
        fig = go.Figure()
        for i, feat in enumerate(features):
            fig.add_trace(go.Heatmap(
                z=feat[0].numpy(),
                name=f'Layer {i}',
                visible=False
            ))
            
        # Create slider
        steps = []
        for i in range(len(features)):
            step = dict(
                method="update",
                args=[{"visible": [j == i for j in range(len(features))]}],
                label=f"Layer {i}"
            )
            steps.append(step)
            
        sliders = [dict(
            active=0,
            steps=steps
        )]
        
        fig.update_layout(sliders=sliders)
        return fig
        
    def plot_quality_distribution(self, dataset_loader):
        """Visualize quality score distribution"""
        quality_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataset_loader:
                outputs = self.model(batch.to(self.device))
                quality_scores.append(outputs['quality_scores'].cpu())
                
        quality_scores = torch.cat(quality_scores)
        
        # Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=quality_scores)
        plt.title('Distribution of Quality Scores')
        plt.ylabel('Quality Score')
        return plt.gcf()
        
    def visualize_model_graph(self):
        """Visualize model architecture as a graph"""
        G = nx.DiGraph()
        
        def add_module_to_graph(module, name=''):
            for child_name, child in module.named_children():
                child_full_name = f"{name}/{child_name}" if name else child_name
                G.add_node(child_full_name, type=type(child).__name__)
                G.add_edge(name, child_full_name)
                add_module_to_graph(child, child_full_name)
                
        add_module_to_graph(self.model)
        
        # Plot graph
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        nx.draw(
            G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=2000,
            font_size=8,
            arrows=True
        )
        return plt.gcf() 