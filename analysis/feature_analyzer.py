import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

class FeatureAnalyzer:
    """Analysis tools for model features"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.feature_stats = {}
        
    def extract_features(self, dataloader):
        """Extract features from data"""
        features = []
        attention_maps = []
        quality_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch.to(self.device))
                features.append(outputs['features'].cpu())
                attention_maps.append(outputs['attention_weights'])
                quality_scores.append(outputs['quality_scores'].cpu())
                
        return {
            'features': torch.cat(features),
            'attention_maps': attention_maps,
            'quality_scores': torch.cat(quality_scores)
        }
        
    def analyze_feature_importance(self, features, labels):
        """Analyze feature importance"""
        # Compute correlation with labels
        correlations = []
        for i in range(features.shape[1]):
            corr, _ = spearmanr(features[:,i], labels)
            correlations.append(abs(corr))
            
        # Sort features by importance
        importance_ranking = np.argsort(correlations)[::-1]
        
        return {
            'correlations': correlations,
            'importance_ranking': importance_ranking
        }
        
    def visualize_feature_space(self, features, labels=None):
        """Visualize feature space using dimensionality reduction"""
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels if labels is not None else None,
            cmap='viridis',
            alpha=0.6
        )
        
        if labels is not None:
            plt.colorbar(scatter)
            
        plt.title('Feature Space Visualization (t-SNE)')
        return plt.gcf()
        
    def analyze_attention_patterns(self, attention_maps):
        """Analyze attention patterns"""
        # Average attention patterns
        avg_attention = torch.mean(torch.stack(attention_maps), dim=0)
        
        # Compute attention statistics
        attention_stats = {
            'mean_attention': avg_attention.mean().item(),
            'attention_std': avg_attention.std().item(),
            'max_attention': avg_attention.max().item(),
            'sparsity': (avg_attention < 0.1).float().mean().item()
        }
        
        return attention_stats
        
    def plot_attention_heatmap(self, attention_map):
        """Plot attention heatmap"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_map.numpy(),
            cmap='viridis',
            center=0,
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title('Attention Pattern Heatmap')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        return plt.gcf()
        
    def analyze_quality_distribution(self, quality_scores):
        """Analyze quality score distribution"""
        quality_stats = {
            'mean_quality': quality_scores.mean().item(),
            'quality_std': quality_scores.std().item(),
            'quality_range': (quality_scores.min().item(), 
                            quality_scores.max().item())
        }
        
        # Plot quality distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(quality_scores.numpy(), bins=50)
        plt.title('Quality Score Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        
        quality_stats['distribution_plot'] = plt.gcf()
        return quality_stats 