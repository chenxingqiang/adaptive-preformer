import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import json

class ConfigVisualizer:
    """配置系统可视化工具"""
    
    def __init__(self, monitor_data: Dict[str, Any]):
        self.data = monitor_data
        self.style_config()
        
    @staticmethod
    def style_config():
        """设置可视化样式"""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_access_heatmap(self, 
                           output_path: Optional[str] = None,
                           figsize: tuple = (12, 8)) -> None:
        """绘制配置访问热力图"""
        # 准备数据
        access_data = pd.DataFrame(self.data['access_history'])
        access_data['hour'] = pd.to_datetime(
            access_data['timestamp'], unit='s'
        ).dt.hour
        access_data['minute'] = pd.to_datetime(
            access_data['timestamp'], unit='s'
        ).dt.minute
        
        # 创建热力图矩阵
        heatmap_data = pd.crosstab(
            access_data['hour'],
            access_data['minute']
        )
        
        # 绘制热力图
        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            annot=False,
            fmt='d',
            cbar_kws={'label': 'Access Count'}
        )
        
        plt.title('Configuration Access Patterns (24-hour)')
        plt.xlabel('Minute')
        plt.ylabel('Hour')
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def plot_dependency_graph(self,
                            output_path: Optional[str] = None,
                            figsize: tuple = (15, 10)) -> None:
        """绘制配置依赖关系图"""
        G = nx.DiGraph()
        
        # 构建依赖图
        for access in self.data['access_history']:
            path_parts = access['path'].split('.')
            for i in range(len(path_parts)-1):
                G.add_edge(path_parts[i], path_parts[i+1])
                
        # 设置节点大小基于访问频率
        node_sizes = []
        for node in G.nodes():
            size = sum(1 for a in self.data['access_history']
                      if node in a['path']) * 100
            node_sizes.append(size)
            
        # 绘制图形
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)
        
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=node_sizes,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                alpha=0.7)
                
        plt.title('Configuration Dependency Graph')
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def create_interactive_timeline(self,
                                  output_path: Optional[str] = None) -> None:
        """创建交互式时间线可视化"""
        df = pd.DataFrame(self.data['access_history'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # 创建时间线图
        fig = go.Figure()
        
        # 添加不同操作类型的散点
        operations = df['operation'].unique()
        colors = px.colors.qualitative.Set3[:len(operations)]
        
        for op, color in zip(operations, colors):
            mask = df['operation'] == op
            fig.add_trace(go.Scatter(
                x=df[mask]['timestamp'],
                y=df[mask]['path'],
                mode='markers',
                name=op,
                marker=dict(size=8, color=color),
                hovertemplate=(
                    '<b>Time</b>: %{x}<br>'
                    '<b>Path</b>: %{y}<br>'
                    '<b>Operation</b>: ' + op + '<br>'
                    '<extra></extra>'
                )
            ))
            
        # 更新布局
        fig.update_layout(
            title='Configuration Access Timeline',
            xaxis_title='Time',
            yaxis_title='Configuration Path',
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        if output_path:
            fig.write_html(output_path)
            
    def plot_performance_metrics(self,
                               output_path: Optional[str] = None,
                               figsize: tuple = (15, 10)) -> None:
        """绘制性能指标图表"""
        performance = self.data['performance']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 内存使用趋势
        memory_data = pd.DataFrame(performance['memory_usage'])
        memory_data.plot(ax=ax1)
        ax1.set_title('Memory Usage Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Memory (MB)')
        ax1.grid(True)
        
        # 访问延迟分布
        latency_data = pd.DataFrame(performance['access_latency'])
        sns.boxplot(data=latency_data, ax=ax2)
        ax2.set_title('Access Latency Distribution')
        ax2.set_ylabel('Latency (ms)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def generate_html_report(self, output_dir: str) -> None:
        """生成HTML格式的可视化报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 生成各种可视化
        self.plot_access_heatmap(output_dir / 'heatmap.png')
        self.plot_dependency_graph(output_dir / 'dependency.png')
        self.create_interactive_timeline(output_dir / 'timeline.html')
        self.plot_performance_metrics(output_dir / 'performance.png')
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Configuration System Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; }}
                .metric {{ 
                    background: #f5f5f5;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Configuration System Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Access Patterns</h2>
                <img src="heatmap.png" alt="Access Heatmap">
                <p>This heatmap shows the distribution of configuration access patterns over time.</p>
            </div>
            
            <div class="section">
                <h2>Dependency Structure</h2>
                <img src="dependency.png" alt="Dependency Graph">
                <p>This graph visualizes the relationships between different configuration components.</p>
            </div>
            
            <div class="section">
                <h2>Interactive Timeline</h2>
                <iframe src="timeline.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <img src="performance.png" alt="Performance Metrics">
                <p>These charts show key performance indicators of the configuration system.</p>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                {self._generate_findings_html()}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html()}
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w') as f:
            f.write(html_content)
            
    def _generate_findings_html(self) -> str:
        """生成关键发现的HTML内容"""
        patterns = self.data.get('patterns', {})
        findings = []
        
        if patterns.get('read_write_ratio'):
            findings.append(
                f"Read/Write Ratio: {patterns['read_write_ratio']:.2f}"
            )
            
        if patterns.get('access_frequency'):
            findings.append(
                f"Average Access Frequency: "
                f"{patterns['access_frequency']['avg_per_second']:.2f} per second"
            )
            
        if patterns.get('peak_times'):
            findings.append(
                f"Peak Access Times: {', '.join(patterns['peak_times'])}"
            )
            
        return '<ul>' + ''.join(f'<li>{f}</li>' for f in findings) + '</ul>'
        
    def _generate_recommendations_html(self) -> str:
        """生成建议的HTML内容"""
        recommendations = self.data.get('recommendations', [])
        return '<ul>' + ''.join(
            f'<li>{r}</li>' for r in recommendations
        ) + '</ul>' 