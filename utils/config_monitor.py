import time
import logging
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import psutil

@dataclass
class ConfigAccessRecord:
    """配置访问记录"""
    timestamp: float
    path: str
    operation: str  # 'read', 'write', 'delete'
    value: Any
    source: str  # 调用来源

class ConfigMonitor:
    """配置监控系统"""
    
    def __init__(self, 
                 max_history: int = 1000,
                 log_file: Optional[str] = None):
        self.max_history = max_history
        self.access_history: List[ConfigAccessRecord] = []
        self.hot_paths: Dict[str, int] = {}  # 热点访问路径计数
        self.lock = threading.Lock()
        
        # 设置日志
        self.logger = logging.getLogger("ConfigMonitor")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def record_access(self, 
                     path: str, 
                     operation: str, 
                     value: Any,
                     source: str = None) -> None:
        """记录配置访问"""
        with self.lock:
            # 创建访问记录
            record = ConfigAccessRecord(
                timestamp=time.time(),
                path=path,
                operation=operation,
                value=value,
                source=source or self._get_caller_info()
            )
            
            # 更新历史记录
            self.access_history.append(record)
            if len(self.access_history) > self.max_history:
                self.access_history.pop(0)
                
            # 更新热点统计
            self.hot_paths[path] = self.hot_paths.get(path, 0) + 1
            
            # 记录日志
            self.logger.debug(
                f"Config {operation}: {path} = {value} from {record.source}"
            )
            
    def get_hot_paths(self, top_n: int = 10) -> List[tuple]:
        """获取最频繁访问的配置路径"""
        with self.lock:
            sorted_paths = sorted(
                self.hot_paths.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_paths[:top_n]
            
    def get_access_patterns(self) -> Dict[str, Any]:
        """分析配置访问模式"""
        with self.lock:
            patterns = {
                "read_write_ratio": self._calculate_rw_ratio(),
                "access_frequency": self._calculate_access_frequency(),
                "peak_times": self._find_peak_access_times(),
                "common_sequences": self._find_common_sequences()
            }
            return patterns
            
    def generate_report(self, 
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成配置访问报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_accesses": len(self.access_history),
                "unique_paths": len(self.hot_paths),
                "hot_paths": self.get_hot_paths(),
                "patterns": self.get_access_patterns()
            },
            "performance": {
                "memory_usage": self._get_memory_usage(),
                "access_latency": self._calculate_access_latency()
            },
            "recommendations": self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
        
    def _get_caller_info(self) -> str:
        """获取调用者信息"""
        import inspect
        stack = inspect.stack()
        # 跳过监控系统的帧
        for frame in stack[2:]:
            filename = Path(frame.filename).name
            if not filename.startswith("config_"):
                return f"{filename}:{frame.lineno}"
        return "unknown"
        
    def _calculate_rw_ratio(self) -> float:
        """计算读写比例"""
        reads = sum(1 for r in self.access_history if r.operation == 'read')
        writes = sum(1 for r in self.access_history if r.operation == 'write')
        return reads / max(writes, 1)
        
    def _calculate_access_frequency(self) -> Dict[str, float]:
        """计算访问频率"""
        if not self.access_history:
            return {"avg_per_second": 0}
            
        duration = self.access_history[-1].timestamp - self.access_history[0].timestamp
        if duration == 0:
            return {"avg_per_second": len(self.access_history)}
            
        return {
            "avg_per_second": len(self.access_history) / duration,
            "peak_per_second": self._calculate_peak_frequency()
        }
        
    def _calculate_peak_frequency(self, window_size: float = 1.0) -> float:
        """计算峰值访问频率"""
        if len(self.access_history) < 2:
            return len(self.access_history)
            
        max_count = 0
        for i in range(len(self.access_history)):
            time_i = self.access_history[i].timestamp
            count = sum(1 for r in self.access_history[i:]
                       if r.timestamp - time_i <= window_size)
            max_count = max(max_count, count)
            
        return max_count / window_size
        
    def _find_peak_access_times(self) -> List[str]:
        """查找访问高峰时间"""
        if not self.access_history:
            return []
            
        # 按小时统计
        hour_counts = {}
        for record in self.access_history:
            hour = datetime.fromtimestamp(record.timestamp).hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
        # 找出高于平均值的时间段
        avg_count = sum(hour_counts.values()) / len(hour_counts)
        peak_hours = [
            f"{hour:02d}:00"
            for hour, count in hour_counts.items()
            if count > avg_count * 1.5
        ]
        return sorted(peak_hours)
        
    def _find_common_sequences(self, min_length: int = 2) -> List[List[str]]:
        """查找常见的访问序列"""
        if len(self.access_history) < min_length:
            return []
            
        # 提取路径序列
        sequences = []
        current_seq = []
        last_time = 0
        
        for record in self.access_history:
            if record.timestamp - last_time > 1.0:  # 1秒内的操作视为序列
                if len(current_seq) >= min_length:
                    sequences.append(current_seq[:])
                current_seq = []
            current_seq.append(record.path)
            last_time = record.timestamp
            
        # 查找重复序列
        sequence_counts = {}
        for seq in sequences:
            seq_key = tuple(seq)
            sequence_counts[seq_key] = sequence_counts.get(seq_key, 0) + 1
            
        # 返回出现次数最多的序列
        common_sequences = sorted(
            sequence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return [list(seq) for seq, _ in common_sequences]
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "history_size": len(self.access_history),
            "unique_paths": len(self.hot_paths)
        }
        
    def _calculate_access_latency(self) -> Dict[str, float]:
        """计算访问延迟统计"""
        if not self.access_history:
            return {"avg_ms": 0, "max_ms": 0}
            
        # 计算相邻访问之间的时间差
        latencies = []
        for i in range(1, len(self.access_history)):
            latency = (self.access_history[i].timestamp - 
                      self.access_history[i-1].timestamp) * 1000  # 转换为毫秒
            latencies.append(latency)
            
        if not latencies:
            return {"avg_ms": 0, "max_ms": 0}
            
        return {
            "avg_ms": sum(latencies) / len(latencies),
            "max_ms": max(latencies),
            "min_ms": min(latencies),
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)]
        }
        
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于访问模式分析
        patterns = self.get_access_patterns()
        if patterns["read_write_ratio"] > 10:
            recommendations.append(
                "考虑增加读缓存以优化频繁读取操作"
            )
            
        # 基于热点分析
        hot_paths = self.get_hot_paths()
        if hot_paths and hot_paths[0][1] > len(self.access_history) * 0.5:
            recommendations.append(
                f"配置路径 '{hot_paths[0][0]}' 访问频率过高，建议优化访问策略"
            )
            
        # 基于内存使用分析
        memory_usage = self._get_memory_usage()
        if memory_usage["rss_mb"] > 100:  # 超过100MB
            recommendations.append(
                "内存使用较高，考虑减少历史记录保留数量"
            )
            
        # 基于访问延迟分析
        latency = self._calculate_access_latency()
        if latency["p95_ms"] > 100:  # P95延迟超过100ms
            recommendations.append(
                "访问延迟较高，建议检查配置存储和加载机制"
            )
            
        return recommendations 