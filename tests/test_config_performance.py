import pytest
import time
import random
import string
import psutil
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from AdaptivePreFormer.utils.config_manager import ConfigManager

def generate_large_config(depth: int, width: int) -> dict:
    """生成大规模嵌套配置"""
    if depth == 0:
        return {
            f"key_{i}": ''.join(random.choices(string.ascii_letters, k=10))
            for i in range(width)
        }
    
    return {
        f"level_{i}": generate_large_config(depth - 1, width)
        for i in range(width)
    }

class TestConfigPerformance:
    """配置系统性能测试"""
    
    def test_load_performance(self):
        """测试配置加载性能"""
        # 生成大规模配置
        large_config = generate_large_config(depth=5, width=10)
        
        # 测量加载时间
        start_time = time.time()
        config = ConfigManager.from_dict(large_config)
        load_time = time.time() - start_time
        
        # 验证加载时间在可接受范围内
        assert load_time < 1.0, f"配置加载时间过长: {load_time:.2f}秒"
        
    def test_access_performance(self):
        """测试配置访问性能"""
        config = ConfigManager.from_dict(generate_large_config(depth=4, width=8))
        
        # 测量随机访问性能
        access_times = []
        for _ in range(1000):
            path = '.'.join(f"level_{random.randint(0, 7)}" for _ in range(4))
            
            start_time = time.time()
            _ = config.get(path)
            access_times.append(time.time() - start_time)
            
        avg_access_time = sum(access_times) / len(access_times)
        assert avg_access_time < 0.001, f"配置访问时间过长: {avg_access_time:.6f}秒"
        
    def test_memory_usage(self):
        """测试内存使用情况"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 加载大规模配置
        large_config = generate_large_config(depth=6, width=10)
        config = ConfigManager.from_dict(large_config)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # 验证内存增长在合理范围内
        assert memory_increase < 100, f"内存使用过高: {memory_increase:.2f}MB"

class TestConfigConcurrency:
    """配置并发测试"""
    
    def test_thread_safety(self):
        """测试线程安全性"""
        config = ConfigManager.from_dict({
            "counter": 0,
            "data": {}
        })
        
        def update_config():
            for _ in range(100):
                current = config.counter
                time.sleep(0.001)  # 模拟实际操作
                config.counter = current + 1
                
        # 使用多线程并发更新
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_config) for _ in range(10)]
            for future in futures:
                future.result()
                
        # 验证最终结果
        assert config.counter == 1000, "线程安全性测试失败"
        
    def test_process_safety(self):
        """测试进程安全性"""
        config_file = Path("test_config.yaml")
        initial_config = {"counter": 0}
        ConfigManager.save_dict(initial_config, config_file)
        
        def update_config_file():
            for _ in range(10):
                config = ConfigManager(config_file)
                current = config.counter
                time.sleep(0.001)  # 模拟实际操作
                config.counter = current + 1
                config.save()
                
        # 使用多进程并发更新
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(update_config_file) for _ in range(4)]
            for future in futures:
                future.result()
                
        # 验证最终结果
        final_config = ConfigManager(config_file)
        assert final_config.counter == 40, "进程安全性测试失败"
        
        # 清理测试文件
        config_file.unlink()

class TestConfigStressTest:
    """配置压力测试"""
    
    def test_rapid_updates(self):
        """测试快速更新"""
        config = ConfigManager.from_dict({"value": 0})
        
        start_time = time.time()
        update_count = 10000
        
        for i in range(update_count):
            config.value = i
            
        end_time = time.time()
        updates_per_second = update_count / (end_time - start_time)
        
        assert updates_per_second > 1000, f"更新性能不足: {updates_per_second:.0f} 更新/秒"
        
    def test_large_scale_validation(self):
        """测试大规模验证"""
        def generate_validation_rules(size: int) -> dict:
            return {
                f"field_{i}": {
                    "type": random.choice(["int", "float", "str"]),
                    "range": (0, 100) if random.random() > 0.5 else None,
                    "required": random.random() > 0.3
                }
                for i in range(size)
            }
            
        validation_rules = generate_validation_rules(1000)
        test_data = {
            f"field_{i}": random.randint(0, 100)
            for i in range(1000)
        }
        
        start_time = time.time()
        config = ConfigManager.from_dict(test_data, validation_rules=validation_rules)
        validation_time = time.time() - start_time
        
        assert validation_time < 1.0, f"验证性能不足: {validation_time:.2f}秒"
        
    def test_concurrent_read_write(self):
        """测试并发读写"""
        config = ConfigManager.from_dict({"data": {}})
        
        def reader():
            for _ in range(1000):
                _ = config.data
                time.sleep(0.0001)
                
        def writer():
            for i in range(100):
                config.data[f"key_{i}"] = i
                time.sleep(0.001)
                
        # 同时启动多个读写线程
        with ThreadPoolExecutor(max_workers=20) as executor:
            readers = [executor.submit(reader) for _ in range(16)]
            writers = [executor.submit(writer) for _ in range(4)]
            
            # 等待所有操作完成
            for future in readers + writers:
                future.result()
                
        # 验证数据完整性
        assert len(config.data) == 100, "并发读写测试失败" 