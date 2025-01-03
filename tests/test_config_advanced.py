import pytest
import os
import yaml
import tempfile
from pathlib import Path
from AdaptivePreFormer.utils.config_manager import ConfigManager

class TestConfigInheritance:
    """测试配置继承机制"""
    
    def test_base_config_inheritance(self):
        base_config = {
            "model": {
                "hidden_size": 256,
                "num_layers": 4
            }
        }
        
        child_config = {
            "inherit_from": "base_config.yaml",
            "model": {
                "num_layers": 6  # 覆盖基础配置
            }
        }
        
        # 创建临时配置文件
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "base_config.yaml"
            child_path = Path(temp_dir) / "child_config.yaml"
            
            with open(base_path, "w") as f:
                yaml.dump(base_config, f)
            with open(child_path, "w") as f:
                yaml.dump(child_config, f)
                
            config = ConfigManager(child_path)
            
            # 验证继承和覆盖
            assert config.model.hidden_size == 256  # 继承的值
            assert config.model.num_layers == 6     # 覆盖的值
            
    def test_multiple_inheritance(self):
        """测试多重继承"""
        base_config = {
            "model": {"hidden_size": 256}
        }
        
        middle_config = {
            "inherit_from": "base_config.yaml",
            "training": {"batch_size": 32}
        }
        
        final_config = {
            "inherit_from": "middle_config.yaml",
            "data": {"normalize": True}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建配置文件链
            for name, content in [
                ("base_config.yaml", base_config),
                ("middle_config.yaml", middle_config),
                ("final_config.yaml", final_config)
            ]:
                path = Path(temp_dir) / name
                with open(path, "w") as f:
                    yaml.dump(content, f)
                    
            config = ConfigManager(Path(temp_dir) / "final_config.yaml")
            
            # 验证多重继承
            assert config.model.hidden_size == 256
            assert config.training.batch_size == 32
            assert config.data.normalize is True

class TestConfigValidation:
    """测试高级配置验证"""
    
    def test_dependent_values(self):
        """测试相互依赖的配置值"""
        config_data = {
            "model": {
                "num_layers": 4,
                "layer_sizes": [256, 128, 64, 32]  # 必须与num_layers匹配
            }
        }
        
        config = ConfigManager.from_dict(config_data)
        assert len(config.model.layer_sizes) == config.model.num_layers
        
        # 测试不匹配的情况
        with pytest.raises(ValueError):
            ConfigManager.from_dict({
                "model": {
                    "num_layers": 4,
                    "layer_sizes": [256, 128]  # 长度不匹配
                }
            })
            
    def test_conditional_validation(self):
        """测试条件验证规则"""
        # 当使用特定优化器时需要额外参数
        config_data = {
            "training": {
                "optimizer": {
                    "type": "adam",
                    "beta1": 0.9,
                    "beta2": 0.999
                }
            }
        }
        
        config = ConfigManager.from_dict(config_data)
        assert config.training.optimizer.beta1 == 0.9
        
        # 使用SGD时不需要beta参数
        config_data["training"]["optimizer"] = {
            "type": "sgd",
            "momentum": 0.9
        }
        config = ConfigManager.from_dict(config_data)
        assert config.training.optimizer.momentum == 0.9

class TestConfigEnvironment:
    """测试环境变量和系统配置集成"""
    
    def test_env_variable_expansion(self):
        """测试环境变量展开"""
        os.environ["MODEL_DIR"] = "/path/to/models"
        
        config_data = {
            "paths": {
                "model_dir": "${MODEL_DIR}",
                "data_dir": "${DATA_DIR:-/default/data/path}"
            }
        }
        
        config = ConfigManager.from_dict(config_data)
        assert config.paths.model_dir == "/path/to/models"
        assert config.paths.data_dir == "/default/data/path"  # 使用默认值
        
    def test_system_info_integration(self):
        """测试系统信息集成"""
        config_data = {
            "system": {
                "num_workers": "auto",  # 应该基于CPU核心数设置
                "memory_limit": "auto"  # 应该基于系统内存设置
            }
        }
        
        config = ConfigManager.from_dict(config_data)
        assert isinstance(config.system.num_workers, int)
        assert isinstance(config.system.memory_limit, int)

class TestConfigVersioning:
    """测试配置版本控制"""
    
    def test_version_compatibility(self):
        """测试配置版本兼容性"""
        old_config = {
            "version": "1.0",
            "model": {
                "hidden_size": 256
            }
        }
        
        # 新版本配置格式
        new_config = {
            "version": "2.0",
            "model": {
                "architecture": {
                    "hidden_dim": 256  # 新的命名方式
                }
            }
        }
        
        # 测试向后兼容性
        config = ConfigManager.from_dict(old_config)
        assert config.model.hidden_size == 256
        
        # 测试新格式
        config = ConfigManager.from_dict(new_config)
        assert config.model.architecture.hidden_dim == 256
        
    def test_version_migration(self):
        """测试配置版本迁移"""
        old_config = {
            "version": "1.0",
            "deprecated_option": "value"
        }
        
        with pytest.warns(DeprecationWarning):
            config = ConfigManager.from_dict(old_config)

class TestConfigSerialization:
    """测试配置序列化"""
    
    def test_config_export_formats(self):
        """测试不同格式的配置导出"""
        config_data = {
            "model": {"hidden_size": 256}
        }
        config = ConfigManager.from_dict(config_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # YAML格式
            yaml_path = Path(temp_dir) / "config.yaml"
            config.save(yaml_path)
            assert yaml_path.exists()
            
            # JSON格式
            json_path = Path(temp_dir) / "config.json"
            config.save(json_path)
            assert json_path.exists()
            
            # 环境变量格式
            env_path = Path(temp_dir) / "config.env"
            config.save(env_path, format="env")
            assert env_path.exists() 