import pytest
import os
import yaml
from pathlib import Path
from AdaptivePreFormer.utils.config_manager import ConfigManager

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"

def setup_module():
    """设置测试环境"""
    TEST_DATA_DIR.mkdir(exist_ok=True)

def create_test_config(config_data):
    """创建测试配置文件"""
    config_path = TEST_DATA_DIR / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path

def test_config_load():
    """测试配置加载"""
    config_data = {
        "model": {
            "hidden_size": 256,
            "num_layers": 4
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    config_path = create_test_config(config_data)
    config = ConfigManager(config_path)
    
    assert config.model.hidden_size == 256
    assert config.training.batch_size == 32

def test_config_validation():
    """测试配置验证"""
    config_data = {
        "model": {
            "hidden_size": "invalid",  # 应该是整数
            "num_layers": 4
        }
    }
    
    config_path = create_test_config(config_data)
    with pytest.raises(ValueError):
        ConfigManager(config_path)

def test_config_update():
    """测试配置更新"""
    config_data = {
        "model": {
            "hidden_size": 256
        }
    }
    
    config_path = create_test_config(config_data)
    config = ConfigManager(config_path)
    
    updates = {
        "model": {
            "hidden_size": 512
        }
    }
    
    config.update(updates)
    assert config.model.hidden_size == 512

def test_config_save():
    """测试配置保存"""
    config_data = {
        "model": {
            "hidden_size": 256
        }
    }
    
    config_path = create_test_config(config_data)
    config = ConfigManager(config_path)
    
    save_path = TEST_DATA_DIR / "saved_config.yaml"
    config.save(save_path)
    
    # 验证保存的配置
    loaded_config = ConfigManager(save_path)
    assert loaded_config.model.hidden_size == 256

def test_missing_config():
    """测试配置文件缺失"""
    with pytest.raises(FileNotFoundError):
        ConfigManager("nonexistent.yaml")

def test_invalid_yaml():
    """测试无效的YAML格式"""
    invalid_yaml = """
    model:
      hidden_size: 256
      - invalid_list  # 无效的YAML语法
    """
    
    config_path = TEST_DATA_DIR / "invalid.yaml"
    with open(config_path, "w") as f:
        f.write(invalid_yaml)
        
    with pytest.raises(yaml.YAMLError):
        ConfigManager(config_path)

def test_nested_config():
    """测试嵌套配置"""
    config_data = {
        "model": {
            "encoder": {
                "layers": {
                    "attention": {
                        "heads": 8
                    }
                }
            }
        }
    }
    
    config_path = create_test_config(config_data)
    config = ConfigManager(config_path)
    
    assert config.model.encoder.layers.attention.heads == 8

def test_config_types():
    """测试不同类型的配置值"""
    config_data = {
        "string_value": "test",
        "int_value": 42,
        "float_value": 3.14,
        "bool_value": True,
        "list_value": [1, 2, 3],
        "dict_value": {"key": "value"}
    }
    
    config_path = create_test_config(config_data)
    config = ConfigManager(config_path)
    
    assert isinstance(config.string_value, str)
    assert isinstance(config.int_value, int)
    assert isinstance(config.float_value, float)
    assert isinstance(config.bool_value, bool)
    assert isinstance(config.list_value, list)
    assert isinstance(config.dict_value, dict)

def test_config_defaults():
    """测试配置默认值"""
    config_data = {
        "model": {
            "hidden_size": 256
        }
    }
    
    config_path = create_test_config(config_data)
    config = ConfigManager(config_path)
    
    # 访问未定义的配置项应返回None
    assert config.model.undefined_option is None

def teardown_module():
    """清理测试环境"""
    import shutil
    shutil.rmtree(TEST_DATA_DIR) 