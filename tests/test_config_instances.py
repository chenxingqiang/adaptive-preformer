import pytest
import torch
from pathlib import Path
from AdaptivePreFormer.utils.config_manager import ConfigManager

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"

def test_model_config():
    """测试模型配置实例"""
    config_data = {
        "model": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "activation": "gelu",
            "positional_encoding": {
                "type": "continuous",
                "max_len": 1000,
                "dim": 256
            },
            "encoder": {
                "type": "multiscale",
                "scales": [1, 2, 4],
                "fusion": "adaptive"
            }
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证模型配置
    assert config.model.hidden_size == 256
    assert config.model.num_layers == 4
    assert config.model.dropout == 0.1
    assert config.model.positional_encoding.type == "continuous"
    assert config.model.encoder.scales == [1, 2, 4]

def test_training_config():
    """测试训练配置实例"""
    config_data = {
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "early_stopping": {
                "patience": 10,
                "min_delta": 0.001
            },
            "optimizer": {
                "type": "adam",
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.01
            },
            "scheduler": {
                "type": "cosine",
                "T_max": 100,
                "eta_min": 1e-6
            }
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证训练配置
    assert config.training.batch_size == 32
    assert config.training.learning_rate == 0.001
    assert config.training.early_stopping.patience == 10
    assert config.training.optimizer.type == "adam"
    assert config.training.scheduler.type == "cosine"

def test_data_config():
    """测试数据配置实例"""
    config_data = {
        "data": {
            "train_path": "data/train",
            "val_path": "data/val",
            "test_path": "data/test",
            "preprocessing": {
                "normalize": True,
                "filter_type": "bandpass",
                "filter_args": {
                    "low_cut": 0.1,
                    "high_cut": 50.0
                }
            },
            "augmentation": {
                "time_shift": True,
                "noise": {
                    "type": "gaussian",
                    "std": 0.1
                },
                "scaling": {
                    "min": 0.8,
                    "max": 1.2
                }
            }
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证数据配置
    assert config.data.preprocessing.normalize is True
    assert config.data.preprocessing.filter_type == "bandpass"
    assert config.data.augmentation.noise.type == "gaussian"

def test_evaluation_config():
    """测试评估配置实例"""
    config_data = {
        "evaluation": {
            "metrics": ["accuracy", "f1", "auc"],
            "batch_size": 64,
            "save_predictions": True,
            "visualization": {
                "attention_maps": True,
                "feature_maps": True,
                "confusion_matrix": True
            },
            "export": {
                "format": "csv",
                "path": "results/"
            }
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证评估配置
    assert "accuracy" in config.evaluation.metrics
    assert config.evaluation.batch_size == 64
    assert config.evaluation.visualization.attention_maps is True

def test_logging_config():
    """测试日志配置实例"""
    config_data = {
        "logging": {
            "level": "INFO",
            "save_dir": "logs/",
            "tensorboard": {
                "enabled": True,
                "update_freq": 100
            },
            "checkpointing": {
                "enabled": True,
                "save_freq": 5,
                "max_to_keep": 3
            }
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证日志配置
    assert config.logging.level == "INFO"
    assert config.logging.tensorboard.enabled is True
    assert config.logging.checkpointing.max_to_keep == 3

def test_full_config_integration():
    """测试完整配置集成"""
    config_data = {
        "model": {
            "hidden_size": 256,
            "num_layers": 4
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "data": {
            "preprocessing": {
                "normalize": True
            }
        },
        "evaluation": {
            "metrics": ["accuracy"]
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证配置集成
    assert config.model.hidden_size == 256
    assert config.training.batch_size == 32
    assert config.data.preprocessing.normalize is True
    assert "accuracy" in config.evaluation.metrics
    assert config.logging.level == "INFO"

def test_config_validation_rules():
    """测试配置验证规则"""
    with pytest.raises(ValueError):
        ConfigManager.from_dict({
            "model": {
                "hidden_size": -256  # 不能为负数
            }
        })
    
    with pytest.raises(ValueError):
        ConfigManager.from_dict({
            "training": {
                "batch_size": 0  # 必须大于0
            }
        })
    
    with pytest.raises(ValueError):
        ConfigManager.from_dict({
            "model": {
                "activation": "invalid"  # 无效的激活函数
            }
        })

def test_device_config():
    """测试设备配置"""
    config_data = {
        "device": {
            "type": "cuda" if torch.cuda.is_available() else "cpu",
            "ids": [0, 1] if torch.cuda.device_count() > 1 else [0],
            "precision": "float32"
        }
    }
    
    config = ConfigManager.from_dict(config_data)
    
    # 验证设备配置
    assert config.device.type in ["cuda", "cpu"]
    assert isinstance(config.device.ids, list)
    assert config.device.precision == "float32" 