import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from AdaptivePreFormer.utils.config_manager import ConfigManager

class TestConfigMerging:
    """测试配置合并功能"""
    
    def test_deep_merge(self):
        """测试深度合并配置"""
        base_config = {
            "model": {
                "encoder": {
                    "layers": 4,
                    "attention": {
                        "heads": 8,
                        "dropout": 0.1
                    }
                }
            }
        }
        
        update_config = {
            "model": {
                "encoder": {
                    "attention": {
                        "heads": 16  # 只更新heads
                    }
                }
            }
        }
        
        config = ConfigManager.from_dict(base_config)
        config.merge(update_config)
        
        # 验证合并结果
        assert config.model.encoder.layers == 4  # 保持不变
        assert config.model.encoder.attention.heads == 16  # 更新
        assert config.model.encoder.attention.dropout == 0.1  # 保持不变
        
    def test_list_merge_strategies(self):
        """测试列表合并策略"""
        base_config = {
            "model": {
                "layer_sizes": [256, 128, 64],
                "dropouts": [0.1, 0.1, 0.1]
            }
        }
        
        # 测试追加策略
        update_append = {
            "model": {
                "layer_sizes": [32],
                "_merge_strategy": "append"
            }
        }
        
        config = ConfigManager.from_dict(base_config)
        config.merge(update_append)
        assert config.model.layer_sizes == [256, 128, 64, 32]
        
        # 测试替换策略
        update_replace = {
            "model": {
                "dropouts": [0.2, 0.2],
                "_merge_strategy": "replace"
            }
        }
        
        config.merge(update_replace)
        assert config.model.dropouts == [0.2, 0.2]

class TestConfigTemplates:
    """测试配置模板功能"""
    
    def test_template_substitution(self):
        """测试模板替换"""
        template_config = {
            "model": {
                "hidden_size": "${base_size}",
                "intermediate_size": "${base_size}*4",
                "num_heads": "${base_size}/64"
            }
        }
        
        variables = {
            "base_size": 256
        }
        
        config = ConfigManager.from_template(template_config, variables)
        assert config.model.hidden_size == 256
        assert config.model.intermediate_size == 1024
        assert config.model.num_heads == 4
        
    def test_conditional_templates(self):
        """测试条件模板"""
        template_config = {
            "model": {
                "type": "transformer",
                "config": {
                    "@if type == 'transformer'": {
                        "attention_type": "self",
                        "num_layers": 6
                    },
                    "@if type == 'cnn'": {
                        "kernel_size": 3,
                        "channels": [64, 128, 256]
                    }
                }
            }
        }
        
        config = ConfigManager.from_template(template_config)
        assert config.model.config.attention_type == "self"
        assert not hasattr(config.model.config, "kernel_size")

class TestConfigValidators:
    """测试配置验证器"""
    
    def test_custom_validators(self):
        """测试自定义验证器"""
        def validate_power_of_two(value: int) -> bool:
            return value > 0 and (value & (value - 1)) == 0
            
        config_data = {
            "model": {
                "hidden_size": 256,
                "_validators": {
                    "hidden_size": validate_power_of_two
                }
            }
        }
        
        config = ConfigManager.from_dict(config_data)
        assert config.model.hidden_size == 256
        
        # 测试无效值
        with pytest.raises(ValueError):
            ConfigManager.from_dict({
                "model": {
                    "hidden_size": 257,  # 不是2的幂
                    "_validators": {
                        "hidden_size": validate_power_of_two
                    }
                }
            })
            
    def test_range_validators(self):
        """测试范围验证器"""
        config_data = {
            "training": {
                "learning_rate": 0.001,
                "dropout": 0.1,
                "_validators": {
                    "learning_rate": "range(0, 1)",
                    "dropout": "range(0, 1)"
                }
            }
        }
        
        config = ConfigManager.from_dict(config_data)
        assert config.training.learning_rate == 0.001
        
        # 测试范围外的值
        with pytest.raises(ValueError):
            ConfigManager.from_dict({
                "training": {
                    "dropout": 1.5,  # 超出范围
                    "_validators": {
                        "dropout": "range(0, 1)"
                    }
                }
            })

class TestConfigPresets:
    """测试配置预设功能"""
    
    def test_load_preset(self):
        """测试加载预设配置"""
        presets = {
            "small": {
                "model": {
                    "hidden_size": 256,
                    "num_layers": 4
                }
            },
            "base": {
                "model": {
                    "hidden_size": 512,
                    "num_layers": 6
                }
            },
            "large": {
                "model": {
                    "hidden_size": 1024,
                    "num_layers": 12
                }
            }
        }
        
        # 测试加载small预设
        config = ConfigManager.from_preset("small", presets)
        assert config.model.hidden_size == 256
        assert config.model.num_layers == 4
        
        # 测试加载large预设
        config = ConfigManager.from_preset("large", presets)
        assert config.model.hidden_size == 1024
        assert config.model.num_layers == 12
        
    def test_preset_with_override(self):
        """测试预设配置与覆盖"""
        presets = {
            "base": {
                "model": {
                    "hidden_size": 512,
                    "num_layers": 6
                }
            }
        }
        
        overrides = {
            "model": {
                "num_layers": 8
            }
        }
        
        config = ConfigManager.from_preset("base", presets, overrides)
        assert config.model.hidden_size == 512  # 从预设
        assert config.model.num_layers == 8     # 被覆盖 