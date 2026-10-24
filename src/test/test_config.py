"""
Tests of configuration
"""

from src.hbs.config import ModelConfig, ProjectConfig, TrainingConfig, TokenizerConfig

def test_default_config_creation() -> None:
    config = ProjectConfig()

    assert config.model.d_model == 256
    assert config.model.n_ssm_layers == 5
    assert config.training.batch_size == 64

def test_model_config_immutable() -> None:
    config = ModelConfig()

    try:
        config.d_model = 512
        assert False, "Should not be mutable"
    except AttributeError:
        pass

def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.aux_weight == 0.5
    assert config.energy_weight == 0.1