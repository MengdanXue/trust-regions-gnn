"""
Configuration Management
========================

Load and manage experiment configurations from YAML files.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: str = "GCN"
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 4  # For GAT


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 0.01
    weight_decay: float = 0.0005
    epochs: int = 200
    patience: int = 50
    min_delta: float = 0.001


@dataclass
class DataConfig:
    """Data split configuration."""
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    normalize_features: bool = True


@dataclass
class ExperimentConfig:
    """Experiment settings."""
    seed: int = 42
    n_runs: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    device: str = "auto"
    log_interval: int = 10


@dataclass
class HSweepConfig:
    """H-Sweep experiment configuration."""
    h_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    n_nodes: int = 1000
    n_edges: int = 7500
    n_features: int = 16
    n_classes: int = 2
    feature_separability: float = 0.5
    n_runs_per_h: int = 10


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    h_sweep: HSweepConfig = field(default_factory=HSweepConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()

        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'experiment' in data:
            config.experiment = ExperimentConfig(**data['experiment'])
        if 'h_sweep' in data:
            config.h_sweep = HSweepConfig(**data['h_sweep'])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__,
            'h_sweep': self.h_sweep.__dict__,
        }

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def get_device(self):
        """Get PyTorch device."""
        import torch
        if self.experiment.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.experiment.device)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Config object with all settings.

    Example:
        >>> config = load_config('configs/default.yaml')
        >>> print(config.model.hidden_channels)
        64
    """
    if config_path is None:
        # Look for default config
        default_paths = [
            Path(__file__).parent.parent / 'configs' / 'default.yaml',
            Path('configs/default.yaml'),
        ]
        for path in default_paths:
            if path.exists():
                return Config.from_yaml(str(path))
        # Return default config if no file found
        return Config()

    return Config.from_yaml(config_path)


# Convenience function
def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
