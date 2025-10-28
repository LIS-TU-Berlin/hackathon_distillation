from typing import Optional, Tuple, Type

import torch
from torch import nn

class MLP(nn.Module): 
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = ()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class Normalizer(torch.nn.Module):
    """
    Helper module for normalizing / unnormalzing torch tensors with a given mean and variance.
    Data is normalized as (x - mean) / sqrt(variance), i.e., zero mean and unit variance.
    
    Supports two modes:
    1. Fixed normalization: Uses provided mean/variance (original behavior)
    2. Running statistics: Updates mean/variance during training based on incoming data
    """

    def __init__(
        self, 
        mean: torch.Tensor, 
        variance: torch.Tensor, 
        min_variance: float = 1e-6
    ):
        """Initializes Normalizer object.

        Args:
            mean: torch.tensor with shape (n,); initial mean of normalization.
            variance: torch.tensor, shape (n,); initial variance of normalization.
            use_running_stats: If True, updates mean/variance during training.
            momentum: Momentum for running statistics update (similar to BatchNorm).
            min_variance: Minimum variance to prevent division by zero.
        """
        super().__init__()

        assert len(mean.shape) == 1
        assert len(variance.shape) == 1
        assert mean.shape[0] == variance.shape[0]

        self.min_variance = min_variance

        # Register buffers for mean and variance
        self.register_buffer("mean", mean.clone())
        self.register_buffer("variance", torch.clamp(variance.clone(), min=min_variance))
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.any(torch.isnan(x)), "Normalize: Input contains NaN values"
        return (x - self.mean) / torch.sqrt(self.variance)
    
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.any(torch.isnan(x)), "Unnormalize: Input contains NaN values"
        return x * torch.sqrt(self.variance) + self.mean

    def get_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current mean and variance."""
        return self.mean.clone(), self.variance.clone()