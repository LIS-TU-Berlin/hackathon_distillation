import math
from typing import Dict, Tuple
import torch
from hackathon_distillation.data_loader.dataset import BaseImageDataset


class _WelfordAcc:
    """Elementwise Welford accumulator that supports vector/array features."""
    def __init__(self, feature_shape: Tuple[int, ...], device: torch.device = torch.device("cpu")):
        self.n = 0  # scalar count
        self.mean = torch.zeros(feature_shape, dtype=torch.float32, device=device)
        self.M2 = torch.zeros(feature_shape, dtype=torch.float32, device=device)
        self.min = torch.full(feature_shape, float('inf'), dtype=torch.float32, device=device)
        self.max = torch.full(feature_shape, float('-inf'), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_batch(self, x: torch.Tensor):
        """
        x: [batch, *feature_shape]
        Performs a parallel Welford update using batch stats.
        """
        if x.numel() == 0:
            return
        # batch stats
        k = x.shape[0]
        batch_min = x.amin(dim=0)
        batch_max = x.amax(dim=0)
        batch_mean = x.mean(dim=0)
        # M2 for the batch: sum((x - batch_mean)^2) over batch
        # This equals (unbiased variance * (k-1)), but we compute directly to avoid numerical drift.
        x_centered = x - batch_mean
        batch_M2 = (x_centered * x_centered).sum(dim=0)

        # combine existing aggregate with batch aggregate
        n = self.n
        if n == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = k
            self.min = batch_min
            self.max = batch_max
            return

        delta = batch_mean - self.mean
        new_n = n + k
        self.mean = self.mean + delta * (k / new_n)
        self.M2 = self.M2 + batch_M2 + (delta * delta) * (n * k / new_n)
        self.n = new_n

        # elementwise min/max
        self.min = torch.minimum(self.min, batch_min)
        self.max = torch.maximum(self.max, batch_max)

    def finalize(self) -> Dict[str, torch.Tensor]:
        # unbiased variance if n > 1, else zeros
        if self.n > 1:
            var = self.M2 / (self.n - 1)
        else:
            var = torch.zeros_like(self.M2)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "mean": self.mean.cpu(),
            "std": std.cpu(),
            "min": self.min.cpu(),
            "max": self.max.cpu(),
        }


def compute_dataset_stats_welford(dataset: BaseImageDataset, chunk_size: int = 8192) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Computes per-key stats over the full replay buffer using Welford's algorithm.
    Returns:
        {
          "obs.img": {"mean": (1,H,W), "std": (1,H,W), "min": (1,H,W), "max": (1,H,W)},
          "obs.state": {"mean": (D,), ...},
          "action": {"mean": (D,), ...},
        }
    Notes:
      - Uses dataset.replay_buffer to include ALL episodes, not just the training mask.
      - Streams in chunks along the first dimension to keep memory usage bounded.
    """
    assert hasattr(dataset, "replay_buffer"), "Dataset must expose a replay_buffer"

    rb = dataset.replay_buffer
    # Map output keys to replay buffer keys
    key_map = {
        "obs.img": "depth",
        "obs.state": "ee_pos",
        "action": "ee_action",
    }

    # Discover feature shapes from the replay buffer
    # Expect shapes like:
    #   depth: [N, H, W]          -> output feature shape (1, H, W) to match provided stats convention
    #   ee_pos / ee_action: [N, D] -> output feature shape (D,)
    accs: Dict[str, _WelfordAcc] = {}

    for out_key, rb_key in key_map.items():
        arr = rb[rb_key]  # expected numpy array or something indexable with shape [N, ...]
        if not hasattr(arr, "shape") or arr.ndim < 2:
            raise ValueError(f"Replay buffer key '{rb_key}' has unexpected shape: {getattr(arr, 'shape', None)}")
        feature_shape = arr.shape[1:]  # drop the sample dimension

        # deprecated
        # if out_key == "obs.img":
        #     feature_shape = (1, *feature_shape)

        accs[out_key] = _WelfordAcc(feature_shape)

    # Stream through each key independently in chunks
    for out_key, rb_key in key_map.items():
        data = rb[rb_key]
        N = data.shape[0]
        is_image = (out_key == "obs.img")

        start = 0
        while start < N:
            end = min(start + chunk_size, N)
            chunk_np = data[start:end]  # shape [n, ...]
            x = torch.from_numpy(chunk_np).to(torch.float32)
            if is_image:
                # current x: [n, H, W] -> [n, 1, H, W]
                x = x.unsqueeze(1)

            # Update accumulator with this batch
            accs[out_key].update_batch(x)
            start = end

    # Finalize
    stats = {k: acc.finalize() for k, acc in accs.items()}
    return stats
