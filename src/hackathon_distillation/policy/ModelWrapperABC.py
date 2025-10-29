import enum
import pathlib
from abc import ABC, abstractmethod

import hydra
import torch as th
from omegaconf import OmegaConf, DictConfig
from schedulefree import ScheduleFreeWrapper, AdamWScheduleFree


class TrainStrategy(enum.Enum):
    fsdp = enum.auto()
    ddp = enum.auto()
    cuda = enum.auto()


class ModelWrapper(ABC):
    model: th.nn.Module
    dataset_stats: dict[str, th.Tensor]
    config: DictConfig
    global_cond_dim: int

    def __init__(self, cfg: DictConfig, dataset_stats: dict[str, th.Tensor] | None = None,) -> None:
        self.config = cfg
        self.dataset_stats = dataset_stats
        self.n_action_steps = cfg.action_horizon

    def configure_optimizers(self, **kwargs) -> tuple[list[th.optim.Optimizer], list[th.optim.lr_scheduler._LRScheduler]]:
        """Return list of optimizers and list of schedulers."""
        optimizer_cls = hydra.utils.get_class(self.config.optimizer._target_)
        optimizer = ScheduleFreeWrapper(optimizer_cls(self.model.parameters(), **self.config.optimizer.kwargs))
        optimizer.train_mode = True

        # Example: Adding a learning rate scheduler if specified
        lr_scheduler = None
        # if hasattr(self.config, 'lr_scheduler') and self.config.lr_scheduler is not None:
        #     scheduler_cls = hydra.utils.get_class(self.config.lr_scheduler._target_)
        #     lr_scheduler = scheduler_cls(optimizer, **self.config.lr_scheduler.kwargs)

        schedulers = [lr_scheduler] if lr_scheduler is not None else []
        return [optimizer], schedulers

    def training_step(
        self,
        model: th.nn.Module,
        batch: dict[str, th.Tensor],
        optimizers: list[th.optim.Optimizer],
        lr_schedulers: list[th.optim.lr_scheduler._LRScheduler],
        scaler: th.cuda.amp.GradScaler,
    ) -> dict[str, th.Tensor]:
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Compute loss with mixed precision.
        with th.amp.autocast(device_type=model.device.type, dtype=th.float32):  # TODO: use high precision
            loss_dict = self.compute_loss(model, batch)

        # Backpropagate loss.
        loss = loss_dict.get("loss", None)
        scaler.scale(loss).backward()

        # Unscale the gradients and clip them.
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
            opt_params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            th.nn.utils.clip_grad_norm_(
                opt_params,
                self.config.grad_clip_norm,
                error_if_nonfinite=False,
            )

            # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

        # Step lr scheduler
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        # Updates the scale for next iteration.
        scaler.update()

        return loss_dict

    def validation_step(self, model: th.nn.Module, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        loss_dict = self.compute_loss(model, batch)
        return loss_dict

    def generate_actions(self, batch: dict[str, th.Tensor]) -> th.Tensor:
        """
        This function expects `batch` to have:
        {
            "obs.state": (B, n_obs_steps, state_dim)

            "obs.imgs": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "obs.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["obs.state"].shape[:2]
        assert n_obs_steps == self.config.obs_horizon, (
            f"Expected {self.config.obs_horizon} observation steps, but got {n_obs_steps}."
        )

        # Run sampling
        actions = self.sample(batch_size, batch=batch)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.action_horizon
        actions = actions[:, start:end]

        return actions

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: pathlib.Path,
        map_location: str | th.device = "cpu",
    ):
        """
        Instantiate the DdpmMlpWrapper from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            map_location (str or torch.device, optional): Device to map the checkpoint to.

        Returns:
            DdpmMlpWrapper: The instantiated wrapper with loaded weights.
        """
        # Load the checkpoint
        checkpoint = th.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Extract dataset_stats from the checkpoint
        if "stats" not in checkpoint:
            raise KeyError(f"Checkpoint does not contain 'stats' key.")

        # Instantiate the wrapper
        cfg = OmegaConf.load(f"{checkpoint_path.parent}/config.yaml")
        instance = cls(cfg, dataset_stats=checkpoint["stats"])

        # Load model state_dict
        if "model" in checkpoint:
            instance.model.load_state_dict(checkpoint["model"])
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict' key.")

        # (Optional) Load optimizer state_dict if present
        # if "optimizer" in checkpoint:
        #     instance.optimizer.load_state_dict(checkpoint["optimizer"])

        return instance

    @abstractmethod
    def compute_loss(self, model: th.nn.Module, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        """Compute the full loss given a batch.

        This method is supposed to be usable in both train and validation step.
        """
        pass

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        batch: th.Tensor | None = None,
        generator: th.Generator | None = None
    ) -> th.Tensor:
        """Inference."""
        pass
