import hydra
import torch as th
import einops
import torch.nn.functional as F
from schedulefree import ScheduleFreeWrapper
from torch import nn
from typing import Optional
from omegaconf import DictConfig

from hackathon_distillation.policy.ModelWrapperABC import ModelWrapper
from hackathon_distillation.policy.utils.normalize import Normalize, Unnormalize
from hackathon_distillation.networks.depth_encoder import DepthImageEncoder


def prepare_global_conditioning(
    depth_encoder: th.nn.Module,
    batch: dict[str, th.Tensor],
    device: th.device,
    use_images: bool = False
) -> th.Tensor:
    """Computes the conditioning from observations and timesteps."""
    batch_size, n_obs_steps = batch["obs.state"].shape[:2]

    global_cond_feats = [batch["obs.state"][:, :n_obs_steps].to(device, non_blocking=True)]

    if use_images:
        imgs = batch["obs.img"][:, :n_obs_steps]
        if imgs.ndim < 6:
            imgs = imgs.unsqueeze(2)  # add extra dimension; hack for datasets where the single camera is implicit
        img_inputs = einops.rearrange(imgs, "b s ... -> (b s) ...")
        img_features = depth_encoder(img_inputs.to(device, non_blocking=True))
        img_features = einops.rearrange(img_features, "(b s) ... -> b s (...)", b=batch_size, s=n_obs_steps)
        global_cond_feats.append(img_features)

    conditioning = th.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
    return conditioning


class DdpmWrapper(ModelWrapper):
    def __init__(
            self,
            cfg: DictConfig,
            dataset_stats: dict[str, dict[str, th.Tensor]] | None = None,
    ):
        super().__init__(cfg, dataset_stats)

        # Normalization
        self.normalize_inputs = Normalize(
            cfg.network.input_shapes, cfg.network.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            cfg.network.output_shapes, cfg.network.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            cfg.network.output_shapes, cfg.network.output_normalization_modes, dataset_stats
        )

        # Setup DDPM config
        self.noise_scheduler = hydra.utils.instantiate(
            cfg.noise_scheduler,
            num_train_timesteps=cfg.noise_scheduler.num_train_timesteps,
            beta_start=cfg.noise_scheduler.beta_start,
            beta_end=cfg.noise_scheduler.beta_end,
            beta_schedule=cfg.noise_scheduler.beta_schedule,
            clip_sample=cfg.noise_scheduler.clip_sample,
            clip_sample_range=cfg.noise_scheduler.clip_sample_range,
            prediction_type=cfg.noise_scheduler.prediction_type,
        )
        self.prediction_type = cfg.prediction_type
        if cfg.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = cfg.num_inference_steps

        # Instantiate the model
        self.model = DdpmModel(cfg)

    def compute_loss(self, model: th.nn.Module, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        """
        This function expects `batch` to have (at least):
        {
            "obs.state": (B, n_obs_steps, state_dim)

            "obs.imgs": (B, n_obs_steps, num_cameras, C, H, W)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"obs.state", "action"})
        horizon = batch["action"].shape[1]
        assert horizon == self.config.pred_horizon, f"MISMATCH: horizon = {horizon}, config.pred_horizon = {self.config.pred_horizon}"

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Sample noise that we'll add to the latents
        trajectory = batch["action"].to(model.device, non_blocking=True)
        eps = th.randn_like(trajectory)
        timesteps = th.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
            dtype=th.long,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = model(noisy_trajectory, timesteps, batch=batch)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded.
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return {"loss": loss.mean()}

    def sample(
        self,
        batch_size: int,
        batch: dict[str, th.Tensor] | None = None,
        generator: th.Generator | None = None
    ) -> th.Tensor:
        """Inference."""
        device = self.model.device

        # Sample prior. -- predict for the entire trajectory
        sample = th.randn(
            size=(
                batch_size,
                self.config.pred_horizon,
                self.config.network.output_shapes["action"][0],
            ),
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.model(
                sample,
                th.full((batch_size,), t, dtype=th.long, device=sample.device),
                batch=batch,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def configure_optimizers(self, **kwargs) -> tuple[list[th.optim.Optimizer], list[th.optim.lr_scheduler._LRScheduler]]:
        """Return list of optimizers and list of schedulers."""
        decay_params, no_decay_params = self.model.network.configure_parameters()
        if self.model._use_images:
            decay_params += list(self.model.depth_encoder.parameters())
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.optimizer.kwargs.weight_decay,
                "lr": self.config.optimizer.kwargs.lr,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": self.config.optimizer.kwargs.lr,
            },
        ]

        optimizer_cls = hydra.utils.get_class(self.config.optimizer._target_)
        optimizer = optimizer_cls(optim_groups, **self.config.optimizer.kwargs)
        optimizer.train_mode = True

        return [optimizer], []


class DdpmModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.depth_encoder = None
        self._use_images = any(k.startswith("obs.img") for k in cfg.network.input_shapes)

        # Compute global_cond_dim
        global_cond_dim = 0
        if cfg.obs_horizon > 0:
            assert "obs.state" in cfg.network.input_shapes
            global_cond_dim += cfg.network.input_shapes["obs.state"][0]

        if self._use_images:
            self.depth_encoder = DepthImageEncoder(feature_dim=cfg.network.spatial_softmax_num_keypoints, pretrained=False, freeze_layers=False) #RgbEncoder(cfg.network)
            num_images = len([k for k in cfg.network.input_shapes if k.startswith("obs.img")])
            global_cond_dim += self.depth_encoder.feature_dim * num_images
        self.global_cond_dim = global_cond_dim

        # Create the UNet model
        self.network = hydra.utils.get_class(cfg.network.network_cls)(
            cfg, global_cond_dim=self.global_cond_dim * cfg.obs_horizon
        )

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        batch: Optional[dict[str, th.Tensor]] = None
    ) -> th.Tensor:
        global_cond = None
        if batch is not None:
            global_cond = prepare_global_conditioning(
                self.depth_encoder,
                batch,
                device=x.device,
                use_images=self._use_images
            )
        return self.network(x, timesteps, global_cond=global_cond)

    @property
    def device(self):
        return next(self.parameters()).device
