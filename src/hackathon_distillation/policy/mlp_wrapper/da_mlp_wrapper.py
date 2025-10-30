import hydra
import torch as th
import einops
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig

from hackathon_distillation.networks.da_encoder import DaEncoder
from hackathon_distillation.networks.depth_encoder import DepthImageEncoder
from hackathon_distillation.policy.ModelWrapperABC import ModelWrapper
from hackathon_distillation.policy.utils.normalize import Normalize, Unnormalize


class DepthMlpWrapper(ModelWrapper):
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
        self._use_images = any(k.startswith("obs.img") for k in cfg.network.input_shapes)
        self._use_state = "obs.state" in cfg.network.input_shapes

        # Instantiate the model
        self.model = DaMlpModel(cfg)

    def compute_loss(self, model: th.nn.Module, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        assert set(batch).issuperset({"obs.state", "action"})
        horizon = batch["action"].shape[1]
        assert horizon == self.config.pred_horizon, (
            f"MISMATCH: horizon = {horizon}, config.pred_horizon = {self.config.pred_horizon}"
        )

        # Normalize inputs and targets (so the model learns in normalized space).
        if self._use_images:
            og_obs = batch["obs.img"][:, :horizon].clone()
        batch = self.normalize_inputs(batch)
        if self._use_images:
            batch["obs.img"] = og_obs
        batch = self.normalize_targets(batch)

        # Forward pass
        pred = model(batch)  # shape: (B, pred_horizon, action_dim)
        target = batch["action"].to(model.device)  # already normalized above
        loss = F.mse_loss(pred, target, reduction="none")  # (B, pred_horizon, action_dim)

        # Optional mask for padded timesteps.
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]  # (B, pred_horizon)
            loss = loss * in_episode_bound.unsqueeze(-1)

        return {"loss": loss.mean()}

    @th.no_grad()
    def sample(
        self,
        batch_size: int,
        batch: dict[str, th.Tensor] | None = None,
        **kwargs
    ) -> th.Tensor:
        if batch is None:
            raise ValueError("`batch` must be provided for inference.")
        pred_norm = self.model(batch)  # (B, pred_horizon, action_dim)
        return pred_norm

    def configure_optimizers(self, **kwargs) -> tuple[list[th.optim.Optimizer], list[th.optim.lr_scheduler._LRScheduler]]:
        """Return list of optimizers and list of schedulers."""
        optimizer_cls = hydra.utils.get_class(self.config.optimizer._target_)
        optimizer = optimizer_cls(self.model.parameters(), **self.config.optimizer.kwargs)
        optimizer.train_mode = True

        return [optimizer], []


class DaMlpModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config
        self._use_images = any(k.startswith("obs.img") for k in config.network.input_shapes)
        self._use_state = "obs.state" in config.network.input_shapes

        print(f"MlpModel: use_images = {self._use_images}, use_state = {self._use_state}")

        self.depth_anything = DaEncoder()
        for param in self.depth_anything.parameters():
            param.requires_grad = False
        self.depth_encoder = None

        # Compute input_dim
        input_dim = 0
        if self.cfg.obs_horizon > 0 and self._use_state:
            input_dim += self.cfg.network.input_shapes["obs.state"][0]

        if self._use_images:
            self.depth_encoder = DepthImageEncoder(feature_dim=self.cfg.network.spatial_softmax_num_keypoints, pretrained=False, freeze_layers=False)
            num_images = len([k for k in self.cfg.network.input_shapes if k.startswith("obs.img")])
            input_dim += self.depth_encoder.feature_dim * num_images

        self.input_dim = input_dim
        print(f"MlpModel: input_dim = {self.input_dim}")

        # Create the MLP model
        self.network = hydra.utils.get_class(self.cfg.network.network_cls)(
            self.cfg, input_dim=self.input_dim * self.cfg.obs_horizon,
            output_dim=self.cfg.network.output_shapes["action"][0]* self.cfg.pred_horizon
        )

    def forward(self, batch: dict[str, th.Tensor]) -> th.Tensor:
        input = None
        if batch is not None:
            features = []
            if self._use_state:
                batch_size, n_obs_steps = batch["obs.state"].shape[:2]
                features.append(batch["obs.state"][:, :n_obs_steps].to(self.device))

            if self._use_images:
                batch_size, n_obs_steps = batch["obs.img"].shape[:2]
                imgs = batch["obs.img"][:, :n_obs_steps]
                img_inputs = einops.rearrange(imgs, "b s ... -> (b s) ...").to(self.device, non_blocking=True)
                depth = self.depth_anything(img_inputs)[:, None]  # add depth shape
                img_features = self.depth_encoder(depth)
                img_features = einops.rearrange(img_features, "(b s) ... -> b s (...)", b=batch_size, s=n_obs_steps)
                features.append(img_features)

            input = th.cat(features, dim=-1).flatten(start_dim=1)
        return self.network(input).view(-1, self.cfg.pred_horizon, self.cfg.network.output_shapes["action"][0])

    def print_batch_shapes(self, batch: dict[str, th.Tensor]):
        for k, v in batch.items():
            print(f"{k}: {v.shape}")

    @property
    def device(self):
        return next(self.parameters()).device
