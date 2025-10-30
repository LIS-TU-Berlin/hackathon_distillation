import hydra
import torch as th
import einops
import torch.nn.functional as F
from torch import nn
from typing import Optional
from omegaconf import DictConfig

from hackathon_distillation.networks.depth_encoder import DepthImageEncoder
from hackathon_distillation.policy.ModelWrapperABC import ModelWrapper
from hackathon_distillation.policy.utils.normalize import Normalize, Unnormalize

def prepare_input(
    rgb_encoder: th.nn.Module,
    batch: dict[str, th.Tensor],
    device: th.device,
    use_images: bool = False, 
    use_state: bool = True,
) -> th.Tensor:
    """Computes the conditioning from observations and timesteps."""
    assert use_images or use_state, "At least one of use_images or use_state must be True."
    if use_state:
        batch_size, n_obs_steps = batch["obs.state"].shape[:2]
        features = [batch["obs.state"][:, :n_obs_steps].to(device)]
    else: 
        batch_size, n_obs_steps = batch["obs.img"].shape[:2]
        features = []

    if use_images:
        imgs = batch["obs.img"][:, :n_obs_steps]
        img_inputs = einops.rearrange(imgs, "b s ... -> (b s) ...")
        img_features = rgb_encoder(img_inputs.to(device))
        img_features = einops.rearrange(img_features, "(b s) ... -> b s (...)", b=batch_size, s=n_obs_steps)
        features.append(img_features)

    input = th.cat(features, dim=-1).flatten(start_dim=1)
    return input


class MlpWrapper(ModelWrapper):
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
        
        # Instantiate the model
        self.model = MlpModel(cfg)

    def compute_loss(self, model: th.nn.Module, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        """
        Expects batch keys:
        - "obs.state": (B, n_obs_steps, state_dim)
        - "action":    (B, pred_horizon, action_dim)
        - optionally "action_is_pad": (B, pred_horizon) to mask padded steps
        """
        assert set(batch).issuperset({"obs.state", "action"})
        horizon = batch["action"].shape[1]
        assert horizon == self.config.pred_horizon, (
            f"MISMATCH: horizon = {horizon}, config.pred_horizon = {self.config.pred_horizon}"
        )

        # Normalize inputs and targets (so the model learns in normalized space).
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Forward pass 
        pred = model(batch)  # shape: (B, pred_horizon, action_dim)

        # Basic MSE between prediction and normalized target actions.
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
        """
        Greedy inference: predict actions directly from observations.
        Batch is already normalized.
        Returns NORMALIZED actions of shape (B, pred_horizon, action_dim).
        """
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
    
class MlpModel(nn.Module): 
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config

        self.rgb_encoder = None
        self._use_images = any(k.startswith("obs.img") for k in config.network.input_shapes)
        self._use_state = "obs.state" in config.network.input_shapes
        
        print(f"MlpModel: use_images = {self._use_images}, use_state = {self._use_state}")
        
        # Compute input_dim
        input_dim = 0
        if self.cfg.obs_horizon > 0:
            if self._use_state:
                input_dim += self.cfg.network.input_shapes["obs.state"][0]

        if self._use_images:
            self.rgb_encoder = DepthImageEncoder(feature_dim=self.cfg.network.spatial_softmax_num_keypoints, pretrained=False, freeze_layers=False) #RgbEncoder(cfg.network)
            num_images = len([k for k in self.cfg.network.input_shapes if k.startswith("obs.img")])
            input_dim += self.rgb_encoder.feature_dim * num_images

        self.input_dim = input_dim
        print(f"MlpModel: input_dim = {self.input_dim}")

        # Create the UNet model
        self.network = hydra.utils.get_class(self.cfg.network.network_cls)(
            self.cfg, input_dim=self.input_dim * self.cfg.obs_horizon,
            output_dim=self.cfg.network.output_shapes["action"][0]* self.cfg.pred_horizon
        )

    def forward(
        self,
        batch: dict[str, th.Tensor], 
    ) -> th.Tensor:
        input = None

        if batch is not None:
            # self.print_batch_shapes(batch)
            input = prepare_input(
                self.rgb_encoder,
                batch,
                device=self.device,
                use_images=self._use_images,
                use_state=self._use_state,
            )
        return self.network(input).view(-1, self.cfg.pred_horizon, self.cfg.network.output_shapes["action"][0])
    
    def print_batch_shapes(self, batch: dict[str, th.Tensor]):
        for k, v in batch.items():
            print(f"{k}: {v.shape}")

    @property
    def device(self):
        return next(self.parameters()).device