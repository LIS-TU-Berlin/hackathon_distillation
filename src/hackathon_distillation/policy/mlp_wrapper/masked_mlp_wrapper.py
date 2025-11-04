import hydra
import torch as th
import einops
import torch.nn.functional as F
from torch import nn
from typing import Optional
from omegaconf import DictConfig

from hackathon_distillation.masker import Masker
from hackathon_distillation.networks.ddpm_rgb_encoder import RgbEncoder
from hackathon_distillation.networks.depth_encoder import DepthImageEncoder
from hackathon_distillation.policy.mlp_wrapper import MlpWrapper
from hackathon_distillation.policy.utils.normalize import Normalize, Unnormalize


class MaskedMlpWrapper(MlpWrapper):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_stats: dict[str, dict[str, th.Tensor]] | None = None,
    ):
        super().__init__(cfg, dataset_stats)
        self.model = MaskedMlpModel(cfg)


class MaskedMlpModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config
        self.masker = Masker()

        self.depth_encoder = None
        self._use_images = any(k.startswith("obs.img") for k in config.network.input_shapes)
        self._use_depth = any(k.startswith("obs.depth") for k in config.network.input_shapes)
        self._use_state = "obs.state" in config.network.input_shapes

        # Compute input_dim
        input_dim = 0
        if self.cfg.obs_horizon > 0:
            if self._use_state:
                input_dim += self.cfg.network.input_shapes["obs.state"][0]

        if self._use_depth:
            self.depth_encoder = DepthImageEncoder(n_channels_in=1, feature_dim=self.cfg.network.spatial_softmax_num_keypoints, pretrained=False, freeze_layers=False)
            # self.depth_encoder = RgbEncoder(config.network)
            num_images = len([k for k in self.cfg.network.input_shapes if k.startswith("obs.depth")])
            input_dim += self.depth_encoder.feature_dim * num_images

        self.input_dim = input_dim

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
            if self._use_state:
                batch_size, n_obs_steps = batch["obs.state"].shape[:2]
                features = [batch["obs.state"][:, :n_obs_steps].to(self.device, non_blocking=True)]
            else:
                batch_size, n_obs_steps = batch["obs.depth"].shape[:2]
                features = []

            if self._use_depth:
                # TODO: change mask operation to make depth = 0 be treated different to masked values
                mask = batch["obs.mask"][:, :n_obs_steps]
                depth = batch["obs.depth"][:, :n_obs_steps]
                #valid = (mask == 255.).float()
                #depth_with_flag = th.cat([depth, valid], dim=2)
                valid = (batch["obs.mask"][:, :n_obs_steps] == 255.)
                depth_with_flag = th.where(valid, depth, th.full_like(depth, -10.))
                img_inputs = einops.rearrange(depth_with_flag, "b s ... -> (b s) ...")
                img_features = self.depth_encoder(img_inputs.to(self.device, non_blocking=True))
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
