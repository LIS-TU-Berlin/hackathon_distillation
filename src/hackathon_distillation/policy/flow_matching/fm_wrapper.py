import torch as th

import torch.nn.functional as F
from omegaconf import DictConfig

from coarse2fine_flow.policy.ModelWrapperABC import ModelWrapper
from coarse2fine_flow.policy.diffusion.ddpm_wrapper import DdpmModel
from coarse2fine_flow.policy.utils.normalize import Normalize, Unnormalize


class FlowMatchingWrapper(ModelWrapper):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_stats: dict[str, dict[str, th.Tensor]] | None = None,
    ):
        """Implements flow matching with optimal transport vector fields.

        Source: Lipman et al. (2022); paper: http://arxiv.org/abs/2210.02747.
        """
        super().__init__(cfg, dataset_stats)
        self.num_inference_steps = cfg.num_inference_steps if cfg.num_inference_steps is not None else cfg.noise_scheduler.num_train_timesteps
        self.model = DdpmModel(cfg)  # ddpm and fm share the same model

        # use normalizers for in and outputs
        self.normalize_inputs = Normalize(cfg.network.input_shapes, cfg.network.input_normalization_modes, dataset_stats)
        self.normalize_targets = Normalize(cfg.network.output_shapes, cfg.network.output_normalization_modes, dataset_stats)
        self.unnormalize_outputs = Unnormalize(cfg.network.output_shapes, cfg.network.output_normalization_modes, dataset_stats)

        # sampling ist as per p0: https://www.physicalintelligence.company/download/pi0.pdf
        self.pos_emb_scale = cfg.pos_emb_scale
        if cfg.use_beta:
            self.beta_dist = th.distributions.Beta(1.5, 1.0)
            self.s_cutoff = 0.999

    def compute_loss(self, model: th.nn.Module, batch: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action"})
        horizon = batch["action"].shape[1]
        n_obs_steps = batch["observation.state"].shape[1]
        assert horizon == self.config.horizon, f"MISMATCH: horizon = {horizon}, config.horizon = {self.config.horizon}"
        assert n_obs_steps == self.config.n_obs_steps, f"MISMATCH: n_obs_steps = {n_obs_steps}, config.n_obs = {self.config.n_obs_steps}"

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Sample noise that we'll add to the latents
        trajectory = batch["action"].to(model.device)
        noise = th.randn_like(trajectory)
        bsz = trajectory.shape[0]
        if self.config.use_beta:
            timesteps = self.s_cutoff * (1. - self.beta_dist.sample((bsz, 1, 1)).to(model.device))  # timesteps ~ Beta
        else:
            timesteps = th.rand((bsz, 1, 1), device=model.device)  # timesteps ~ U[0, 1]

        # Corrupt
        noisy_trajectory = timesteps * trajectory + (1 - timesteps) * noise

        # Predict the noise residual
        target = trajectory - noise
        pred = model(noisy_trajectory, timesteps.squeeze() * self.pos_emb_scale, batch=batch)
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
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
        batch: th.Tensor | None = None,
        generator: th.Generator | None = None
    ) -> th.Tensor:
        """Inference.

        This implements the linear Euler-style sampling.
        """
        device = self.model.device

        # disabled for now
        # Sample prior. -- predict for the entire trajectory
        # if self.prior_sample is not None:
        #     sample = self.prior_sample.clone().to(device)
        # else:
        sample = th.randn(
            size=(
                batch_size,
                self.config.horizon,
                self.config.network.output_shapes["action"][0],
            ),
            device=device,
            generator=generator,
        )

        num_inference_steps = self.num_inference_steps
        delta = 1.0 / num_inference_steps
        for t in range(num_inference_steps):
            # Predict model output.
            timesteps = th.full((batch_size,), fill_value=t / num_inference_steps, device=sample.device)
            dxn_dn = self.model(
                sample,
                timesteps,
                batch=batch,
            )
            # Compute previous sample: x_t -> x_t-1
            sample += dxn_dn * delta

        return sample
