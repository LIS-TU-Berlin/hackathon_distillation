import hydra
import torch as th
from omegaconf import DictConfig
from schedulefree import ScheduleFreeWrapper

from coarse2fine_flow.networks.ada_unet import ConditionalUnet1DwithVarianceEstimation
from coarse2fine_flow.policy.ModelWrapperABC import ModelWrapper
from coarse2fine_flow.policy.utils.normalize import Normalize, Unnormalize


class AdaflowWrapper(ModelWrapper):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_stats: dict[str, dict[str, th.Tensor]] | None = None,
    ):
        super().__init__(cfg, dataset_stats)
        self.num_inference_steps = cfg.num_inference_steps if cfg.num_inference_steps is not None else cfg.noise_scheduler.num_train_timesteps
        self.model = ConditionalUnet1DwithVarianceEstimation(cfg)

        # use normalizers for in and outputs
        self.normalize_inputs = Normalize(cfg.network.input_shapes, cfg.network.input_normalization_modes,
                                          dataset_stats)
        self.normalize_targets = Normalize(cfg.network.output_shapes, cfg.network.output_normalization_modes,
                                           dataset_stats)
        self.unnormalize_outputs = Unnormalize(cfg.network.output_shapes, cfg.network.output_normalization_modes,
                                               dataset_stats)

        self.pos_emb_scale = cfg.pos_emb_scale
        self.eta = cfg.eta
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
        }"""
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
        pred, log_sqrt_var_pred = model(noisy_trajectory, timesteps.squeeze() * self.pos_emb_scale, batch=batch)
        error = (target - pred).pow(2)
        loss = 1 / (2. * th.exp(2 * log_sqrt_var_pred)) * error.sum(dim=(-1, -2)) + log_sqrt_var_pred

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return {"loss": loss.mean(), "fm_loss": error.mean(), "log_sqrt_var": log_sqrt_var_pred.mean()}

    def sample(
        self,
        batch_size: int,
        batch: th.Tensor | None = None,
        generator: th.Generator | None = None
    ) -> th.Tensor:
        device = self.model.device

        sample = th.randn(
            size=(
                batch_size,
                self.config.horizon,
                self.config.network.output_shapes["action"][0],
            ),
            device=device,
            generator=generator,
        )
        valid_action = th.zeros_like(sample)
        valid_action_found = th.zeros(batch_size).bool()
        num_steps_taken = th.zeros(batch_size)

        # inf_steps defines min. step length
        min_step_size = th.tensor([1 / self.num_inference_steps]).to(device)
        current_t = th.zeros(batch_size).to(device)
        for i in range(self.num_inference_steps):
            dxn_dn, log_sqrt_var_pred = self.model(sample, current_t * self.pos_emb_scale, batch=batch)

            # Compute step size from model
            step_size = th.max(self.eta / log_sqrt_var_pred.exp(), min_step_size)
            step_size = th.min(step_size, 1 - current_t)   # clip so we dont overstep
            sample += dxn_dn * step_size[:, None, None]

            current_t += step_size

            # Stop on samples in batch that are finished
            if current_t.max() >= 1:
                mask = (current_t >= 1).cpu() & (~valid_action_found)
                valid_action[mask] = sample[mask]
                valid_action_found[mask] = True
                num_steps_taken[mask] = i + 1

            if valid_action_found.all():
                break

        return sample

    def configure_optimizers(self, **kwargs) -> tuple[list[th.optim.Optimizer], list[th.optim.lr_scheduler._LRScheduler]]:
        """Return list of optimizers and list of schedulers."""
        decay_params, no_decay_params = self.model.network.configure_parameters()
        decay_params += list(self.model.rgb_encoder.parameters()) + list(self.model.var_est.parameters())
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
        optimizer = ScheduleFreeWrapper(optimizer_cls(optim_groups, **self.config.optimizer.kwargs))
        optimizer.train_mode = True

        return [optimizer], []