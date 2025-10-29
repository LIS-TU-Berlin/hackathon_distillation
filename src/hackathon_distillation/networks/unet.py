"""Diffusion Unet from LeRobot:
https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/diffusion/modeling_diffusion.py#L187
"""
import math
from typing import Optional

import torch.random

import einops
import torch as th
from torch import nn
from torch import Tensor
from omegaconf import DictConfig


def _init_weights(model: nn.Module, seed: int = 123):
    """Initialize weights using Kaiming Normal Initialization and biases with zeros."""
    generator = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu', generator=generator)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu', generator=generator)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


class AttentionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UnetConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class UnetResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 0,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
        dropout: float = 0.
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = UnetConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.conv2 = UnetConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels)) if cond_dim > 0 else None

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding.
        #  Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        if self.cond_encoder is not None:
            cond_embed = self.cond_encoder(cond).unsqueeze(-1)
            if self.use_film_scale_modulation:
                # Treat the embedding as a list of scales and biases.
                scale = cond_embed[:, : self.out_channels]
                bias = cond_embed[:, self.out_channels :]
                out = scale * out + bias
            else:
                # Treat the embedding as biases.
                out = out + cond_embed

        out = self.dropout(out)
        out = self.conv2(out)
        out += self.residual_conv(x)
        return out


class ConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DictConfig, global_cond_dim: int):
        super().__init__()

        config = config.network
        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            AttentionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder.
        # For the decoder, we just reverse these.
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
            "dropout": config.dropout
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        UnetResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        UnetResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                UnetResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                UnetResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        # This automatically restores the right number of channels by doubling them
                        UnetResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        UnetResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            UnetConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        )

        # Initialize weights
        _init_weights(self)

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None, return_feat: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = th.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        feat = x.mean(dim=-1)  # bottleneck features for variance estimator

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = th.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")

        if return_feat:
            return x, feat

        return x

    def configure_parameters(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        # validate that we considered every parameter
        decay = [p for (pn, p) in self.named_parameters()]
        return decay, []

    @property
    def device(self):
        return next(self.parameters()).device


class UnconditionalUnetEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        cond_dim = 0
        input_shape = cfg.input_shapes["observation.state"]
        in_out = [(input_shape[-1], cfg.down_dims[0])] + list(
            zip(cfg.down_dims[:-1], cfg.down_dims[1:], strict=True)
        )
        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    UnetResidualBlock1d(dim_in, dim_out, cond_dim, cfg.kernel_size, cfg.n_groups),
                    UnetResidualBlock1d(dim_out, dim_out, cond_dim, cfg.kernel_size, cfg.n_groups),
                    nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=2, padding=1),  # Downsample
                )
                for dim_in, dim_out in in_out
            ]
        )

        # Global pooling and fully connected layers for mu and logvar
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc_mu = th.nn.Linear(cfg.down_dims[-1], cfg.z_channels)
        self.fc_logvar = th.nn.Linear(cfg.down_dims[-1], cfg.z_channels)

        _init_weights(self)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, D).

        Returns:
            Tuple[Tensor, Tensor]: Mean and log-variance tensors of shape (B, latent_dim).
        """
        # Rearrange to (B, D, T) for Conv1d
        x = einops.rearrange(x, "b t d -> b d t")

        # Pass through each encoder layer
        for layer in self.encoder_layers:
            x = layer(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # Shape: (B, C)
        print(x.shape)

        # Compute mu and logvar
        mu = self.fc_mu(x)  # Shape: (B, latent_dim)
        logvar = self.fc_logvar(x)  # Shape: (B, latent_dim)
        return mu, logvar


class UnconditionalUnetDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Actual net
        cond_dim = 0
        input_shape = cfg.input_shapes["observation.state"]
        in_out = list(
            zip(cfg.down_dims[:-1], cfg.down_dims[1:], strict=True)
        )
        self.decoder_layers = nn.ModuleList([])
        for (dim_out, dim_in) in reversed(in_out):
            self.decoder_layers.extend(
                [
                    UnetResidualBlock1d(dim_in, dim_out, cond_dim, cfg.kernel_size, cfg.n_groups),
                    UnetResidualBlock1d(dim_out, dim_out, cond_dim, cfg.kernel_size, cfg.n_groups),
                    nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
                ]
            )

        # latent to net projection
        self.initial_time = int(input_shape[0] / 2 ** (len(cfg.down_dims) - 1))
        self.decoder_input = th.nn.Linear(cfg.z_channels, in_out[-1][-1] * self.initial_time)

        # Final convolution to match input dimensions
        self.final_conv = nn.Sequential(
            UnetConv1dBlock(cfg.down_dims[0], cfg.down_dims[0], cfg.kernel_size, cfg.n_groups),
            nn.Conv1d(cfg.down_dims[0], input_shape[-1], kernel_size=1)
        )

        _init_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Latent tensor of shape (B, latent_dim).

        Returns:
            Tensor: Reconstructed tensor of shape (B, T, D).
        """
        x = self.decoder_input(x)
        x = x.view(x.size(0), -1, self.initial_time)

        # Pass through each decoder layer
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)

        # Final convolution to get back to input dimensions
        x = self.final_conv(x)  # Shape: (B, D, T)

        # Rearrange back to (B, T, D)
        x = einops.rearrange(x, "b d t -> b t d")
        return x
