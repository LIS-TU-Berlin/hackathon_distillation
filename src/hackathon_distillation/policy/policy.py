#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"""

from collections import deque
from pathlib import Path

import hydra.utils
import numpy as np
import torch as th
from omegaconf import OmegaConf
from torch import nn, Tensor

from hackathon_distillation.policy.utils.normalize import Normalize, Unnormalize
from hackathon_distillation.utils.utils_cornelius import to_device


def populate_queues(queues, batch):
    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


class Policy(nn.Module):
    """
    Reactive Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "policy"

    def __init__(
        self,
        checkpoint_path: Path,
        datastat_path: Path,
        map_location: str | th.device = "cpu",
    ):
        super().__init__()

        self._queues = None

        cfg = OmegaConf.load(f"{checkpoint_path.parent}/config.yaml")
        self.policy = hydra.utils.get_class(cfg.model_type).from_checkpoint(checkpoint_path, map_location)
        #self.expected_image_keys = [k for k in self.policy.config.network.input_shapes if k.startswith("obs.img")]  # todo: refactor keys
        self.expected_image_keys = ["obs.depth", "obs.mask"]
        self.use_env_state = "obs.state" in self.policy.config.network.input_shapes
        with datastat_path.open("r") as f:
            data_stats = th.load(datastat_path)
        self.policy.model.normalize_inputs = Normalize(
            self.policy.config.network.input_shapes, self.policy.config.network.input_normalization_modes, data_stats
        )
        self.policy.model.normalize_targets = Normalize(
            self.policy.config.network.output_shapes, self.policy.config.network.output_normalization_modes, data_stats
        )
        self.policy.model.unnormalize_outputs = Unnormalize(
            self.policy.config.network.output_shapes, self.policy.config.network.output_normalization_modes, data_stats
        )
        self.policy.model.to(map_location)

        self.reset()

    @property
    def torch_module(self):
        return self.policy.model

    @property
    def config(self):
        return self.policy.config

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "action": deque(maxlen=self.config.action_horizon),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["obs.img"] = deque(maxlen=self.config.obs_horizon)
        if self.use_env_state:
            self._queues["obs.state"] = deque(maxlen=self.config.obs_horizon)
        self._queues["obs.depth"] = deque(maxlen=self.config.obs_horizon)
        self._queues["obs.mask"] = deque(maxlen=self.config.obs_horizon)
        self.policy.prior_sample = None

    @th.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> np.ndarray:
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            # batch["obs.depth"] = th.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        # actual prediction
        if len(self._queues["action"]) == 0:
            batch = {k: th.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            batch = to_device(batch, device=self.torch_module.device)
            batch = self.torch_module.normalize_inputs(batch)
            # batch["obs.img"] = batch["obs.img"].squeeze(2)
            actions = self.policy.generate_actions(batch)
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action.detach().cpu().numpy()

