from typing import Dict
import torch
import numpy as np
import copy
from hackathon_distillation.data_loader.pytorch_util import dict_apply
from hackathon_distillation.data_loader.replay_buffer import ReplayBuffer
from hackathon_distillation.data_loader.sampler import SequenceSampler, get_val_mask, downsample_mask
from hackathon_distillation.networks.base_models import Normalizer


class BaseImageDataset(torch.utils.data.Dataset):
    def get_normalizer(self, **kwargs) -> Normalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()

class BallImageDataset(BaseImageDataset):
    def __init__(self,
            data_path: str, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.obs_horizon = pad_before + 1
        self.action_horizon = pad_after + 1

        self.replay_buffer = ReplayBuffer.copy_from_path(
            data_path, keys=['depth', 'ee_pos', 'ee_action', 'rgb'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        # TODO: remove this after testing 

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    # def get_normalizer(self, mode='limits', **kwargs):
    #     data = {
    #         'action': self.replay_buffer['action'],
    #         'agent_pos': self.replay_buffer['state'][...,:2]
    #     }
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
    #     normalizer['image'] = get_image_range_normalizer()
    #     return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        rgb = sample['rgb']  # T, 360, 640, 3
        rgb_transposed = np.moveaxis(rgb, -1,1)  # T, 3, 360, 640
        data = {
            'obs.img': sample['depth'][:self.obs_horizon], # T, 360, 640
            'obs.state': sample['ee_pos'][:self.obs_horizon], # T, 3
            #'obs.img': rgb_transposed[:self.obs_horizon], # T, 3, 360, 640
            'action': sample['ee_action'] # T, 3
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == '__main__':
    data_path = '/home/data/hackathon/data.zarr'

    # parameters
    pred_horizon = 12
    obs_horizon = 2
    action_horizon = 5

    # create dataset from file
    dataset = BallImageDataset(
        data_path=data_path,
        horizon=pred_horizon,
        pad_before=obs_horizon-1,
        pad_after=action_horizon-1
    )
    # # save training data statistics (min, max) for each dim
    # stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=False,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # # visualize data in batch
    batch = next(iter(dataloader))
    print("obs['depth'].shape:", batch['obs.img'].shape)
    print("obs['rgb'].shape:", batch['obs.rgb'].shape)
    print("obs['ee_pos'].shape:", batch['obs.state'].shape)
    print("batch['ee_action'].shape", batch['action'].shape)