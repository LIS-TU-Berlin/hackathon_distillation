import torch 
from hackathon_distillation.data_loader.dataset import BallImageDataset

if __name__ == '__main__':
    data_path = '/home/data/hackathon/data.zarr'


    # parameters
    pred_horizon = 1
    obs_horizon = 1
    action_horizon = 1

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
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # # visualize data in batch
    batch = next(iter(dataloader))
    obs = batch['obs']
    print("obs['depth'].shape:", obs['depth'].shape)
    print("obs['ee_pos'].shape:", obs['ee_pos'].shape)
    print("batch['ee_action'].shape", batch['ee_action'].shape)