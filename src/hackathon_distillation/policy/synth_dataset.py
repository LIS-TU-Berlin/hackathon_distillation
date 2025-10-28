import torch as th
from copy import copy
from torch.utils.data.dataset import Dataset


class _DPSyntheticDataset(Dataset):
    """
    Synthetic diffusion-policy style dataset.
    Output keys:
        obs.img: (T_obs, C, H, W), float32 in [0,1]
        obs.state: (T_obs, D)
        action:    (T,    A)
    Sequence sampled inside episode boundaries.
    """
    def __init__(
        self,
        *,
        horizon: int,
        n_obs_steps: int | None,
        n_latency_steps: int = 0,
        n_episodes: int = 8,
        episode_len: int = 50,
        val_ratio: float = 0.2,
        seed: int = 42,
        which_split: str = "train",
        img_shape=(3, 64, 64),
        state_dim=4,
        action_dim=2,
    ):
        import numpy as np
        assert which_split in {"train", "val"}

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.which_split = which_split
        self.n_episodes = n_episodes
        self.episode_len = episode_len

        C, H, W = img_shape
        N = n_episodes * episode_len
        self.episode_ends = np.cumsum([episode_len] * n_episodes)

        # store raw arrays (time-major)
        self._buf_img = th.randint(0, 256, size=(N, H, W, C), dtype=th.float32)
        self._buf_state = th.randn((N, state_dim), dtype=th.float32) * 0.2
        actions = th.randn((N, action_dim), dtype=th.float32)
        self._buf_action = actions

        # train/val split by episode
        ep_ids = th.arange(n_episodes)
        n_val = max(1, int(round(n_episodes * val_ratio)))
        val_eps = set(ep_ids[:n_val])
        train_eps = set(ep_ids[n_val:])

        keep = train_eps if which_split == "train" else val_eps
        self.mask = self._episode_mask(keep)
        self._build_index()

        # dataset "stats" for normalizer compatibility
        self.stats = {
            "obs.img": {
                "mean": th.mean(self._buf_img, axis=0),
                "std": th.std(self._buf_img, axis=0) + 1e-6,
                "min": th.min(self._buf_img, axis=0)[0],
                "max": th.max(self._buf_img, axis=0)[0],
            },
            "obs.state": {
                "mean": th.mean(self._buf_state, axis=0),
                "std": th.std(self._buf_state, axis=0) + 1e-6,
                "min": th.min(self._buf_state, axis=0)[0],
                "max": th.max(self._buf_state, axis=0)[0],
            },
            "action": {
                "mean": th.mean(actions, axis=0),
                "std": th.std(actions, axis=0) + 1e-6,
                "min": th.min(actions, axis=0)[0],
                "max": th.max(actions, axis=0)[0],
            },
        }

    def _episode_mask(self, keep_eps: set) -> th.Tensor:
        mask = th.zeros(self.n_episodes, dtype=th.bool)
        for e in keep_eps: mask[e] = True
        return mask

    def _build_index(self):
        idxs = []
        start = 0
        for e, end in enumerate(self.episode_ends):
            if not self.mask[e]:
                start = end
                continue
            max_start = max(start, end - (self.horizon + self.n_latency_steps))
            for t in range(start, max_start):
                idxs.append(t)
            start = end
        self._starts = th.tensor(idxs, dtype=th.int32)

    def __len__(self): return len(self._starts)

    def __getitem__(self, idx: int):
        s = int(self._starts[idx])
        T = self.horizon + self.n_latency_steps
        sl = slice(s, s + T)

        # select and slice observation horizon
        T_obs = slice(None if self.n_obs_steps is None else 0,
                      None if self.n_obs_steps is None else self.n_obs_steps)

        # image: (T,C,H,W)
        img = self._buf_img[sl] / 255.0
        #img = th.moveaxis(img, -1, 1)
        img = img[T_obs]

        # state: (T,D)
        state = self._buf_state[sl][T_obs]

        # action: (T,A) then drop latency
        act = self._buf_action[sl]
        if self.n_latency_steps > 0:
            act = act[self.n_latency_steps:]

        return {
            "obs.img": img,
            "obs.state": state,
            "action": act,
        }

    def get_validation_dataset(self):
        val = copy(self)
        val.which_split = "val"
        val.mask = ~self.mask
        val._build_index()
        return val

    def get_all_actions(self):
        return th.from_numpy(self._buf_action)
