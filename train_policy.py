from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path

import torch as th
import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig
from rich.pretty import pprint

from hackathon_distillation.policy.trainer import _train_mp
from hackathon_distillation.policy.utils.logger import _get_run_name


REPO_PATH = Path(os.environ["REPO_PATH"])
DATA_PATH = Path(os.environ["DATA_PATH"])
config_path = str(REPO_PATH / "config")


@hydra.main(config_path=config_path, config_name="mlp", version_base=None)
def train_cli(cfg: DictConfig) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(d) for d in cfg.devices])

    pprint(cfg)
    n_devices = th.cuda.device_count()
    print(f"Training on {n_devices} devices.")
    run_name = _get_run_name(DATA_PATH / cfg.log_dir, cfg.get("run_name")) if not cfg.is_debug else "test"  # TODO: fix perhaps
    mp.spawn(
        _train_mp,
        args=(
            n_devices,
            cfg,
            run_name
        ),
        nprocs=n_devices,
        join=True,
    )


if __name__ == "__main__":
    train_cli()
