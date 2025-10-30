from dotenv import load_dotenv
load_dotenv()

import functools
import os
import warnings
import pathlib
import json

import hydra.utils
import torch as th
import torch.distributed as dist
import torch.nn as nn
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf, DictConfig
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from tqdm.auto import tqdm

from hackathon_distillation.dataset.compute_stats import compute_dataset_stats_welford
from hackathon_distillation.policy.ModelWrapperABC import TrainStrategy, ModelWrapper
from hackathon_distillation.policy.logger import LoggerCollection, TensorboardLogger
from hackathon_distillation.utils.utils_cornelius import set_global_seed, to_list, to_tensor, lists_to_tensors
from hackathon_distillation.dataset.dataset import BallImageDataset

ROOT = pathlib.Path(os.environ["REPO_PATH"])
CODEBASE_VERSION = os.environ.get("CODEBASE_VERSION")
DATA_PATH = pathlib.Path(os.environ.get("DATA_PATH"))


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Trainer:
    def __init__(
        self,
        *,
        pid: int,
        world_size: int,
        wrapper: ModelWrapper,
        strategy: TrainStrategy,
    ):
        self.pid = pid
        self.world_size = world_size
        self.strategy = strategy
        self.wrapper = wrapper
        self.model = wrapper.model.to(self.pid)
        self.model = self._initialize_model(self.model)   # automatically puts model on device
        self._is_setup = False
        self.is_debug = False

    def _initialize_model(self, model: nn.Module):
        match self.strategy:
            case TrainStrategy.fsdp:
                return self._toFSDP(model)
            case TrainStrategy.ddp:
                return self._toDDP(model)
            case TrainStrategy.cuda:
                return self._toCUDA(model)
            case _:
                raise ValueError("Invalid strategy")

    def _toFSDP(self, model: nn.Module):
        """Creates a fully parallel model, i.e., parallel data & model."""
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            recurse=True,
            min_num_params=int(2e9),
            # exclude_wrap_modules={Prototype}
        )
        return FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            use_orig_params=True,
            # forward_prefetch=True,
            # cpu_offload=CPUOffload(offload_params=True),
        )

    def _toDDP(self, model):
        """Creates a data parallel model."""
        return DDP(model, device_ids=[self.pid], find_unused_parameters=False)

    def _toCUDA(self, model):
        return model.to(f"cuda:{self.pid}")

    def _configure_optimizers(
        self,
        opt_state_dict_lst: list | None = None,
        lr_scheduler_state_dict_lst: list | None = None,
        **kwargs
    ):
        steps_per_epoch = kwargs.get("steps_per_epoch")
        print(f"Starting training for {steps_per_epoch * self.wrapper.config.epochs} steps")
        self.optimizers, self.lr_schedulers = self.wrapper.configure_optimizers(**kwargs)
        if opt_state_dict_lst is not None:
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(opt_state_dict_lst[i])
        if lr_scheduler_state_dict_lst is not None:
            for i, lr in enumerate(self.lr_schedulers):
                lr.load_state_dict(lr_scheduler_state_dict_lst[i])

    def summary(self, *, input_data: dict):
        if self.pid == 0:
            summary(
                self.model,
                verbose=1,
                input_data=input_data,
            )
            # print(f"{self.model}")

    def setup(
        self,
        run_name: str,
        logger_lst: list,
        ckpt_save_path: str | None = None,
        ckpt_save_every: int = 5,
        call_back_every: int = 1000,
        ckpt_save_max: int = 5,
        misc_save_path: str | None = None,
        val_every: int = 10,
        callbacks: list[callable] | None = None,
        steps_per_epoch: int = 299,
        is_debug: bool = False
    ) -> None:
        self._configure_optimizers(steps_per_epoch=steps_per_epoch)
        self.is_debug = is_debug
        self.logger = LoggerCollection(logger_lst) if not is_debug else None
        self.run_name = run_name
        self.ckpt_save_path = ROOT / ckpt_save_path / run_name
        self.ckpt_save_every = ckpt_save_every
        self.ckpt_save_max = ckpt_save_max
        self.misc_save_path = ROOT / misc_save_path
        self.val_every = val_every
        self.call_back_every = call_back_every
        self.callbacks = callbacks
        self.scaler = th.amp.GradScaler()
        self._is_setup = True

    def train(
        self,
        *,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        assert self._is_setup, "You must call trainer.setup() before training!"

        for epoch in range(1, epochs + 1):
            train_loader.sampler.set_epoch(epoch)
            self._train(train_loader, epoch)

            # Validation
            if self.val_every is not None and epoch % self.val_every == 0:
                self._validate(epoch, val_loader)

            # Save model
            if not self.is_debug and self.ckpt_save_path is not None and epoch % self.ckpt_save_every == 0:
                self._save_model(f"ep={epoch}")

            if self.call_back_every is not None and self.callbacks is not None and epoch % self.call_back_every == 0:
                for callback in self.callbacks:
                    callback(self.model)

        if self.ckpt_save_path is not None:
            self._save_model("last")

        if not self.is_debug:
            self.logger.finish()

    def _train(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        loader = tqdm(
            train_loader,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            colour="blue",
            leave=True,
            disable=dist.get_rank() != 0,
        )
        loader.set_description(f"Epoch {epoch}")

        ddp_loss = th.zeros(2).to(self.pid)
        for batch_idx, batch in enumerate(loader):
            loss_dict = self.wrapper.training_step(
                self.model, batch, self.optimizers, self.lr_schedulers, self.scaler
            )
            loss = loss_dict["loss"]
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

            if dist.get_rank() == 0:
                loader.set_postfix({"train/loss": loss.item()}, refresh=False)

            if not self.is_debug:
                for key, value in loss_dict.items():
                    self.logger.log_scalar(
                        f"train/{key}", value.item() if isinstance(value, th.Tensor) else value,
                        (epoch - 1) * self.world_size * len(loader) + (batch_idx + 1) * self.pid
                    )
                if len(self.lr_schedulers) > 0:
                    self.logger.log_scalar(
                        "lr",
                           self.lr_schedulers[0].get_last_lr()[0],
                           (epoch - 1) * self.world_size * len(loader) + (batch_idx + 1) * self.pid
                    )

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            loader.set_postfix({"train/loss": ddp_loss[0] / ddp_loss[1]}, refresh=False)
            loader.close()
            if not self.is_debug:
                self.logger.log_scalar("train/loss_epoch", ddp_loss[0] / ddp_loss[1], epoch)

    def _validate(self, epoch: int, val_loader: DataLoader | None = None):
        self.model.eval()
        with th.no_grad():
            loader = tqdm(
                val_loader,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                colour="blue",
                leave=True,
                disable=dist.get_rank() != 0,
            )
            loader.set_description(f"Validation on Epoch {epoch}")

            ddp_loss = th.zeros(2).to(self.pid)
            for _, batch in enumerate(loader):
                loss_dict = self.wrapper.validation_step(self.model, batch)
                loss = loss_dict["loss"]
                ddp_loss[0] += loss.item()
                ddp_loss[1] += 1

                if dist.get_rank() == 0:
                    loader.set_postfix({"val/loss": loss.item()}, refresh=False)

            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                loader.set_postfix({"val/loss": ddp_loss[0] / ddp_loss[1]}, refresh=False)
                loader.close()
                if not self.is_debug:
                    self.logger.log_scalar("val/loss", ddp_loss[0] / ddp_loss[1], epoch)

    def _save_model(self, ckpt_name: str):
        os.makedirs(self.ckpt_save_path, exist_ok=True)
        OmegaConf.save(self.wrapper.config, os.path.join(self.ckpt_save_path, "config.yaml"))  # store conf

        dist.barrier()
        if dist.get_rank() == 0:
            # Delete if too many checkpoints in path
            ckpt_files = [os.path.join(self.ckpt_save_path, ckpt_file) for ckpt_file in os.listdir(self.ckpt_save_path)]
            if len(ckpt_files) >= self.ckpt_save_max:
                file_to_remove = sorted(ckpt_files, key=os.path.getctime)[0]
                os.remove(file_to_remove)

        if self.strategy == TrainStrategy.fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": [FSDP.optim_state_dict(self.model, opt) for opt in self.optimizers],
                    #"lr_scheduler": [lr.state_dict() for lr in self.lr_schedulers],
                    "stats": self.wrapper.dataset_stats
                }
        elif self.strategy == TrainStrategy.ddp:
            state_dict = {
                "model": self.model.module.state_dict(),
                "optimizer": [opt for opt in self.optimizers],
                #"lr_scheduler": [lr for lr in self.lr_schedulers],
                "stats": self.wrapper.dataset_stats
            }
        else:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": [opt for opt in self.optimizers],
                #"lr_scheduler": [lr for lr in self.lr_schedulers],
                "stats": self.wrapper.dataset_stats
            }
        if dist.get_rank() == 0:
            th.save(
                state_dict,
                os.path.join(f"{self.ckpt_save_path}", f"{ckpt_name}.pt"),
            )


def _train_mp(pid: int, n_devices: int, cfg: DictConfig, run_name: str) -> None:
    set_global_seed(cfg.seed + pid)

    # Environment setup -- specify precisino etc for
    warnings.filterwarnings("ignore", category=UserWarning)
    #th.set_float32_matmul_precision("high")

    th.cuda.set_device(pid)
    setup(pid, n_devices, str(cfg.port))

    # Create dataset -- TODO: find proper split and collation etc
    # data_augmentations = img_transforms_wrapper(cfg)
    if cfg.pred_horizon % 2 != 0:
        print("Horizon is not a multiple of 2! Are you sure you use th proper config?")
    # train_episodes, val_episodes = train_test_split(list(range(len(ds_meta.episodes))), train_size=.8, random_state=cfg.seed)
    # diffusion-policy style synthetic dataset with flat obs keys

    # create dataset from file
    train_data = BallImageDataset(
        data_path=DATA_PATH / "new_data.zarr",
        # data_path=DATA_PATH / "test_data.zarr",
        horizon=cfg.pred_horizon,
        pad_before=cfg.obs_horizon-1,
        pad_after=cfg.action_horizon-1, 
        val_ratio=0.2,
    )
    val_data = train_data.get_validation_dataset()
    stats_file = DATA_PATH / "new_data_stats.pt"
    if stats_file.exists():
        with stats_file.open("r") as f:
            data_stats = th.load(stats_file)
    else:
        data_stats = compute_dataset_stats_welford(train_data)
        with stats_file.open("w") as f:
            data_stats = compute_dataset_stats_welford(train_data)
            th.save(data_stats, stats_file)
    print("Dataset creation finished.")

    # TODO: check with the samplers
    n_workers = 1
    pin_memory = False
    if cfg.strategy in ("ddp", "fsdp"):
        n_workers = 1
        pin_memory = True
    train_sampler = DistributedSampler(
        train_data,
        rank=pid,
        num_replicas=n_devices,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=n_workers,
        persistent_workers=True,    # must be True!
        pin_memory=pin_memory,            # must be True!
        drop_last=False,
        shuffle=False
    )

    val_sampler = DistributedSampler(
        val_data,
        rank=pid,
        num_replicas=n_devices,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=n_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    # else:
    #     # no ddp
    #     train_loader = DataLoader(
    #         train_data, batch_size=cfg.batch_size, shuffle=True,
    #         num_workers= max(4, os.cpu_count() // 2),
    #         prefetch_factor=4, persistent_workers=True,
    #         pin_memory=True, drop_last=True
    #     )
    #     val_loader = DataLoader(
    #         val_data, batch_size=cfg.batch_size, shuffle=False,
    #         num_workers= max(4, os.cpu_count() // 2),
    #         persistent_workers=True, pin_memory=True
    #     )

    # Get model & trainer
    wrapper = hydra.utils.get_class(cfg.model_type)(cfg, dataset_stats=data_stats)
    trainer = Trainer(
        pid=pid,
        world_size=n_devices,
        wrapper=wrapper,
        strategy=TrainStrategy[cfg.strategy]
    )

    # Run loop
    # TODO: add wandb logger code
    trainer.setup(
        logger_lst=[
            TensorboardLogger(
                DATA_PATH / cfg.log_dir,
                run_name=run_name
            ),
        ] if not cfg.is_debug else [],
        run_name=run_name,
        ckpt_save_path=cfg.ckpt_save_path,
        ckpt_save_every=cfg.save_every,
        ckpt_save_max=cfg.save_max,
        misc_save_path=cfg.misc_save_path,
        val_every=cfg.val_every,
        steps_per_epoch=len(train_loader),
        is_debug=cfg.is_debug
    )
    trainer.train(
        epochs=cfg.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    cleanup()
