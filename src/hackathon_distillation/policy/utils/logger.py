import os
import shutil
from abc import ABC, abstractmethod

import yaml


def _get_run_name(log_path: str, run_name: str = None) -> str:
    if run_name is None:
        prefix = "run_"
        for i in range(1, 100):
            path_cand = os.path.join(log_path, f"{prefix}{i:02d}")
            if not os.path.exists(path_cand):
                os.makedirs(path_cand, exist_ok=True)
                break
        return f"{prefix}{i:02d}"
    else:
        path = os.path.join(log_path, run_name)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            os.mkdir(path)
        return run_name

def save_config(log_path: str, run_name: str, config: dict):
    yaml.Dumper.ignore_aliases = lambda *args : True
    with open(os.path.join(log_path, run_name, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(file_path: str):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def prepare_folder(log_dir, run_name):
    ckpt_save_path = os.path.join(log_dir, run_name, "checkpoints")
    misc_save_path = os.path.join(log_dir, run_name, "misc")
    if not (os.path.isdir(ckpt_save_path)):
        os.makedirs(ckpt_save_path, exist_ok=True)

    if not (os.path.isdir(misc_save_path)):
        os.makedirs(misc_save_path, exist_ok=True)

    return ckpt_save_path, misc_save_path

class AbstractLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag, scalar_value, global_step):
        raise NotImplementedError

    @abstractmethod
    def log_volume(self, tag, obj3d_file_path, global_step):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        pass


class LoggerCollection(AbstractLogger):
    def __init__(self, loggers: list):
        self.loggers = loggers

    def log_scalar(self, tag, scalar_value, global_step):
        for logger in self.loggers:
            logger.log_scalar(tag, scalar_value, global_step)

    def log_volume(self, tag, obj3d_file_path, global_step):
        for logger in self.loggers:
            logger.log_volume(tag, obj3d_file_path, global_step)

    def finish(self):
        for logger in self.loggers:
            logger.finish()


class TensorboardLogger(AbstractLogger):
    def __init__(self, log_dir, run_name=None):
        from torch.utils.tensorboard import SummaryWriter

        if run_name is None:
            run_name = _get_run_name(log_dir)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    def log_scalar(self, tag, scalar_value, gloabl_step):
        self.summary_writer.add_scalar(tag, scalar_value, gloabl_step)

    def log_volume(self, tag, obj3d_file_path, global_step):
        pass

    def finish(self):
        self.summary_writer.close()
