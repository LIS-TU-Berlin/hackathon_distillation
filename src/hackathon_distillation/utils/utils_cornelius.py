import random
import torch
import numpy as np


def get_attribute_ddp(model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel, attribute: str):
    unwrapped_model = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        unwrapped_model = model.module
    return getattr(unwrapped_model, attribute)


def list_dict2dict_list(list_dict: list[dict[str, np.ndarray | list]]) -> dict[str, np.ndarray | list]:
    """Transpose list of dictionaries to dictionary of lists.

    Args:
        list_dict: list of dictionaries.

    Returns: A dictionary of lists or numpy arrays.
    """
    return {k: np.concatenate([dic[k] for dic in list_dict]) for k in list_dict[0]}


def set_global_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsample_data_dict(data_dict: dict, subsample_factor: int) -> dict:
    """
    Subsamples entries in data_dict that have more than one dimension.
    Keeps every nth step where n = subsample_factor.
    """
    for key, value in data_dict.items():
        # Check if the entry has more than 1 dimension and is a numpy array
        if isinstance(value, np.ndarray) and value.ndim > 2:
            # Subsample along the first axis (time axis)
            data_dict[key] = value[:, ::subsample_factor]
    return data_dict
