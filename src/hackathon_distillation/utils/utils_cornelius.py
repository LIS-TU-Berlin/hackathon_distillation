import collections
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




def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))



def to_device(x, device):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, d=device: x.to(d),
            type(None): lambda x: x,
        }
    )


def to_tensor(x):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x,
            np.ndarray: lambda x: torch.from_numpy(x),
            type(None): lambda x: x,
        }
    )
