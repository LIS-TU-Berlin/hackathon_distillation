from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor], 
        dtype: torch.dtype = torch.float32,
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func, dtype=dtype)
        else:
            result[key] = func(value).to(dtype)
    return result