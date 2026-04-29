from typing import Sequence

import torch
import torch.nn as nn


def get_number_of_parameters(module: nn.Module) -> int:
    """
    Returns the number of parameters in the module.
    :param module: PyTorch Module.
    :return: Number of parameters in the module.
    """
    return sum(p.numel() for p in module.parameters())


def get_use_gpu(disable_gpu: bool = False) -> bool:
    """
    Returns true if cuda is available and no explicit disable cuda flag given.
    """
    return torch.cuda.is_available() and not disable_gpu


def get_device(disable_gpu: bool = False, cuda_id: int = 0):
    """
    Returns a gpu cuda device if available and cpu device otherwise.
    """
    if get_use_gpu(disable_gpu):
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def compute_grad_norm(module: nn.Module, detach: bool = False):
    """
    Returns the Euclidean norm of the current gradients.
    :param module: pytorch module.
    :param detach: whether to detach gradients from computation graph before computing the norm.
    """
    grads = []
    for param in module.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().flatten() if detach else param.grad.flatten())

    return torch.cat(grads).norm()


def compute_params_norm(module: nn.Module):
    """
    Returns the Euclidean norm of the module parameters.
    @param module: pytorch module
    """
    params = list(get_parameters_iter(module))
    flattened_params_vector = torch.cat([param.view(-1) for param in params])
    return flattened_params_vector.norm()


def get_parameters_iter(module: nn.Module, exclude: Sequence[type] = None, include_only: Sequence[type] = None,
                        exclude_by_name_part: Sequence[str] = None) -> Sequence[nn.Parameter]:
    """
    Gets an iterator that iterates over the module parameters.
    @param module: module to get an iterator over its parameters.
    @param exclude: sequence of module types to exclude.
    @param include_only: sequence of module types to include only. If None, then will include by default all layer types.
    @param exclude_by_name_part: sequence of strings to exclude parameters which include one of the given names in as part of their name.
    @return iterator over module parameters.
    """
    exclude_by_name_part = exclude_by_name_part if exclude_by_name_part is not None else []
    for child_module in module.modules():
        if include_only is not None and child_module.__class__ not in include_only:
            continue

        if exclude is not None and child_module.__class__ in exclude:
            continue

        for name, param in child_module.named_parameters(recurse=False):
            exclude_param = any([exclude_name_part in name for exclude_name_part in exclude_by_name_part])
            if exclude_param:
                continue

            yield param
