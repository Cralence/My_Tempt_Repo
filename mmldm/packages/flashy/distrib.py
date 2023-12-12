from contextlib import contextmanager
from functools import partial, wraps
import pickle
import typing as tp

import torch
from torch import distributed
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=distributed.ReduceOp.SUM):
    if is_distributed():
        return distributed.all_reduce(tensor, op)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()