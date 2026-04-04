# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import numpy as np
import tensordict as td
import torch

__all__ = ["defer"]


@typing.no_type_check
def defer(value):
    from aioway import tensors

    if value is None:
        return None

    if isinstance(value, int | float | bool | slice):
        return value

    if isinstance(value, tensors.TensorFn | tensors.TensorDictFn):
        return value

    if isinstance(value, np.ndarray):
        return defer(torch.from_numpy(value))

    if isinstance(value, torch.Tensor):
        return tensors.TensorFn.from_tensor(value)

    if isinstance(value, td.TensorDict):
        return tensors.tdict(value)

    if isinstance(value, cabc.Sequence) and len({type(i) for i in value}) <= 1:
        return tensors.TensorFn.from_tensor(torch.as_tensor(value))

    if isinstance(value, cabc.Mapping):
        return tensors.tdict(td.TensorDict.from_any(value))

    raise TypeError(f"We do not know how to handle: {value=}, {type(value)=}")
