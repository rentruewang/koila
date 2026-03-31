# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Mapping, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor

from aioway import _typing

__all__ = ["defer"]


@typing.no_type_check
def defer(value):
    from aioway.tdicts import TensorDictFn
    from aioway.tensors import TensorFn

    if value is None:
        return None

    if isinstance(value, int | float | bool | slice):
        return value

    if isinstance(value, TensorFn | TensorDictFn):
        return value

    if _typing.is_array(value):
        return torch.from_numpy(value)

    if isinstance(value, Tensor):
        return TensorFn.from_tensor(value)

    if isinstance(value, TensorDict):
        return TensorDictFn.from_tensordict(value)

    if isinstance(value, Sequence) and len({type(i) for i in value}) <= 1:
        return TensorFn.from_tensor(torch.as_tensor(value))

    if isinstance(value, Mapping):
        return TensorDictFn.from_tensordict(TensorDict.from_any(value))

    raise TypeError(f"We do not know how to handle: {value=}, {type(value)=}")
