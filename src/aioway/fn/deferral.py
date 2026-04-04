# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc
from typing import Any

import numpy as np
import tensordict as td
import torch
from numpy import typing as npt

from . import tdicts, tensors

__all__ = ["defer"]


@typing.overload
def defer(value: bool) -> bool: ...


@typing.overload
def defer(value: int) -> int: ...


@typing.overload
def defer(value: float) -> float: ...


@typing.overload
def defer(value: slice) -> slice: ...


@typing.overload
def defer(value: None) -> None: ...


@typing.overload
def defer(value: npt.NDArray) -> tensors.TensorFn: ...


@typing.overload
def defer(value: torch.Tensor | tensors.TensorFn) -> tensors.TensorFn: ...


@typing.overload
def defer(value: td.TensorDict | tdicts.TensorDictFn) -> tdicts.TensorDictFn: ...


@typing.overload
def defer(value: cabc.Sequence[Any]) -> tensors.TensorFn: ...


@typing.overload
def defer(value: cabc.Mapping[str, Any]) -> tdicts.TensorDictFn: ...


def defer(value):

    if value is None:
        return None

    if isinstance(value, int | float | bool | slice):
        return value

    if isinstance(value, tensors.TensorFn | tdicts.TensorDictFn):
        return value

    if isinstance(value, np.ndarray):
        return defer(torch.from_numpy(value))

    if isinstance(value, torch.Tensor):
        return tensors.tensor(value)

    if isinstance(value, td.TensorDict):
        return tdicts.tdict(value)

    if isinstance(value, cabc.Sequence) and len({type(i) for i in value}) <= 1:
        return tensors.tensor(torch.as_tensor(value))

    if isinstance(value, cabc.Mapping):
        return tdicts.tdict(td.TensorDict.from_any(value))

    raise TypeError(f"We do not know how to handle: {value=}, {type(value)=}")
