# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Mapping, Sequence
from typing import Any

from tensordict import TensorDict
from torch import Tensor

from aioway.tdicts import TensorDictFn
from aioway.tensors import TensorFn

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
def defer(value: Tensor | TensorFn) -> TensorFn: ...
@typing.overload
def defer(value: TensorDict | TensorDictFn) -> TensorDictFn: ...
@typing.overload
def defer(value: Sequence[Any]) -> TensorFn: ...
@typing.overload
def defer(value: Mapping[str, Any]) -> TensorDictFn: ...
