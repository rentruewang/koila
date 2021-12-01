from __future__ import annotations

import dataclasses as dcls
import math
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.types import Number


@dataclass(init=False)
class ArgsKwargs:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    args: Sequence[Any] = dcls.field(default_factory=tuple)
    kwargs: Dict[str, Any] = dcls.field(default_factory=dict)


@dataclass(init=False)
class Caller:
    func: Callable[..., Any]
    arguments: Sequence[ArgsKwargs] = dcls.field(default_factory=list)

    def __init__(
        self,
        func: Callable[..., Any],
        arguments: Sequence[ArgsKwargs | Sequence[Any] | Dict[str, Any]],
    ) -> None:
        self.func = func
        self.arguments = []

        for argument in arguments:
            if isinstance(argument, Sequence):
                argument = ArgsKwargs(*argument)

            if isinstance(argument, dict):
                assert all(isinstance(key, str) for key in argument.keys())
                argument = ArgsKwargs(**argument)

            self.arguments.append(argument)

    def call(self) -> None:
        for argument in self.arguments:
            self.func(*argument.args, **argument.kwargs)


def call(
    func: Callable[..., Any],
    arguments: Sequence[ArgsKwargs | Sequence[Any] | Dict[str, Any]],
) -> None:
    Caller(func, arguments=arguments).call()


def assert_equal(
    input: Tensor | ndarray | Number, other: Tensor | ndarray | Number
) -> None:
    if isinstance(input, ndarray) or isinstance(other, ndarray):
        assert np.all(input == other), input != other
        return

    if isinstance(input, Tensor) or isinstance(other, Tensor):
        assert typing.cast(Tensor, input == other).all(), input != other
        return

    assert input == other, [input, other]


def assert_isclose(
    input: Tensor | ndarray | Number, other: Tensor | ndarray | Number
) -> None:
    if isinstance(input, ndarray) or isinstance(other, ndarray):
        assert np.allclose(input, other, atol=1e-5), [input, other]
        return

    if isinstance(input, Tensor) and isinstance(other, Tensor):
        assert torch.allclose(input, other, atol=1e-5), [input, other]
        return

    assert math.isclose(input, other, abs_tol=1e-5), [input, other]


def is_notimplemented(func: Callable[[], Any]) -> bool:
    try:
        func()
        return False
    except:
        return True
