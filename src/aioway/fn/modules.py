# Copyright (c) AIoWay Authors - All Rights Reserved

"The preview module protocol definition."

import functools
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway import _common, ctx
from aioway._tracking import logging

from . import de, fn, tensors

__all__ = ["ModuleFn", "FakableModule"]

LOGGER = logging.get_logger(__name__)


class FakableModule[**P, M: nn.Module]:
    """
    `FakableModule` allows `nn.Module` to be used with fake mode.

    Example:
        `FakableModule(torch.nn.Linear, in_features=1, out_features=2)`.
        Note that the arguments following the first type argument will be passed into the `nn.Module` type,
        so they are exactly as those on the `torch` documentation.
    """

    def __init__(
        self, nn: cabc.Callable[P, M], /, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self._module = nn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, tensor: torch.Tensor | tensors.TensorFn, /):
        tensor = de.eager(tensor)
        return self.module(tensor)

    @functools.cached_property
    @ctx.fake_mode_func
    def fake(self) -> M:
        return self._module(*self._args, **self._kwargs)

    @functools.cached_property
    def real(self) -> M:
        return self._module(*self._args, **self._kwargs)

    @property
    def module(self):
        return self.fake if ctx.enabled_fake_mode() else self.real

    def parameters(self) -> cabc.Generator[nn.Parameter]:
        yield from self.module.parameters()


@_common.dcls_no_eq_no_repr
class ModuleFn[**P, M: nn.Module](tensors.TensorFn):
    """
    `ModuleFn` informs us how `nn.Module` would behave without initializing it.
    """

    tensor: tensors.TensorFn
    builder: FakableModule[P, M]

    @typing.override
    def deps(self) -> tuple[fn.Fn[object], ...]:
        return tuple(self._deps())

    def _deps(self):
        yield self.tensor
        yield from self.params_fn()

    @typing.override
    def forward(self) -> torch.Tensor:
        tensor = self.tensor.do()
        module = self.builder.module
        return module(tensor)

    def parameters(self) -> cabc.Generator[nn.Parameter]:
        yield from self.builder.parameters()

    def params_fn(self):
        for param in self.parameters():
            yield de.defer(param)
