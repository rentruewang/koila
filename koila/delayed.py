from __future__ import annotations

import dataclasses as dcls
import logging
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Dict, Tuple, final

from numpy import ndarray
from rich.logging import RichHandler
from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from . import immediate
from .prepasses import PrePass
from .runnables import Runnable
from .tensors import TensorLike

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())

# FIXME: Currently disregards RunnableTensor API


@final
@dataclass(init=False)
class DelayedTensor(TensorLike):
    func: Callable[..., Tensor]
    prepass: PrePass
    args: Tuple[Runnable[Any], ...] = dcls.field(default_factory=tuple)
    kwargs: Dict[str, Runnable[Any]] = dcls.field(default_factory=dict)

    def __init__(
        self,
        func: Callable[..., Tensor],
        prepass: PrePass,
        *args: Runnable[Tensor] | Tensor | Number,
        **kwargs: Runnable[Tensor] | Tensor | Number,
    ) -> None:
        self.func = func
        self.prepass = prepass
        self.args = tuple(delayed_input(arg) for arg in args)
        self.kwargs = dict((k, delayed_input(v)) for (k, v) in kwargs.items())

    def __hash__(self) -> int:
        # Evaluations are unique.
        return id(self)

    # def run(self, partial: Tuple[int, int] | None = None) -> Tensor:
    def run(self) -> Tensor:
        real_args = [arg.run() for arg in self.args]
        real_kwargs = {k: v.run() for (k, v) in self.kwargs.items()}

        result = self.func(*real_args, **real_kwargs)

        # Checks the shape only when pre-passing.
        # If partial is supplemented, it means the tensors are really evaluated
        # if partial is None:
        #     assert self.prepass.shape == result.shape, [self.prepass, result.shape]
        # elif (reducer := self.prepass.reducer()) is None:
        #     raise UnsupportedError("Cannot safely parallelize.")
        # else:
        #     logger.debug(
        #         "Evaluation taking batch: (%s, %s), low=%s, high=%s",
        #         self.size(),
        #         self.batch(),
        #         partial[0],
        #         partial[1],
        #     )
        #     callback = reducer(input, *self.args, **self.kwargs)
        #     result = callback(result)

        assert self.prepass.shape == result.shape
        return result

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        shape = self.prepass.shape
        if dim is not None:
            return shape[dim]
        else:
            return shape

    @property
    def dtype(self) -> DType:
        return self.prepass.dtype

    @property
    def device(self) -> str | Device:
        return self.prepass.device


def delayed_input(input: Runnable[Any] | Tensor | ndarray | Number) -> Runnable[Any]:
    if isinstance(input, Runnable):
        return input

    return immediate.wrap(input)
