# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from abc import ABC
from collections.abc import Callable, Iterable
from typing import ClassVar, TypeGuard

from aioway.attrs import Shape

from . import _funcs, _tensors
from .arrays import Array

__all__ = [
    "Kernel",
    "PermuteKernel",
    "AggKernel",
    "BroadCastSameKernel",
    "Matmul2dKernel",
]


@dcls.dataclass(frozen=True)
class Kernel(ABC):
    INPUT_TYPES: ClassVar[tuple[type[Array], ...]]
    OUTPUT_TYPE: ClassVar[type[Array]]

    def __post_init__(self):
        if len(tuple(self.inputs())) != len(self.INPUT_TYPES):
            raise TypeError(f"The input number mis-matches with {self.INPUT_TYPES}")

        for dat, typ in zip(self.inputs(), self.INPUT_TYPES):
            if not isinstance(dat, typ):
                raise ValueError(f"{dat} not instance of {typ}")

    def __call__(self) -> Array:
        input_costs = sum(i.cost for i in self.inputs())
        result = self.compute()
        result.cost += input_costs
        return result

    @abc.abstractmethod
    def compute(self) -> Array: ...

    @abc.abstractmethod
    def inputs(self) -> Iterable[Array]: ...


@dcls.dataclass(frozen=True)
class PermuteKernel(Kernel):
    INPUT_TYPES = (Array,)
    OUTPUT_TYPE = Array

    source: Array
    permutation: list[int]

    @typing.override
    def compute(self) -> Array:
        return Array(
            shape=_funcs.permute(self.source.shape, self.permutation),
            dtype=self.source.dtype,
            cost=self.source.shape.size,
        )

    @typing.override
    def inputs(self) -> Iterable[Array]:
        yield self.source


@dcls.dataclass(frozen=True)
class AggKernel(Kernel):
    INPUT_TYPES = (Array,)
    OUTPUT_TYPE = Array

    source: Array
    dims: list[int]

    @typing.override
    def compute(self) -> Array:
        return Array(
            shape=_funcs.agg(self.source.shape, self.dims),
            dtype=self.source.dtype,
            cost=self.source.shape.size,
        )

    @typing.override
    def inputs(self) -> Iterable[Array]:
        yield self.source


@dcls.dataclass(frozen=True)
class BroadCastSameKernel(Kernel):
    INPUT_TYPES = Array, Array
    OUTPUT_TYPE = Array

    left: Array
    right: Array

    @typing.override
    def compute(self) -> Array:
        shape = _funcs.bcast_same_dim(self.left.shape, self.right.shape)
        dtype = _tensors.promote_dtype(str(self.left.dtype), str(self.right.dtype))
        buffer_size = Shape.wrap(
            [max(l, r) for l, r in zip(self.left.shape, self.right.shape)]
        ).size
        cost = buffer_size * dtype.bits
        return Array(shape=shape, dtype=dtype, cost=cost)

    @typing.override
    def inputs(self) -> Iterable[Array]:
        yield self.left
        yield self.right


@dcls.dataclass(frozen=True)
class Matmul2dKernel(Kernel):
    INPUT_TYPES = Array, Array
    OUTPUT_TYPE = Array

    left: Array
    right: Array

    @typing.override
    def compute(self) -> Array:
        shape = _funcs.matmul_2d(self.left.shape, self.right.shape)
        dtype = _tensors.promote_dtype(str(self.left.dtype), str(self.right.dtype))
        cost = shape.size * dtype.bits
        return Array(shape=shape, dtype=dtype, cost=cost)

    @typing.override
    def inputs(self) -> Iterable[Array]:
        yield self.left
        yield self.right


def _signature_check(kernel: Callable) -> TypeGuard[Kernel]:
    if not callable(kernel):
        raise ValueError(f"{kernel} not callable.")

    inspect.signature(kernel)

    raise NotImplementedError
