# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import FunctionType
from typing import Any, Self

import numpy as np
import pandas as pd
import tensordict
import torch
from numpy import ndarray as ArrayType
from numpy.typing import NDArray
from pandas import DataFrame
from tensordict import TensorDict
from torch import Tensor

from aioway.errors import AiowayError
from aioway.schemas import Schema

from ..blocks.rows import Record
from ..buffers.tensors import Buffer

__all__ = ["Frame"]


def _unary_op[S: "Block"](op: Callable[[TensorDict], TensorDict]) -> FunctionType:
    def func(self: S) -> S:
        return type(self)(op(self.data))

    assert isinstance(func, FunctionType)
    return func


def _unary_tensor_op[S: "Block", E](op: Callable[[Tensor], E]) -> FunctionType:
    def func(self: S) -> S:
        mapping = {key: op(val) for key, val in self.data.items()}
        tensordict = self._tensordict_init(mapping)
        return type(self)(tensordict)

    assert isinstance(func, FunctionType)
    return func


_RHS = int | float | NDArray | Tensor


def _binary_op[
    S: "Frame"
](op: Callable[[TensorDict, S | _RHS | TensorDict], TensorDict]) -> FunctionType:
    def func(self: S, other: S | S | _RHS | TensorDict) -> S:
        if not isinstance(other, Frame):
            other_data = other
        else:
            other_data = other.data

        return op(self.data, other_data)

    assert isinstance(func, FunctionType)
    return func


def _binary_tensor_op[S: "Block"](op: Callable[[Tensor, _RHS], Tensor]) -> FunctionType:
    def func(self: S, other: S) -> S:
        if sorted(self.columns) != sorted(other.columns):
            raise KeyError(
                f"Blocks have different keys, element-wise operation not allowed! "
                f"{sorted(self)=}, {sorted(other)=}"
            )

        # Note:
        #   Making use of the implementation detail of ``Buffer`` being backed by ``Tensor``.
        #   If this function is to generalize, this shall be changed.
        mapping = {key: op(self[key].data, other[key].data) for key in self}
        tensordict = self._tensordict_init(mapping)
        return type(self)(tensordict)

    assert isinstance(func, FunctionType)
    return func


@dcls.dataclass(frozen=True)
class Frame:
    """
    ``Frame`` represents a chunk / batch of heterogenious data stored in memory,
    it is the main physical abstraction in ``aioway`` to represent eager computation.

    Think of it as a normal ``pandas.DataFrame`` or ``torch.Tensor`` or ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Todo:
        I have decided that ``Frame`` is abstract,
        and that it represents bounded, in-memory dataframe.

        This means that memory layouts like ``arrow``, ``pandas`` can easily be supported.

        However, to be fast, instead of serializing to python objects,
        we serialize to a concrete ``Batch`` object (to be introduced),
        that is a thin wrapper over ``TensorDict``, representing the current batch,
        allowing computing on GPUs.

        This way, UDFs can still be implemented, but native methods can be used as well.
    """

    data: TensorDict
    """
    The underlying data for the ``Frame`` class.

    Note:
        Since ``TensorDict`` itself is mutable,
        ``Block`` acts as an immutable wrapper such that only read operations are allowed.
        This would help the entire (pure) functional approach.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDict):
            raise TypeError(
                f"Expected data to be of type TensorDict, got {type(self.data)=}"
            )

    def __contains__(self, key: object) -> bool:
        return key in self.data.keys()

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    @typing.overload
    def __getitem__(self, key: str) -> Buffer: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> Self: ...

    @typing.overload
    def __getitem__(self, key: int) -> Record: ...

    @typing.overload
    def __getitem__(self, key: slice | list[int] | NDArray) -> Self: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._getitem_str(key)

        if isinstance(key, int):
            return self._getitem_int(key)

        if isinstance(key, slice):
            return self._slice_of_rows(key)

        if isinstance(key, ArrayType):
            return self._list_of_rows(key)

        if isinstance(key, list):
            # Note:
            #   Normally, here we would be using ``isinstance`` checks on each individual indices.
            #   However, doing that is very time consuming,
            #   and we would not want subclasses of ``int`` or ``str`` here anyways.
            #   For example, ``bool`` is a subclass of ``int``, but is undesired here.
            types = {type(i) for i in key}

            if types == {int}:
                return self._list_of_rows(key)

            if types == {str}:
                return self.project(*key)

            raise BlockGetItemTypeError(
                f"List must be a list of `int`s or `str`. Got {types}"
            )

        raise BlockGetItemTypeError(f"{type(key)=} is not supported!")

    def select(self, idx: list[int], /) -> Self:
        len_self = self.count()

        if idx and (min(idx) < -len_self or max(idx) >= len_self):
            out_of_bounds = [i for i in idx if i >= len_self or i < -len_self]
            raise IndexError(
                f"Index: {out_of_bounds} out of bounds for Block of length {len_self}."
            )

        return self._select(idx)

    def project(self, *cols: str) -> Self:
        if extras := set(cols).difference(names := self.schema().names):
            raise ValueError(
                f"Columns: {extras} specified, "
                f"but not found in schema's columns: {names}."
            )

        return self._project(*cols)

    @property
    def columns(self) -> list[str]:
        return self.schema().names

    __invert__ = _unary_op(operator.invert)
    __neg__ = _unary_op(operator.neg)

    __eq__ = _binary_op(operator.eq)
    __ne__ = _binary_op(operator.ne)

    __ge__ = _binary_tensor_op(operator.ge)
    __gt__ = _binary_tensor_op(operator.gt)
    __le__ = _binary_tensor_op(operator.le)
    __lt__ = _binary_tensor_op(operator.lt)

    __add__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __truediv__ = _binary_op(operator.truediv)
    __floordiv__ = _binary_op(operator.floordiv)
    __mod__ = _binary_op(operator.mod)
    __pow__ = _binary_op(operator.pow)

    max = _unary_tensor_op(lambda data: data.max().item())
    min = _unary_tensor_op(lambda data: data.min().item())
    all = _unary_tensor_op(lambda data: data.all())
    any = _unary_tensor_op(lambda data: data.any())

    def __index_no_wrap(self, idx):
        return self.data[idx]

    _getitem_str = _getitem_int = __index_no_wrap

    def __index_then_wrap(self, idx) -> Self:
        return type(self)(self.data[idx])

    _select = _slice_of_rows = _list_of_rows = __index_then_wrap

    def map(self, f: Callable[[TensorDict], TensorDict]) -> Self:
        return type(self)(f(self.data))

    def filter(self, predicate: Callable[[Tensor], Tensor], on: str) -> Self:
        keep = predicate(self.data[on])

        assert keep.dtype is torch.bool

        return type(self)(self.data[keep])

    def gather(self, dim: int, index: Sequence[int]) -> Self:
        return type(self)(self.data.gather(dim=dim, index=torch.tensor(index).long()))

    def rename(self, **names: str) -> Self:
        return type(self)(self.data.rename(**names))

    def _project(self, *names: str) -> Self:
        return type(self)(self.data.select(*names))

    def union(self, other: Self) -> Self:
        return type(self)(tensordict.cat([self.data, other.data], dim=0))

    def pandas(self) -> DataFrame:
        return DataFrame({key: self[key].to_pandas() for key in self})

    @property
    def dtype(self) -> str | None:
        return str(self.data.dtype) if self.data.dtype is not None else None

    def schema(self) -> Schema:
        """
        The schema type from the current block.
        """

        return Schema.mapping({key: self[key].dtype() for key in self})

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.data.batch_size

    @property
    def device(self) -> str | None:
        return str(self.data.device) if self.data.device is not None else None

    @device.setter
    def device(self, dev: str) -> None:
        self.to(dev)

    def count(self) -> int:
        return len(self.data)

    def reduce[I](self, f: Callable[[TensorDict, I], I], init: I) -> I:
        length = self.count()
        result = init

        for i in range(length):
            row = self._slice_of_rows(slice(i, i + 1))
            result = f(row, result)

        return result

    @property
    def requires_grad(self) -> bool:
        return self.data.requires_grad

    def to(self, device: str, /) -> Self:
        self.data.to(device)
        return self

    def to_tensordict(self) -> TensorDict:
        return self.data

    def to_pandas(self) -> DataFrame:
        data = self.data.cpu()
        np_dicts = {str(key): np.array(val) for key, val in data.items()}
        return DataFrame(np_dicts)

    def to_csv(self, path: str | Path, /) -> None:
        df = self.to_pandas()
        return df.to_csv(path)

    def _tensordict_init(
        self, mapping: dict[str, Any], names: list[str] | None = None
    ) -> TensorDict:
        return TensorDict(mapping, names=names)

    @classmethod
    def from_pandas(cls, df: DataFrame, /) -> Self:
        dicts = df.to_dict("series")
        np_dicts = {str(key): np.array(val) for key, val in dicts.items()}
        torch_dicts = {key: torch.from_numpy(val) for key, val in np_dicts.items()}
        return cls(TensorDict(torch_dicts, batch_size=len(df)))

    @classmethod
    def from_csv(cls, csv: str | Path, /) -> Self:
        df = pd.read_csv(csv)
        return cls.from_pandas(df)


class BlockGetItemTypeError(AiowayError, TypeError): ...


class BlockIndexOutOfBoundsError(AiowayError, IndexError): ...
