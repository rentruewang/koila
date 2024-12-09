# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections import defaultdict as DefaultDict
from collections.abc import Callable, Hashable, Iterator, Sequence
from pathlib import Path
from types import FunctionType
from typing import Any, Self

import numpy as np
import pandas as pd
import tensordict
import torch
from pandas import DataFrame
from tensordict import TensorDict
from torch import Tensor

from aioway.blocks import Block
from aioway.blocks._typing import TensorNumber
from aioway.schemas import TableSchema

from .buffers import TensorBuffer

__all__ = ["TensorBlock"]


def _unary_op[S: "TensorBlock"](op: Callable[[TensorDict], TensorDict]) -> FunctionType:
    def func(self: S) -> S:
        return type(self)(op(self.data))

    assert isinstance(func, FunctionType)
    return func


def _unary_tensor_op[S: "TensorBlock", E](op: Callable[[Tensor], E]) -> FunctionType:
    def func(self: S) -> S:
        mapping = {key: op(val) for key, val in self.data.items()}
        tensordict = self._tensordict_init(mapping)
        return type(self)(tensordict)

    assert isinstance(func, FunctionType)
    return func


def _binary_op[
    S: "TensorBlock"
](op: Callable[[TensorDict, TensorDict | TensorNumber], TensorDict]) -> FunctionType:
    def func(self: S, other: S | TensorDict | TensorNumber) -> S:
        if not isinstance(other, TensorBlock):
            other_data = other
        else:
            other_data = other.data

        return op(self.data, other_data)

    assert isinstance(func, FunctionType)
    return func


def _binary_tensor_op[
    S: "TensorBlock"
](op: Callable[[Tensor, TensorNumber], Tensor]) -> FunctionType:
    def func(self: S, other: S) -> S:
        other_data = other.data

        if sorted(self.columns) != sorted(other.columns):
            raise KeyError(
                f"TorchBlocks have different keys, element-wise operation not allowed! "
                f"{list(self)=}, {list(other)=}"
            )

        # Note:
        #   Making use of the implementation detail of ``TorchBuffer`` being backed by ``Tensor``.
        #   If this function is to generalize, this shall be changed.
        mapping = {key: op(self[key].data, other[key].data) for key in self}
        tensordict = self._tensordict_init(mapping)
        return type(self)(tensordict)

    assert isinstance(func, FunctionType)
    return func


@typing.final
@dcls.dataclass(frozen=True)
class TensorBlock(Block[TensorBuffer]):
    data: TensorDict
    """
    The underlying data for the ``TorchDataFrame`` class.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDict):
            raise TypeError(
                f"Expected data to be of type TensorDict, got {type(self.data)=}"
            )

    def __contains__(self, key: object) -> bool:
        return key in self.data.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self.data.keys())

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

    _col = _row = __index_no_wrap

    def __index_then_wrap(self, idx) -> Self:
        return type(self)(self.data[idx])

    _select = _slice_of_rows = _list_of_rows = __index_then_wrap

    def map(self, module: Callable[[TensorDict], TensorDict]) -> Self:
        data = self.data.clone()
        return type(self)(module(data))

    def gather(self, dim: int, index: Sequence[int]) -> Self:
        return type(self)(self.data.gather(dim=dim, index=torch.tensor(index).long()))

    def rename(self, **names: str) -> Self:
        return type(self)(self.data.rename(**names))

    def product(self):
        raise NotImplementedError("Join type is not implemented yet!")

    def _project(self, *names: str) -> Self:
        return type(self)(self.data.select(*names))

    def zip(self, other: Self) -> Self:
        if same := set(self.data.keys()) & set(other.data.keys()):
            raise ValueError(f"Cannot concatenate blocks with the same keys: {same}")

        if len(self) != len(other):
            raise ValueError(
                "Cannot concatenate blocks with different lengths: "
                f"{len(self)=} != {len(other)=}"
            )

        if self.batch_size != other.batch_size:
            raise ValueError(
                "Cannot concatenate blocks due to a batch_size mismatch: "
                f"{self.batch_size=} != {other.batch_size=}"
            )

        return type(self)(
            TensorDict(
                {**self.data, **other.data},
                device=self.device,
                batch_size=self.batch_size,
            )
        )

    def join(self, other: Self, on: str) -> Self:
        """
        Todo:
            Right now the implementation is hash-based for simplicity.
            Add support for both hash-based join and comparison-based join.
        """

        if on not in self:
            raise KeyError(f"Lhs does not contain key {on}")

        if on not in other:
            raise KeyError(f"Rhs doesn't contain key {on}")

        # if self.count() == 0 or other.count() == 0:
        #     return type(self)(self._tensordict_init({}, names=self.columns))

        lhs: DefaultDict[Hashable, list[int]] = DefaultDict(list)
        for idx, key in enumerate(self[on]):
            lhs[key.item()].append(idx)

        rhs: DefaultDict[Hashable, list[int]] = DefaultDict(list)
        for idx, key in enumerate(other[on]):
            rhs[key.item()].append(idx)

        common_keys = {*lhs.keys()} & {*rhs.keys()}
        print(common_keys)
        results: list[TensorDict] = []
        for k in common_keys:
            left_idx = lhs[k]
            right_idx = rhs[k]

            left_cnt = len(left_idx)
            right_cnt = len(right_idx)

            left_keys = [left_idx[i] for i in range(left_cnt)] * right_cnt
            right_keys = sum(([right_idx[i]] * left_cnt for i in range(right_cnt)), [])

            left = self.select(left_keys).data
            right = other.select(right_keys).data

            assert (left[on] == right[on]).all().item()

            result = tensordict.merge_tensordicts(left, right)
            results.append(result)

        return type(self)(tensordict.cat(results))

    def union(self, other: Self) -> Self:
        return type(self)(tensordict.cat([self.data, other.data], dim=0))

    def pandas(self) -> DataFrame:
        return DataFrame({key: self[key].to_pandas() for key in self})

    @property
    def dtype(self) -> str | None:
        return str(self.data.dtype) if self.data.dtype is not None else None

    def schema(self) -> TableSchema:
        """
        The schema type from the current block.
        """

        return TableSchema.mapping({key: self[key].dtype for key in self})

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
