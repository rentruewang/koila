# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections.abc import Callable, Generator, Mapping, Sequence
from numbers import Number
from typing import Self

import numpy as np
import tensordict
import torch
from pandas import DataFrame
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from aioway.schemas import TableSchema

from .buffers import Buffer

__all__ = ["Block"]


@dcls.dataclass(frozen=True)
class Block(Mapping):
    """
    ``Block`` represents a chunk / batch of data stored in memory.
    This is the core representation of in-memory data in ``aioway``.
    """

    data: TensorDict
    """
    The underlying data for the ``TorchDataFrame`` class.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDict):
            raise TypeError(
                f"Expected data to be of type TensorDict, got {type(self.data)=}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        return key in self.data

    @typing.overload
    def __getitem__(self, key: str) -> Buffer: ...

    @typing.overload
    def __getitem__(self, key: int | list[int] | slice) -> Self: ...

    def __getitem__(self, key: str | int | list[int] | slice) -> Self | Buffer:
        result = self.data[key]

        if isinstance(key, str):
            return Buffer(result)

        return type(self)(result)

    def __iter__(self):
        return iter(self.data)

    def __str__(self) -> str:
        return str(dict(self.data.items()))

    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        return self.__bin_op(other, operator.eq)

    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        return self.__bin_op(other, operator.ne)

    def __add__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.add)

    def __sub__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.sub)

    def __mul__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.mul)

    def __truediv__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.truediv)

    def __bin_op(
        self,
        other: Self | TensorDict | Number,
        op: Callable[[TensorDict, TensorDict | Number], TensorDict],
    ) -> Self:
        # Usually, the ``numpy`` family's arithmetic operations would work with python number,
        # therefore, they are included in the type checks for implemented methods.
        if not isinstance(other, (type(self), TensorDict, Number)):
            return NotImplemented

        if not callable(op):
            raise ValueError(
                f"Operation {op} is not callable. Must be a callable with 2 arguments."
            )

        # We want to leverage the existing operators implemented by ``TensorDict``,
        # since they support built-in types and other ``TensorDict``s, but not our types.
        if not isinstance(other, Block):
            operand = other
        else:
            operand = other.data

        return type(self)(op(self.data, operand))

    def keys(self) -> Generator[str, None, None]:  # type: ignore[override]
        yield from self.data.keys()

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def tensordict_map(self, module: TensorDictModule) -> Self:
        data = self.data.clone()
        return type(self)(module(data))

    def all(self) -> bool:
        return self.data.all()

    def any(self) -> bool:
        return self.data.any()

    def gather(self, dim: int, index: Sequence[int]) -> Self:
        return type(self)(self.data.gather(dim=dim, index=torch.tensor(index).long()))

    def rename(self, **names: str) -> Self:
        return type(self)(self.data.rename(**names))

    def product(self):
        raise NotImplementedError("Join type is not implemented yet!")

    def project(self, *names: str) -> Self:
        return type(self)(self.data.select(*names))

    def select(self, index: Sequence[bool]) -> Self:
        return type(self)(self.data[torch.tensor(index).bool()])

    def concat(self, other: Self) -> Self:
        if keys := set(self.keys()) & set(other.keys()):
            raise ValueError(f"Cannot concatenate blocks with the same keys: {keys}")

        if len(self) != len(other):
            raise ValueError(
                f"Cannot concatenate blocks: {len(self)=} != {len(other)=}"
            )

        return type(self)({**self.data, **other.data})

    def union(self, other: Self) -> Self:
        return type(self)(tensordict.cat([self.data, other.data], dim=0))

    def pandas(self) -> DataFrame:
        return DataFrame({key: val.series() for key, val in self.items()})

    @property
    def dtype(self) -> str | None:
        return str(self.data.dtype) if self.data.dtype is not None else None

    @property
    def device(self) -> str | None:
        return str(self.data.device) if self.data.device is not None else None

    @property
    def schema(self) -> TableSchema:
        """
        The schema type from the current block.
        """

        return TableSchema.mapping({key: val.datatype for key, val in self.items()})

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.data.batch_size

    @property
    def requires_grad(self) -> bool:
        return self.data.requires_grad

    @classmethod
    def from_pandas(cls, df: DataFrame, /) -> Self:
        dicts = df.to_dict("series")
        np_dicts = {str(key): np.array(val) for key, val in dicts.items()}
        torch_dicts = {key: val for key, val in np_dicts.items()}
        return cls(TensorDict(torch_dicts))
