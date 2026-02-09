# Copyright (c) AIoWay Authors - All Rights Reserved

"Chunk is a heterogenious collection of in-memory tensor batches."

import dataclasses as dcls
import logging
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Self, TypeIs

import tensordict
from numpy import ndarray as NpArr
from sympy import Expr
from tensordict import TensorDict
from torch import Size, Tensor

from aioway.attrs import Attr, AttrSet, _validation
from aioway.attrs import funcs as atf

from . import funcs
from .vectors import Vector

__all__ = ["Chunk"]

LOGGER = logging.getLogger(__name__)

type DataLike = TensorDict | dict[str, Tensor]
type SchemaLike = AttrSet | dict[str, Attr]
type ChunkLike = Chunk | dict[str, Vector]


@typing.final
@dcls.dataclass(frozen=True)
class Chunk(Mapping[str, Vector]):
    """
    A ``Chunk`` represents a batch of data, following a specific scheam.

    For now, it is immutable as implementing mutable interface is harder.
    This can change in the future.
    """

    data: TensorDict
    "The underlying data."

    schema: AttrSet
    "The schema for the ``Chunk``."

    def __post_init__(self) -> None:
        _validation.validate_schema(self.schema, self.data)

    @typing.override
    def __len__(self) -> int:
        "The length of the current batch."

        return len(self.data)

    @typing.override
    def __repr__(self) -> str:
        return f"{self.schema!r}({len(self)})"

    @typing.override
    def __eq__(self, rhs: object) -> bool:
        # Check if schema and data are both equal.
        if isinstance(rhs, Chunk):
            return self.schema == rhs.schema and (self.data == rhs.data).all()

        # If tensordict, don't compare schema.
        if isinstance(rhs, dict | TensorDict):
            return (self.data == rhs).all()

        return NotImplemented

    @typing.overload
    def __getitem__(self, idx: str) -> Vector: ...

    @typing.overload
    def __getitem__(
        self, idx: int | slice | list[int] | list[str] | NpArr | Tensor, /
    ) -> Self: ...

    @typing.override
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.col(idx)

        # Doing the batch operations that simply does type check first,
        # to avoid expensive iteration over the input.
        if isinstance(idx, int | slice | NpArr | Tensor):
            return self._getitem_direct(idx)

        if _is_seq_of_str(idx):
            return self.select(*idx)

        if _is_list_of_int(idx):
            return self._getitem_direct(idx)

        raise TypeError(type(idx))

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return iter(self.schema)

    def rename(self, **renames: str) -> Self:
        if not renames:
            return self

        schema = atf.renames(self.schema, **renames)
        data = funcs.rename(self.data, **renames)
        return self.from_data_schema(data=data, schema=schema)

    def filter(self, expr: str | Expr) -> Self:
        return self.from_data_schema(
            data=funcs.filter(self.data, expr), schema=self.schema
        )

    def zip(self, rhs: Self) -> Self:
        data = tensordict.merge_tensordicts(self.data, rhs.data)
        schema = {**self.schema, **rhs.schema}
        return self.from_data_schema(data=data, schema=schema)

    def col(self, key: str) -> Vector:
        tensor = self.data[key]
        attr = self.schema[key]

        # No need to validate as it's already verified in ``Chunk.__post_init__``.
        return Vector(data=tensor, attr=attr, validate=False)

    def select(self, *keys: str) -> Self:
        data = self.data.select(*keys)
        schema = {key: self.schema[key] for key in keys}
        return self.from_data_schema(data=data, schema=schema)

    def _getitem_direct(self, idx: int | slice | list[int] | NpArr | Tensor) -> Self:
        return self.from_data_schema(
            data=self.data[idx], schema=atf.index(self.schema, idx)
        )

    @property
    def shape(self) -> Size:
        return self.data.shape

    def torch(self):
        return self.data

    @classmethod
    def cat(cls, chunks: Sequence[Self]) -> Self:
        if not chunks:
            raise ValueError("Given an empty sequence. Not sure what to do.")

        if len({chunk.schema for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same schema before joining.")

        schema = chunks[0].schema

        data = tensordict.cat([c.data for c in chunks])

        return cls(schema=schema, data=data)

    @classmethod
    def from_data_schema(cls, data: DataLike, schema: SchemaLike) -> Self:
        td = _as_tensordict(data)
        td.auto_batch_size_()
        td.auto_device_()
        attrs = _as_schema(schema)
        return cls(data=td, schema=attrs)

    @classmethod
    def from_mapping(cls, chunk: ChunkLike) -> Self:
        if isinstance(chunk, cls):
            return chunk

        if _is_mapping_of_vector(chunk):
            data = {key: chunk[key].data for key in chunk.keys()}
            schema = {key: chunk[key].attr for key in chunk.keys()}
            return cls.from_data_schema(data=data, schema=schema)

        raise TypeError


def parse_chunk(*, data: DataLike, schema: SchemaLike) -> Chunk:
    return Chunk.from_data_schema(data=data, schema=schema)


def _as_tensordict(data: DataLike, /) -> TensorDict:
    if isinstance(data, TensorDict):
        return data

    if _is_dict_of_tensor(data):
        return TensorDict(data)

    raise TypeError(
        f"Unknown: {type(data)=}. Only accepts `TensorDict` or `dict[str, Tensor]`."
    )


def _as_schema(schema: SchemaLike, /) -> AttrSet:
    if isinstance(schema, AttrSet):
        return schema

    if _is_dict_of_attr(schema):
        return AttrSet.from_dict(schema)

    raise TypeError(
        f"Unknown: {type(schema)=}. Only accepts `AttrSet` or `dict[str, Attr]`."
    )


def _is_dict_of_tensor(obj) -> TypeIs[dict[str, Tensor]]:
    return (
        True
        and isinstance(obj, dict)
        and all(isinstance(key, str) for key in obj.keys())
        and all(isinstance(val, Tensor) for val in obj.values())
    )


def _is_mapping_of_vector(obj) -> TypeIs[Mapping[str, Vector]]:
    return (
        True
        and isinstance(obj, Mapping)
        and all(isinstance(key, str) for key in obj.keys())
        and all(isinstance(val, Vector) for val in obj.values())
    )


def _is_dict_of_attr(obj) -> TypeIs[dict[str, Attr]]:
    return (
        True
        and isinstance(obj, dict)
        and all(isinstance(key, str) for key in obj.keys())
        and all(isinstance(val, Attr) for val in obj.values())
    )


def _is_seq_of_str(obj) -> TypeIs[Sequence[str]]:
    return isinstance(obj, Sequence) and all(isinstance(i, str) for i in obj)


def _is_list_of_int(obj) -> TypeIs[list[int]]:
    return isinstance(obj, list) and all(isinstance(i, int) for i in obj)
