# Copyright (c) AIoWay Authors - All Rights Reserved

"Chunk is a heterogenious collection of in-memory tensor batches."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict
import tensordict as td
import torch

from aioway import _typing, tdicts
from aioway._tracking import logging
from aioway.tdicts import attrs

from . import vectors

__all__ = ["Chunk"]

LOGGER = logging.get_logger(__name__)


type TensorDictLike = td.TensorDict | dict[str, torch.Tensor]
type ChunkLike = Chunk | dict[str, vectors.Vector]


@typing.final
@dcls.dataclass(frozen=True)
class Chunk(cabc.Mapping[str, vectors.Vector]):
    """
    A `Chunk` represents a batch of data, following a specific scheam.

    For now, it is immutable as implementing mutable interface is harder.
    This can change in the future.
    """

    data: td.TensorDict
    "The underlying data."

    @typing.override
    def __len__(self) -> int:
        "The length of the current batch."

        return len(self.data)

    @typing.override
    def __repr__(self) -> str:
        return f"{self.attrs!r}({len(self)})"

    @typing.override
    @LOGGER.function("DEBUG")
    def __eq__(self, rhs: object) -> bool:
        # Check if schema and data are both equal.
        if isinstance(rhs, Chunk):
            return self.attrs == rhs.attrs and (self.data == rhs.data).all()

        # If tensordict, don't compare schema.
        if isinstance(rhs, dict | td.TensorDict):
            try:
                return (self.data == rhs).all()
            except KeyError:
                return NotImplemented

        return NotImplemented

    def __getitem__(self, key):
        result = self.fn()[key].do()

        if isinstance(key, str):
            assert isinstance(result, torch.Tensor)
            return vectors.Vector(result)
        else:
            return type(self)(result)

    @typing.override
    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self.attrs)

    def fn(self):
        return tdicts.tdict(self.data)

    @LOGGER.function("DEBUG")
    def select(self, *names: str):
        return Chunk(self.fn().select(*names).do())

    @LOGGER.function("DEBUG")
    def column(self, col: str):
        return vectors.Vector(self.fn()[col].do())

    @LOGGER.function("DEBUG")
    def rename(self, **renames: str):
        if not renames:
            return self

        return Chunk(self.fn().rename(**renames).do())

    @LOGGER.function("DEBUG")
    def zip(self, rhs: typing.Self):
        return Chunk(self.fn().zip(rhs.fn()).do())

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def torch(self):
        return self.data

    @classmethod
    @LOGGER.function("DEBUG")
    def cat(cls, chunks: cabc.Sequence[typing.Self]) -> typing.Self:
        if not chunks:
            raise ValueError("Given an empty sequence. Not sure what to do.")

        if len({tuple(chunk.attrs.devices) for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same devices before joining.")

        if len({tuple(chunk.attrs.dtypes) for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same dtypes before joining.")

        schema = chunks[0].attrs

        data = td.cat([c.data for c in chunks])

        return cls.from_data_schema(schema=schema, data=data)

    @property
    def attrs(self):
        return attrs.attr_set(self.data)

    @classmethod
    def from_data_schema(
        cls, data: TensorDictLike, schema: tdicts.AttrSetLike
    ) -> typing.Self:
        td = _as_tensordict(data)
        td.auto_batch_size_()
        td.auto_device_()
        return cls(data=td)

    @classmethod
    def from_mapping(cls, chunk: ChunkLike) -> typing.Self:
        if isinstance(chunk, cls):
            return chunk

        if _is_mapping_of_vector(chunk):
            data = {key: chunk[key].torch() for key in chunk.keys()}
            schema = {key: chunk[key].typeof() for key in chunk.keys()}
            return cls.from_data_schema(data=data, schema=schema)

        raise TypeError(type(chunk))


def _as_tensordict(data: TensorDictLike, /) -> td.TensorDict:
    if isinstance(data, td.TensorDict):
        return data

    _is_dict_of_tensor = _typing.is_dict_of_str_to(torch.Tensor)
    if _is_dict_of_tensor(data):
        return td.TensorDict(data)

    raise TypeError(type(data))


@typing.no_type_check
def _is_mapping_of_vector(obj) -> typing.TypeGuard[dict[str, vectors.Vector]]:
    # Wrapper function because `mypy` doesn't do well with abstract type guards.
    return _typing.is_dict_of_str_to(vectors.Vector)(obj)
