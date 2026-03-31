# Copyright (c) AIoWay Authors - All Rights Reserved

"Chunk is a heterogenious collection of in-memory tensor batches."

import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict
from tensordict import TensorDict
from torch import Size, Tensor

from aioway import _typing
from aioway._tensor_exprs import SourceTensorDictExpr
from aioway._tracking import logging
from aioway.tdicts import AttrSet, AttrSetLike, _validation

from ..vectors import Vector

__all__ = ["Chunk"]

LOGGER = logging.get_logger(__name__)


type TensorDictLike = TensorDict | dict[str, Tensor]
type ChunkLike = Chunk | dict[str, Vector]


@typing.final
@dcls.dataclass(frozen=True)
class Chunk(cabc.Mapping[str, Vector]):
    """
    A `Chunk` represents a batch of data, following a specific scheam.

    For now, it is immutable as implementing mutable interface is harder.
    This can change in the future.
    """

    data: TensorDict
    "The underlying data."

    attrs: AttrSet
    "The schema for the `Chunk`."

    def __post_init__(self) -> None:
        _validation.validate_schema(self.attrs, self.data)

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
        if isinstance(rhs, dict | TensorDict):
            try:
                return (self.data == rhs).all()
            except KeyError:
                return NotImplemented

        return NotImplemented

    def __getitem__(self, key):
        return self.expr()[key].compute()

    @typing.override
    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self.attrs)

    def expr(self):
        from .exprs import ChunkExpr

        return ChunkExpr(
            tensordict=SourceTensorDictExpr(self.data),
            attrs=self.attrs,
        )

    @LOGGER.function("DEBUG")
    def select(self, *names: str):
        return self.expr().select(*names).compute()

    @LOGGER.function("DEBUG")
    def column(self, col: str):
        return self.expr().column(col).compute()

    @LOGGER.function("DEBUG")
    def rename(self, **renames: str):
        if not renames:
            return self

        return self.expr().rename(**renames).compute()

    @LOGGER.function("DEBUG")
    def zip(self, rhs: typing.Self):
        return self.expr().zip(rhs).compute()

    @property
    def shape(self) -> Size:
        return self.data.shape

    def torch(self):
        return self.data

    @classmethod
    @LOGGER.function("DEBUG")
    def cat(cls, chunks: cabc.Sequence[typing.Self]) -> typing.Self:
        if not chunks:
            raise ValueError("Given an empty sequence. Not sure what to do.")

        if len({chunk.attrs for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same schema before joining.")

        schema = chunks[0].attrs

        data = tensordict.cat([c.data for c in chunks])

        return cls.from_data_schema(schema=schema, data=data)

    @classmethod
    def from_data_schema(cls, data: TensorDictLike, schema: AttrSetLike) -> typing.Self:
        td = _as_tensordict(data)
        td.auto_batch_size_()
        td.auto_device_()
        aset = AttrSet.parse(schema)
        return cls(data=td, attrs=aset)

    @classmethod
    def from_mapping(cls, chunk: ChunkLike) -> typing.Self:
        if isinstance(chunk, cls):
            return chunk

        if _is_mapping_of_vector(chunk):
            data = {key: chunk[key].torch() for key in chunk.keys()}
            schema = {key: chunk[key].typeof() for key in chunk.keys()}
            return cls.from_data_schema(data=data, schema=schema)

        raise TypeError(type(chunk))


def _as_tensordict(data: TensorDictLike, /) -> TensorDict:
    if isinstance(data, TensorDict):
        return data

    _is_dict_of_tensor = _typing.is_dict_of_str_to(Tensor)
    if _is_dict_of_tensor(data):
        return TensorDict(data)

    raise TypeError(type(data))


@typing.no_type_check
def _is_mapping_of_vector(obj) -> typing.TypeGuard[dict[str, Vector]]:
    # Wrapper function because `mypy` doesn't do well with abstract type guards.
    return _typing.is_dict_of_str_to(Vector)(obj)
