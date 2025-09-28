# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from aioway._errors import AiowayError

__all__ = [
    "IndexOp",
    "IndexEq",
    "IndexNe",
    "IndexAnn",
    "IndexGt",
    "IndexGe",
    "IndexLt",
    "IndexLe",
]


class IndexOp(ABC):
    """
    `IndexOp` is the dynamic feature we would want to retrieve from an ``Index``.
    """

    @abc.abstractmethod
    def __str__(self) -> str: ...


@dcls.dataclass(frozen=True, repr=False)
class IndexEq(IndexOp):
    @typing.override
    def __str__(self):
        return "=="


@dcls.dataclass(frozen=True, repr=False)
class IndexNe(IndexOp):
    @typing.override
    def __str__(self):
        return "!="


@dcls.dataclass(frozen=True, repr=False)
class IndexAnn(IndexOp):
    """
    The approximate nearest neighbors for indices.
    """

    k: int
    """
    The top k items to use.
    """

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise IndexReqKError(
                f"Number of nearest neighbors must be a natural number. Got {self.k}"
            )

    @typing.override
    def __str__(self):
        return f"~{self.k}"


@dcls.dataclass(frozen=True, repr=False)
class IndexGt(IndexOp):
    @typing.override
    def __str__(self):
        return "<"


@dcls.dataclass(frozen=True, repr=False)
class IndexGe(IndexOp):
    @typing.override
    def __str__(self):
        return "<="


@dcls.dataclass(frozen=True, repr=False)
class IndexLt(IndexOp):
    @typing.override
    def __str__(self):
        return ">"


@dcls.dataclass(frozen=True, repr=False)
class IndexLe(IndexOp):
    @typing.override
    def __str__(self):
        return ">="


class IndexReqKError(AiowayError, ValueError): ...
